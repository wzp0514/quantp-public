"""
LLM 因子挖掘 — CogAlpha 的实用替代方案

CogAlpha 是一个多智能体 LLM 因子挖掘学术框架，需要复杂的 Agent 编排。
这里做的是实用简化版：DeepSeek API → 提因子公式 → IC 验证 → 好的留下。

原理：
  1. 把现有因子和IC结果发给 DeepSeek
  2. DeepSeek 基于金融领域知识提出新的因子公式
  3. 程序自动计算新因子的 IC
  4. IC > 0.03 的因子留下
  5. 循环迭代

成本：DeepSeek API 约 ¥1/百万 token，每轮对话约 2000 token = ¥0.002。便宜到几乎免费。

前置条件：
  1. 注册 DeepSeek API: https://platform.deepseek.com/
  2. 获取 API Key
  3. 填入 config/settings.local.yaml:
     llm:
       provider: "deepseek"
       api_key: "sk-xxx"
       model: "deepseek-chat"  # 或 deepseek-reasoner

用法
--------
>>> from backtest.analysis.llm_factor_miner import LLMFactorMiner
>>> miner = LLMFactorMiner(df)
>>> new_factors = miner.iterate(rounds=3)  # 跑 3 轮
>>> for f in new_factors:
...     print(f"{f['name']}: IC={f['ic']:.4f}")
"""

import ast
import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保能从项目根目录导入 config 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("llm_factor_miner")


# ── AST 白名单安全校验（防止 exec() 执行恶意代码）──

_SAFE_NODES = {
    ast.Expression, ast.Module, ast.Expr,
    ast.Constant, ast.Name, ast.Load, ast.Store,
    ast.BinOp, ast.UnaryOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
    ast.BoolOp, ast.And, ast.Or,
    ast.Subscript, ast.Slice, ast.Tuple, ast.List,
    ast.IfExp, ast.Attribute,
    ast.keyword,
}

_SAFE_CALLABLES = {
    # pandas Series/DataFrame methods
    "rolling", "pct_change", "shift", "mean", "std", "var", "sum", "min", "max",
    "abs", "ewm", "expanding", "rank", "diff", "fillna", "dropna", "clip",
    "rolling_mean", "rolling_std", "corr", "cov", "cumsum", "cumprod",
    "quantile", "iloc", "loc", "values", "index", "columns",
    "copy", "astype", "replace", "where", "mask",
    # numpy functions
    "log", "log10", "log2", "exp", "sqrt", "abs", "sign", "sin", "cos",
    "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "isnan", "isfinite", "isinf",
    "ones", "zeros", "full", "arange", "linspace",
    "nanmean", "nanstd", "nanpercentile", "nanmedian", "nansum",
    "dot", "power", "divide", "multiply", "add", "subtract",
    "concatenate", "vstack", "hstack",
    # builtins
    "abs", "min", "max", "round", "len", "int", "float", "bool", "str",
    "list", "dict", "tuple", "set", "range", "zip", "enumerate", "sum",
}


def _validate_formula_ast(formula: str) -> bool:
    """AST 白名单校验：只允许数学运算和安全的 pandas/numpy 调用。"""
    try:
        tree = ast.parse(formula.strip(), mode="exec")
    except SyntaxError:
        logger.warning(f"Formula syntax error, rejected: {formula[:80]}")
        return False

    for node in ast.walk(tree):
        node_type = type(node)

        if node_type in _SAFE_NODES:
            continue

        if node_type is ast.Call:
            # 只允许白名单内的函数调用
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name not in _SAFE_CALLABLES:
                logger.warning(f"Blocked function call: {func_name}")
                return False
            continue

        # 禁止 import, exec, eval, lambda, 函数定义, 类定义 等
        forbidden = {
            ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
            ast.ClassDef, ast.Lambda, ast.GeneratorExp, ast.ListComp,
            ast.SetComp, ast.DictComp, ast.Yield, ast.YieldFrom,
            ast.Await, ast.AsyncFor, ast.AsyncWith,
            ast.Global, ast.Nonlocal, ast.Delete, ast.Raise, ast.Try,
            ast.Assert, ast.With, ast.withitem, ast.Pass, ast.Break, ast.Continue,
        }
        if node_type in forbidden:
            logger.warning(f"Blocked unsafe AST node: {node_type.__name__}")
            return False

        # 未知节点类型也拒绝
        logger.warning(f"Unknown AST node rejected: {node_type.__name__}")
        return False

    return True
def _load_llm_config() -> dict:
    """从配置文件读取 LLM API 设置（通过统一配置加载器）"""
    from config.loader import get_llm_config
    return get_llm_config()


def _call_deepseek(prompt: str, system_prompt: str = "") -> Optional[str]:
    """调用 DeepSeek API"""
    cfg = _load_llm_config()
    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "deepseek-chat")

    if not api_key or api_key == "你的DeepSeek API Key":
        logger.warning("DeepSeek API Key 未配置。请在 settings.local.yaml 中填写: llm.api_key")
        return None

    url = "https://api.deepseek.com/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4000,
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            # DeepSeek v4-pro 等 reasoning 模型可能在 reasoning_content 字段
            content = msg.get("content", "") or msg.get("reasoning_content", "")
            if not content:
                logger.warning(f"DeepSeek 返回空内容。可用字段: {list(msg.keys())}")
                logger.debug(f"完整响应: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}")
            else:
                logger.info(f"DeepSeek 响应: {len(content)} 字符")
            return content
    except Exception as e:
        logger.error(f"DeepSeek API 调用失败: {e}")
        return None


class LLMFactorMiner:
    """
    LLM 因子挖掘器。

    用法
    --------
    >>> miner = LLMFactorMiner(df)
    >>> new_factors = miner.iterate(rounds=3)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df["close"]
        self.volume = df.get("volume", pd.Series())
        self.high = df.get("high", pd.Series())
        self.low = df.get("low", pd.Series())

        # 已有因子（先跑一遍传统因子挖掘）
        from backtest.analysis.factor_miner import compute_factors
        self.factor_df = compute_factors(df)
        self.discovered_factors: list[dict] = []

        # 先获取传统因子的 IC 作为 baseline
        from backtest.analysis.factor_miner import compute_ic
        self.existing_ics = {}
        factor_cols = [c for c in self.factor_df.columns
                       if c not in ("date","open","high","low","close","volume")]
        for col in factor_cols:
            ic_result = compute_ic(self.factor_df, col)
            self.existing_ics[col] = ic_result["ic"]

    def iterate(self, rounds: int = 3) -> list[dict]:
        """
        多轮迭代挖掘新因子。

        每轮：发给 DeepSeek → 收到因子公式 → 验证 IC → 保留好的 →
        下一轮把好的也发过去，让它在此基础上继续想。
        """
        all_new = []

        for r in range(rounds):
            logger.info(f"LLM 因子挖掘 第 {r+1}/{rounds} 轮...")

            # 构建 prompt
            prompt = self._build_prompt(all_new)

            # 调 DeepSeek
            response = _call_deepseek(prompt, self._system_prompt())
            if response is None:
                cfg = _load_llm_config()
                if not cfg.get("api_key") or cfg["api_key"] == "你的DeepSeek API Key":
                    logger.warning("DeepSeek API Key 未配置，跳过 LLM 因子挖掘。请在 settings.local.yaml 的 llm.api_key 中配置。")
                else:
                    logger.warning("API 调用失败（可能是网络问题），停止迭代")
                break

            # 解析因子公式
            new_factors = self._parse_factors(response)
            logger.info(f"  提取到 {len(new_factors)} 个候选因子")

            # 逐个验证
            for f in new_factors:
                ic_result = self._validate_factor(f)
                if ic_result and abs(ic_result["ic"]) > 0.03:
                    f["ic"] = ic_result["ic"]
                    f["ic_ir"] = ic_result.get("ic_ir", 0)
                    f["round"] = r + 1
                    self.discovered_factors.append(f)
                    all_new.append(f)
                    logger.info(f"  [OK] {f['name']}: IC={ic_result['ic']:+.4f}")

        # 按 |IC| 排序
        all_new.sort(key=lambda x: abs(x["ic"]), reverse=True)
        return all_new

    def _build_prompt(self, prev_new: list[dict]) -> str:
        """构建发送给 DeepSeek 的 prompt"""
        # 现有因子 top 5
        sorted_ics = sorted(self.existing_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        top5_text = "\n".join(
            f"  {name}: IC={ic:+.4f}"
            for name, ic in sorted_ics[:5]
        )

        # 新发现因子
        new_text = ""
        if prev_new:
            new_text = "\n已发现的新因子:\n" + "\n".join(
                f"  {f['name']}: IC={f['ic']:+.4f} — {f.get('description','')}"
                for f in prev_new[:5]
            )

        # 数据统计
        stats = f"数据: {len(self.df)} 条日线, 收盘价均值 {self.close.mean():.2f}"

        return f"""你是一位量化金融研究员。请基于以下信息，提出 3-5 个新的技术因子公式。

{stats}

现有因子的预测能力（IC，越大越好）：
{top5_text}
{new_text}

要求：
1. 每个因子给出 Python/pandas 计算公式（可直接执行的代码）
2. 因子要有金融逻辑支持（简单解释为什么这个因子可能有效）
3. 不要重复已有因子
4. 优先考虑：波动率结构、成交量价格关系、趋势强度、反转信号、市场微观结构

请按以下 JSON 格式输出（直接输出 JSON，不要其他文字）：
```json
[
  {{
    "name": "因子名",
    "formula": "df['因子名'] = 具体pandas计算表达式",
    "description": "一句话解释金融逻辑"
  }},
  ...
]
```"""

    def _system_prompt(self) -> str:
        return "你是一位量化金融研究员，专长于A股市场的技术因子挖掘。只用JSON格式回复。"

    def _parse_factors(self, response: str) -> list[dict]:
        """从 DeepSeek 响应中提取因子列表"""
        # 提取 JSON 块
        import re
        match = re.search(r'```json\s*([\s\S]*?)```', response)
        if not match:
            match = re.search(r'\[[\s\S]*\]', response)

        if match:
            try:
                factors = json.loads(match.group(1) if match.lastindex else match.group(0))
                return factors if isinstance(factors, list) else []
            except json.JSONDecodeError:
                pass

        # 尝试直接解析
        try:
            factors = json.loads(response)
            return factors if isinstance(factors, list) else []
        except json.JSONDecodeError:
            pass

        logger.warning(f"无法解析 DeepSeek 响应: {response[:200]}...")
        return []

    def _validate_factor(self, factor: dict) -> Optional[dict]:
        """执行因子公式并计算 IC"""
        formula = factor.get("formula", "")
        if not formula:
            return None

        # AST 白名单安全校验
        if not _validate_formula_ast(formula):
            logger.warning(f"Factor formula rejected by AST validator: {factor.get('name', '?')}")
            return None

        # 安全执行（受限命名空间 + AST 白名单）
        namespace = {
            "df": self.df.copy(),
            "pd": pd,
            "np": np,
            "__builtins__": {},
        }

        try:
            exec(formula, namespace)
        except Exception as e:
            logger.debug(f"因子公式执行失败: {factor.get('name','?')} — {e}")
            return None

        # 找到新因子列
        new_cols = set(namespace["df"].columns) - set(self.df.columns)
        if not new_cols:
            return None

        factor_col = list(new_cols)[0]
        df_with_factor = namespace["df"]

        from backtest.analysis.factor_miner import compute_ic
        ic_result = compute_ic(df_with_factor, factor_col)
        return ic_result


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/llm_factor_miner.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    print("=" * 60)
    print("LLM 因子挖掘 (CogAlpha 替代)")
    print("=" * 60)

    cfg = _load_llm_config()
    if not cfg.get("api_key") or cfg["api_key"] == "你的DeepSeek API Key":
        print("\nDeepSeek API Key 未配置。")
        print("1. 注册 https://platform.deepseek.com/")
        print("2. 在 config/settings.local.yaml 添加:")
        print("   llm:")
        print("     provider: deepseek")
        print("     api_key: sk-xxx")
        print("     model: deepseek-chat")
        print("\n成本: 约 ¥0.002/轮，几乎免费。")
    else:
        print(f"API Key: {cfg['api_key'][:10]}...")
        df = fetch_index_daily("沪深300", "20230101", "20250601")
        miner = LLMFactorMiner(df)
        results = miner.iterate(rounds=2)
        print(f"\n发现 {len(results)} 个有效因子:")
        for f in results:
            print(f"  {f['name']}: IC={f['ic']:+.4f} | {f.get('description','')[:60]}")
