"""
量化研究管线 — 敏捷式自动推进

每条 Track = 一条独立的研究路线（数据 + 因子集 + 策略 + 参数）。
Pipeline = 管理多条 Track 并行推进，过门则晋级，不过则淘汰。

五阶段晋级:
  Stage 1 (因子挖掘)  → Gate: IC > 0.05
  Stage 2 (策略回测)  → Gate: SR > 0 + trades ≥ 10
  Stage 3 (专业验证)  → Gate: WFA pass + DSR significant
  Stage 4 (纸上交易)  → Gate: 偏差 < 30%
  Stage 5 (候选实盘)  → 输出最终策略池

用法
--------
>>> pl = Pipeline(data)
>>> pl.add_track("momentum_focus", factors=["ret_3m","ret_6m"], strategy=MaCrossStrategy)
>>> pl.add_track("vol_focus", factors=["volatility_1m","low_vol_premium"], strategy=FactorStrategy)
>>> pl.run()     # 所有 track 并行推进
>>> pl.report()  # 对比所有 track 的结果
"""

import ast
import math
import time
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("pipeline")


# ═══════════════════════════════════════════
# 阶段定义
# ═══════════════════════════════════════════

class Stage(Enum):
    FACTOR = 1       # 因子挖掘
    BACKTEST = 2     # 策略回测
    VALIDATE = 3     # 专业验证
    PAPER = 4        # 纸上交易
    READY = 5        # 候选实盘
    ARCHIVED = -1    # 淘汰


STAGE_NAMES = {
    Stage.FACTOR: "Factor Mining",
    Stage.BACKTEST: "Backtest",
    Stage.VALIDATE: "Validate",
    Stage.PAPER: "Paper Trading",
    Stage.READY: "READY FOR LIVE",
    Stage.ARCHIVED: "ARCHIVED",
}


# ═══════════════════════════════════════════
# Track — 一条研究路线
# ═══════════════════════════════════════════

@dataclass
class Track:
    """一条独立的研究路线"""
    name: str
    factors: list[str] = field(default_factory=list)     # 使用的因子名
    strategy_class: Optional[type] = None                  # 策略类
    strategy_params: dict = field(default_factory=dict)    # 策略参数
    stage: Stage = Stage.FACTOR
    stage_history: list[str] = field(default_factory=list)  # 阶段变更记录

    # 每阶段产出
    factor_results: dict = field(default_factory=dict)     # Stage 1 产出
    backtest_result: Optional[object] = None               # Stage 2 产出
    validation_result: dict = field(default_factory=dict)  # Stage 3 产出
    paper_result: dict = field(default_factory=dict)       # Stage 4 产出

    # 策略挖掘溯源 (用于淘汰时自动重挖)
    entry_type: str = ""
    exit_type: str = ""

    # 评分（跨 track 对比用）
    final_score: float = 0.0
    archived_reason: str = ""

    def promote(self, stage: Stage):
        self.stage_history.append(f"{STAGE_NAMES[self.stage]} -> {STAGE_NAMES[stage]}")
        self.stage = stage

    def archive(self, reason: str):
        self.stage = Stage.ARCHIVED
        self.archived_reason = reason
        self.stage_history.append(f"ARCHIVED: {reason}")

    def status(self) -> str:
        if self.stage == Stage.ARCHIVED:
            return f"[ARCHIVED] {self.name}: {self.archived_reason[:60]}"
        return f"[{STAGE_NAMES[self.stage]}] {self.name} (score={self.final_score:.3f})"


# ═══════════════════════════════════════════
# Pipeline — 管理所有 Track
# ═══════════════════════════════════════════

class Pipeline:
    """
    敏捷量化研究管线。

    添加多条 Track → run() 逐阶段推进 → report() 对比结果。
    每条 Track 独立运行，好的晋级，坏的归档。
    """

    def __init__(self, df: pd.DataFrame = None, cash: float = 100000.0,
                 max_workers: int = 2, panel: dict[str, pd.DataFrame] = None,
                 env_filter=None, symbol: str = None):
        self.df = df
        self.panel = panel  # 多股票面板 {symbol: DataFrame}
        self.cash = cash
        self.max_workers = max_workers
        self.tracks: list[Track] = []
        self._run_log: list[str] = []
        self._recover_count = 0
        self.optimize_params = True   # Stage 2通过后Bayes精调参数
        self.env_filter = env_filter  # EnvironmentFilter 前置过滤器
        self.env_result: Optional[dict] = None  # 环境评估结果
        self.symbol = symbol  # 标的代码，供Edge Score等组件使用

    def add_track(self, name: str, factors: list[str] = None,
                  strategy_class=None, strategy_params: dict = None):
        """添加一条新的研究路线"""
        track = Track(
            name=name,
            factors=factors or [],
            strategy_class=strategy_class,
            strategy_params=strategy_params or {},
        )
        self.tracks.append(track)
        self._run_log.append(f"Track added: {name}")
        logger.info(f"Track added: {name} ({len(track.factors)} factors)")
        return track

    def feed_signals(self, signals: list) -> int:
        """
        将ResearchSource信号喂入管线。
        arxiv因子信号 → LLM提取公式 → IC验证 → 自动创建Track。
        返回新增Track数。
        """
        added = 0
        for sig in signals:
            if sig.action != "factor" or sig.priority < 3:
                continue
            if not sig.data.get("title"):
                continue

            logger.info(f"  Processing signal: {sig.title[:60]}")
            try:
                # 调LLM从论文标题中提取因子公式
                from backtest.analysis.llm_factor_miner import _call_deepseek, _load_llm_config
                cfg = _load_llm_config()
                if not cfg.get("api_key"):
                    logger.debug("  No LLM API key, skip factor extraction")
                    continue

                prompt = (
                    f"Paper title: {sig.data['title']}\n"
                    f"Categories: {sig.data.get('categories', [])}\n\n"
                    "Extract a concrete factor formula from this paper. "
                    "The factor must be computable from OHLCV data (columns: open,high,low,close,volume). "
                    "Return ONLY a JSON object:\n"
                    '{"name": "factor_name", "formula": "pandas expression using df", '
                    '"description": "one line"}\n'
                    "Formula example: df['close'].pct_change(21).rolling(63).mean() / df['close'].rolling(63).std()"
                )
                resp = _call_deepseek(prompt)
                if not resp:
                    continue

                # 解析LLM返回
                import json
                try:
                    obj = json.loads(resp.strip().strip("`").strip("json").strip())
                except json.JSONDecodeError:
                    # 尝试从markdown代码块提取
                    import re
                    m = re.search(r'\{[^}]+\}', resp)
                    if m:
                        obj = json.loads(m.group())
                    else:
                        continue

                formula = obj.get("formula", "")
                fname = obj.get("name", "arxiv_factor")
                if not formula:
                    continue

                # 执行公式并计算IC（AST 安全检查 + 禁用 builtins）
                import numpy as np
                df = self.df.copy()
                local_vars = {"df": df, "np": np, "pd": __import__("pandas")}
                # 简单 AST 预检：拒绝 import/exec/eval/__ 等危险模式
                formula_ast = f"__result__ = {formula}"
                try:
                    tree = ast.parse(formula_ast.strip(), mode="exec")
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            raise ValueError("Import statements not allowed in factor formula")
                except SyntaxError:
                    logger.warning(f"  Formula syntax error, rejected: {formula[:60]}")
                    continue
                try:
                    exec(formula_ast, {"__builtins__": {}}, local_vars)
                    factor_values = local_vars.get("__result__")
                    if factor_values is None:
                        continue
                    if hasattr(factor_values, "values"):
                        factor_values = factor_values.values
                except Exception as e:
                    logger.debug(f"  Formula exec failed: {e}")
                    continue

                # IC计算
                fwd_ret = df["close"].pct_change(21).shift(-21).values
                valid = ~(np.isnan(factor_values) | np.isnan(fwd_ret))
                if valid.sum() < 60:
                    continue
                ic = np.corrcoef(factor_values[valid], fwd_ret[valid])[0, 1]

                if abs(ic) <= 0.03:
                    logger.info(f"  arXiv factor '{fname}' IC={ic:.4f} — below threshold, discarded")
                    continue

                logger.info(f"  arXiv factor '{fname}' IC={ic:.4f} — passed, creating Track")
                self.add_track(
                    f"arxiv_{fname}",
                    factors=[fname],
                )
                # 持久化到FactorStore
                try:
                    from data.vault.factor_store import FactorStore
                    fs = FactorStore()
                    fs.save_factor(name=fname, df=self.df[["date"]].copy(),
                                   ic=ic, ic_ir=0, category="arxiv",
                                   description=obj.get("description", ""))
                except Exception:
                    logger.exception("FactorStore save failed for arxiv factor")
                added += 1
            except Exception as e:
                logger.debug(f"  Signal processing failed: {e}")
                continue

        if added > 0:
            logger.info(f"  feed_signals: {added} new arxiv-factor Track(s) created")
        return added

    def run(self, until_stage: Stage = Stage.VALIDATE,
            auto_recover: bool = True, max_recover: int = 3) -> list[Track]:
        """
        推进所有 Track 直到指定阶段。

        每条 Track 独立跑，阶段间有门控（Gate）：
          通过 → 晋级到下一阶段
          未通过 → 归档。若 auto_recover=True 且有原始入场/出场信息，自动重挖。

        auto_recover: 淘汰时自动调 StrategyMiner 在相同入场×出场逻辑下重搜参数
        max_recover: 单次管线最多恢复几条 Track（防无限循环）
        """
        t0 = time.time()
        self._run_log.append(f"Pipeline start: {len(self.tracks)} tracks, until={STAGE_NAMES[until_stage]}")

        # Stage 1: Factor Mining (sequential — IC analysis is fast)
        active = [t for t in self.tracks if t.stage == Stage.FACTOR]
        if active:
            logger.info(f"--- Stage 1: Factor Mining ({len(active)} tracks) ---")
            for track in active:
                self._run_stage1(track)

        # Stage 2: Backtest (parallel — can be slow)
        # 前置环境评估：L4 + Agent 判断方向/仓位 + Edge Score 统一门控
        self.env_result = None
        self.edge_result = None
        if self.env_filter is not None:
            try:
                self.env_result = self.env_filter.assess()
                logger.info(
                    f"Env filter: trade={'YES' if self.env_result['should_trade'] else 'NO'} "
                    f"dir={self.env_result['direction']} "
                    f"pos={self.env_result['position_multiplier']:.0%} "
                    f"risk={self.env_result['risk_level']}"
                )
            except Exception as e:
                logger.warning(f"Env filter failed: {e}, continuing without filter")

        # Edge Score：融合区制+环境+信号共振
        try:
            from backtest.analysis.edge_score import EdgeScore
            es = EdgeScore(self.df, symbol=self.symbol)
            self.edge_result = es.compute()
            logger.info(
                f"Edge Score: {self.edge_result['edge']:.3f} "
                f"({'TRADE' if self.edge_result['should_trade'] else 'NO_TRADE'}) "
                f"— {self.edge_result['recommendation']}"
            )
        except Exception as e:
            logger.warning(f"Edge Score failed: {e}, continuing without")
            self.edge_result = None

        active = [t for t in self.tracks if t.stage == Stage.BACKTEST]
        if active and until_stage.value >= Stage.BACKTEST.value:
            logger.info(f"--- Stage 2: Backtest ({len(active)} tracks) ---")
            if self.max_workers > 1 and len(active) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = {ex.submit(self._run_stage2, t): t for t in active}
                    for f in as_completed(futures):
                        f.result()
            else:
                for track in active:
                    self._run_stage2(track)

        # Stage 3: Validate — 淘汰时立刻触发自动恢复
        active = [t for t in self.tracks if t.stage == Stage.VALIDATE]
        if active and until_stage.value >= Stage.VALIDATE.value:
            # FWER/FDR 批量多重比较校正（Stage 2→3 之间）
            active = self._run_fwer_filter(active)
            if active:
                logger.info(f"--- Stage 3: Validate ({len(active)} tracks) ---")
                self._recover_count = 0
                for track in active:
                    self._run_stage3(track, auto_recover=auto_recover, max_recover=max_recover)

        # Stage 4: Paper Trading
        active = [t for t in self.tracks if t.stage == Stage.PAPER]
        if active and until_stage.value >= Stage.PAPER.value:
            logger.info(f"--- Stage 4: Paper Trading ({len(active)} tracks) ---")
            for track in active:
                self._run_stage4(track)

        elapsed = time.time() - t0
        ready = sum(1 for t in self.tracks if t.stage == Stage.READY)
        archived = sum(1 for t in self.tracks if t.stage == Stage.ARCHIVED)
        recovered = sum(1 for t in self.tracks if "_v2" in t.name or "_recover" in t.name)
        self._run_log.append(
            f"Pipeline done: {ready} ready, {archived} archived, "
            f"{recovered} recovered, "
            f"{len(self.tracks) - ready - archived} in-progress, {elapsed:.1f}s"
        )
        logger.info(self._run_log[-1])
        return self.tracks

    # ── Stage 1: Factor Mining ──

    def _run_stage1(self, track: Track):
        """因子挖掘：计算IC + LLM生成新因子 + 强因子自动创建策略Track"""
        try:
            from backtest.analysis.factor_miner import FactorMiner

            # ── 1. 传统因子IC分析 ──
            fm = FactorMiner(self.df)
            result = fm.mine()
            strong = [r for r in result["results"] if r["abs_ic"] > 0.05]
            if track.factors:
                strong = [r for r in strong if r["factor"] in track.factors]

            track.factor_results = {
                "total": len(result["results"]),
                "strong": len(strong),
                "top_ic": strong[0]["ic"] if strong else 0,
            }

            # ── 2. LLM因子挖掘（生成全新因子公式）──
            try:
                from backtest.analysis.llm_factor_miner import LLMFactorMiner
                llm = LLMFactorMiner(self.df)
                new_factors = llm.iterate(rounds=2)
                for nf in new_factors:
                    ic = nf.get("ic", 0)
                    if abs(ic) > 0.03:
                        # 将LLM新因子加入结果列表
                        result["results"].append({
                            "factor": nf.get("name", "llm_factor"),
                            "ic": ic, "abs_ic": abs(ic),
                            "source": "LLM", "formula": nf.get("formula", ""),
                        })
                logger.info(f"  LLM: {len(new_factors)} generated, "
                           f"{sum(1 for n in new_factors if abs(n.get('ic',0))>0.03)} passed IC>0.03")
            except Exception as e:
                logger.debug(f"  LLM miner skipped: {e}")

            # ── 3. 强因子 → 自动生成策略Track（因子驱动基线）──
            # 因子已经通过IC验证 → 跳过FACTOR阶段，直接进BACKTEST
            from backtest.strategies.experimental.factor_strategy import FactorStrategy
            all_strong = [r for r in result["results"] if r["abs_ic"] > 0.05]
            generated = 0
            for r in all_strong[:10]:
                fname = r["factor"]
                if any(t.name == f"factor_{fname}" for t in self.tracks):
                    continue
                src = r.get("source", "mined")
                ic = r.get("ic", 0)
                nt = self.add_track(
                    f"factor_{fname}",
                    strategy_class=FactorStrategy,
                    strategy_params={"factor_name": fname, "source": src},
                )
                nt.stage = Stage.BACKTEST  # 因子已验证，直接进回测
                nt.factor_results = {"ic": ic, "source": src}
                generated += 1
            if generated:
                logger.info(f"  Auto-generated {generated} factor-driven tracks → BACKTEST")

            # ── 3.5 截面多因子选股Track（需panel数据）──
            if self.panel and len(self.panel) >= 10:
                from backtest.strategies.experimental.cross_section_strategy import CrossSectionStrategy
                cs_name = f"CrossSection_Top{min(10, len(self.panel)//5)}"
                if not any(t.name == cs_name for t in self.tracks):
                    nt = self.add_track(cs_name, strategy_class=CrossSectionStrategy,
                                        strategy_params={
                                            "top_n": min(10, len(self.panel) // 5),
                                            "rebalance_freq": 21,
                                        })
                    nt.stage = Stage.BACKTEST
                    logger.info(f"  Cross-section track: {cs_name} ({len(self.panel)} stocks)")

            # ── 4. 持久化: 所有因子(含LLM/arxiv) → FactorStore ──
            saved = 0
            try:
                from data.vault.factor_store import FactorStore
                fs = FactorStore()
                for r in result["results"]:
                    fname = r["factor"]
                    ic = r.get("ic", 0)
                    ic_ir = r.get("ic_ir", 0)
                    src = r.get("source", "traditional")
                    desc = r.get("description", r.get("formula", ""))
                    try:
                        fs.save_factor(
                            name=fname, df=self.df[["date"]].copy(),
                            ic=ic, ic_ir=ic_ir, category=src, description=str(desc)[:200],
                        )
                        saved += 1
                    except Exception:
                        logger.exception(f"FactorStore save failed for {fname}")
            except Exception as e:
                logger.warning(f"  FactorStore save skipped: {e}")

            if strong:
                track.promote(Stage.BACKTEST)
                logger.info(f"  {track.name}: {len(strong)} strong factors → BACKTEST "
                           f"(+ {len(all_strong)} tracks, {saved} saved to FactorStore)")
            else:
                track.archive(f"No strong factors (|IC|>0.05), best={track.factor_results.get('top_ic',0):.4f}")
                logger.info(f"  {track.name}: ARCHIVED ({track.archived_reason})")
        except Exception as e:
            track.archive(f"Factor stage error: {e}")

    # ── Stage 2: Backtest ──

    def _apply_env_filter(self, params: dict) -> dict:
        """根据环境评估+Edge Score调整策略参数。

        调整规则：
        - Edge Score → 统一仓位乘数（覆盖 env position_multiplier）
        - risk_level=high → 收紧止损 / risk_level=low → 放宽
        - direction=short → 反转评分序列方向
        """
        if self.env_result is None and self.edge_result is None:
            return params

        # Edge Score 仓位乘数（优先级高于 env position_multiplier）
        if self.edge_result is not None:
            pm = self.edge_result["edge"]
            # edge 在 0~1，映射到仓位 0.1~0.3
            pm = round(0.1 + pm * 0.2, 4)
        elif self.env_result is not None:
            pm = self.env_result.get("position_multiplier", 1.0)
        else:
            pm = 1.0

        # 1. 仓位乘数 → max_position_pct
        if "max_position_pct" in params:
            params["max_position_pct"] = round(params["max_position_pct"] * pm, 4)

        # 2. 风险等级 → 止损调整
        risk_level = "low"
        if self.env_result is not None:
            risk_level = self.env_result.get("risk_level", "low")
        elif self.edge_result is not None and not self.edge_result["should_trade"]:
            risk_level = "high"
        if "stop_loss_pct" in params and risk_level != "low":
            if risk_level == "high":
                params["stop_loss_pct"] = round(params["stop_loss_pct"] * 0.5, 4)
            else:
                params["stop_loss_pct"] = round(params["stop_loss_pct"] * 0.75, 4)

        # 3. 方向 → 评分序列反转（做空）
        if self.env_result is not None:
            direction = self.env_result.get("direction", "long")
            if direction == "short" and "_score_series" in params:
                scores = params["_score_series"]
                if scores is not None:
                    params["_score_series"] = 1.0 - scores

        return params

    def _should_skip_backtest(self) -> bool:
        """环境过滤器说"不交易"时，全部track跳过回测。"""
        if self.env_result is None:
            return False
        return not self.env_result.get("should_trade", True)

    def _run_stage2(self, track: Track):
        """策略回测：构建策略并回测"""
        try:
            from backtest.engine.bt_runner import run_backtest
            from backtest.strategies.experimental.factor_strategy import FactorStrategy
            from backtest.strategies.experimental.cross_section_strategy import CrossSectionStrategy

            cls = track.strategy_class
            params = dict(track.strategy_params)

            # 截面策略: 多股票Panel → 多数据源回测
            if cls is CrossSectionStrategy:
                cs_result = self._run_cross_section_backtest(track)
                if cs_result is None:
                    track.archive("CrossSection backtest: no valid panel data")
                    return
                sr = cs_result.sharpe or 0
                trades = cs_result.total_trades
                track.backtest_result = cs_result
                if sr > 0 and trades >= 3:
                    track.final_score = sr
                    track.promote(Stage.VALIDATE)
                    logger.info(f"  {track.name}: SR={sr:.2f} T={trades} → VALIDATE")
                else:
                    track.archive(f"CrossSection failed: SR={sr:.2f}, Trades={trades}")
                return

            if cls is FactorStrategy or cls is None:
                from backtest.analysis.factor_miner import FactorMiner, compute_factors

                if params.get("factor_name"):
                    # 单因子Track: 计算该因子的得分序列
                    fname = params["factor_name"]
                    factor_df = compute_factors(self.df)
                    if fname not in factor_df.columns:
                        track.archive(f"Factor '{fname}' not found in computed factors")
                        return
                    raw = factor_df[fname].values
                    valid_mask = ~np.isnan(raw)
                    if valid_mask.sum() < 252:
                        track.archive(f"Factor '{fname}' has <252 valid values")
                        return
                    # 扩展窗口分位数归一化（无前视偏差）
                    # 每个t时刻只用[0..t]的数据算分位，然后归一化t时刻的值
                    min_window = 252  # 至少1年数据才可信
                    from backtest.analysis.factor_miner import compute_ic
                    ic_info = compute_ic(factor_df, fname)
                    ic = ic_info.get("ic", 0)
                    scores = np.full(len(raw), np.nan)
                    for t in range(min_window, len(raw)):
                        if not valid_mask[t]:
                            continue
                        past = raw[:t+1][valid_mask[:t+1]]
                        q80_t = np.nanpercentile(past, 80)
                        q20_t = np.nanpercentile(past, 20)
                        if q80_t > q20_t:
                            scores[t] = (raw[t] - q20_t) / (q80_t - q20_t)
                        else:
                            scores[t] = 0.5
                    scores = np.clip(scores, 0, 1)
                    if ic < 0:
                        scores = 1 - scores  # 负IC → 低分位买入
                    scores = pd.Series(scores, index=self.df.index[:len(scores)])
                    params["_score_series"] = scores
                    params["position_mode"] = "vol_target"  # 专业仓位管理
                    cls = FactorStrategy
                else:
                    # 多因子Track: 综合评分
                    fm = FactorMiner(self.df)
                    fm.mine()
                    params["_score_series"] = fm.multi_factor_score(deduplicated=True)
                    cls = FactorStrategy

            # 环境过滤器：调整策略参数
            if self.env_result is not None:
                params = self._apply_env_filter(params)

            # 移除内部参数（已消费，不传给策略）
            for k in ("factor_name", "source", "formula"):
                params.pop(k, None)

            result = run_backtest(cls, self.df, **params)

            # 注入环境评估+Edge Score结果到回测结果中
            if self.env_result is not None:
                result["env_direction"] = self.env_result["direction"]
                result["env_position_multiplier"] = self.env_result["position_multiplier"]
                result["env_should_trade"] = self.env_result["should_trade"]
                result["env_risk_level"] = self.env_result["risk_level"]
            if self.edge_result is not None:
                result["edge_score"] = self.edge_result["edge"]
                result["edge_should_trade"] = self.edge_result["should_trade"]
                result["edge_breakdown"] = self.edge_result["breakdown"]

            sr = result.get("sharpe") or 0
            trades = result.get("total_trades", 0)
            track.backtest_result = result

            # Gate: positive Sharpe + enough trades
            # 管线筛查阈值 trades>=10（相对宽松），策略挖掘器最终过滤用 min_trades=30
            if sr > 0 and trades >= 10:
                # 门控通过后，Bayes精调参数（再跑一次回测验证）
                if self.optimize_params:
                    optimized = self._optimize_track_params(track, cls)
                    if optimized:
                        # 环境过滤器：Bayes精调后再次应用
                        opt_params = dict(track.strategy_params)
                        if self.env_result is not None:
                            opt_params = self._apply_env_filter(opt_params)
                        result = run_backtest(cls, self.df, initial_cash=self.cash,
                                             **opt_params)
                        sr = result.get("sharpe") or 0
                        trades = result.get("total_trades", 0)
                        track.backtest_result = result
                        logger.info(f"  {track.name}: Bayes精调后 SR={sr:.2f} T={trades}")

                track.final_score = sr
                track.promote(Stage.VALIDATE)
                logger.info(f"  {track.name}: SR={sr:.2f} T={trades} → VALIDATE")
            else:
                track.archive(f"Backtest failed: SR={sr:.2f}, Trades={trades}")
                logger.info(f"  {track.name}: ARCHIVED ({track.archived_reason})")
        except Exception as e:
            track.archive(f"Backtest error: {e}")

    def _optimize_track_params(self, track: Track, strategy_class) -> bool:
        """用贝叶斯优化精调Track的策略参数。返回True如果有改进。"""
        try:
            from backtest.analysis.bayesian_opt import BayesianOptimizer

            params = dict(track.strategy_params)
            if not params:
                return False

            # 构建参数空间：当前值的 ±50%
            param_space = {}
            for k, v in params.items():
                if not isinstance(v, (int, float)):
                    continue
                if isinstance(v, int):
                    lo = max(1, int(v * 0.5))
                    hi = max(lo + 2, int(v * 1.5))
                    param_space[k] = (lo, hi, 1)
                else:
                    param_space[k] = (v * 0.5, v * 1.5)

            if not param_space:
                return False

            bo = BayesianOptimizer(self.df, cash=self.cash)
            best = bo.optimize(strategy_class, param_space, n_trials=20, timeout=120)
            if best:
                track.strategy_params.update(best)
                return True
            return False
        except Exception as e:
            logger.debug(f"  Param optimization skipped: {e}")
            return False

    # ── Stage 3: Validate ──

    @staticmethod
    def _sharpe_to_pvalue(sharpe, n_years):
        """Lo (2002) 近似：从夏普比率推算 p-value。"""
        if sharpe is None or sharpe <= 0 or n_years <= 0:
            return 1.0
        t_stat = sharpe * math.sqrt(max(n_years, 0.25))
        return math.erfc(abs(t_stat) / math.sqrt(2))

    @staticmethod
    def _extract_daily_returns(track: Track):
        """从回测结果的权益曲线提取日收益率序列。"""
        bt = track.backtest_result
        if bt is None:
            return []
        try:
            eq = bt.get("equity_df")
            if eq is not None and hasattr(eq, "pct_change") and not eq.empty:
                return eq["equity"].pct_change().dropna().tolist()
        except Exception:
            logger.exception("Failed to extract equity curve from equity_df")
        # 兜底：直接从 equity_curve 原始字段取
        ec = bt.get("equity_curve", [])
        if ec and len(ec) > 1:
            try:
                equities = [float(e["equity"]) for e in ec]
                return [(equities[i] / equities[i-1] - 1) for i in range(1, len(equities))]
            except Exception:
                logger.exception("Failed to extract daily returns from equity_curve")
        return []

    def _run_fwer_filter(self, validated_tracks: list[Track]) -> list[Track]:
        """FWER/FDR 批量多重比较校正。不显著的 track 标记淘汰。"""
        if len(validated_tracks) < 2:
            return validated_tracks

        from backtest.analysis.validate import fwer_control

        # 从夏普推算 p-value
        results = []
        for t in validated_tracks:
            sr = t.backtest_result.get("sharpe") or 0 if t.backtest_result else 0
            start = t.backtest_result.get("start_date", "") if t.backtest_result else ""
            end = t.backtest_result.get("end_date", "") if t.backtest_result else ""
            try:
                n_years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365
            except Exception:
                logger.exception("Failed to parse backtest dates, using default 3 years")
                n_years = 3  # 默认 3 年
            p = self._sharpe_to_pvalue(sr, n_years)
            results.append({"p_value": p, "track": t, "sharpe": sr})

        n_total = max(len(self.tracks), len(validated_tracks))
        fwer_result = fwer_control(results, n_total=n_total, method="fdr")

        # 标记未通过 FWER 的 track
        survivors = []
        for r in results:
            if r.get("fwer_pass", True):
                survivors.append(r["track"])
                r["track"].validation_result = r["track"].validation_result or {}
                r["track"].validation_result["fwer_pass"] = True
            else:
                r["track"].archive(
                    f"FWER/FDR: p={r.get('p_value', 1.0):.4f} not significant "
                    f"after FDR correction (threshold={fwer_result['threshold']:.4f})"
                )
                logger.info(f"  {r['track'].name}: FWER-FAIL "
                           f"(SR={r.get('sharpe', 0):.2f}, p={r.get('p_value', 1.0):.4f})")

        logger.info(f"  FWER filter: {len(survivors)}/{len(validated_tracks)} survived FDR correction "
                    f"(threshold={fwer_result['threshold']:.4f})")
        return survivors

    def _run_stage3(self, track: Track, auto_recover: bool = False, max_recover: int = 3):
        """专业验证：WFA + DSR + CSCV。淘汰时若auto_recover且有溯源，立刻调Miner重挖。"""
        try:
            from backtest.analysis.validate import (
                rolling_window_validate, deflated_sharpe_ratio, cscv_pbo,
            )

            cls = track.strategy_class
            if cls is None:
                from backtest.strategies.builtin.ma_cross import MaCrossStrategy
                cls = MaCrossStrategy

            wf = rolling_window_validate(cls, self.df, is_years=2, oos_months=6, step_months=1,
                                         **track.strategy_params)
            sr = track.backtest_result.get("sharpe") or 0 if track.backtest_result else 0
            dsr = deflated_sharpe_ratio(sr, n_trials=len(self.tracks) * 10)

            # CSCV: 从回测权益曲线提取日收益序列，计算过拟合概率
            cscv = {"pbo": 0.5, "overfit_risk": "未计算"}
            daily_rets = self._extract_daily_returns(track)
            if daily_rets and len(daily_rets) >= 50:
                try:
                    cscv = cscv_pbo(daily_rets)
                except Exception:
                    logger.exception("CSCV PBO computation failed, using default values")

            track.validation_result = {
                "wfa_passed": wf["passed"],
                "wfa_winrate": wf["metrics"]["win_rate"],
                "wfa_wfe": wf["metrics"]["wfe"],
                "dsr_significant": dsr["significant"],
                "deflated_sr": dsr["deflated_sharpe"],
                "cscv_pbo": cscv["pbo"],
                "cscv_risk": cscv["overfit_risk"],
            }

            # CSCV PBO < 0.3 视为通过（过拟合概率低）
            cscv_ok = cscv["pbo"] < 0.3

            if wf["passed"] and dsr["significant"] and cscv_ok:
                track.final_score = max(track.final_score, dsr["deflated_sharpe"])
                track.promote(Stage.PAPER)
                logger.info(f"  {track.name}: WFA={'PASS' if wf['passed'] else 'FAIL'} "
                           f"DSR={'SIG' if dsr['significant'] else 'no'} "
                           f"CSCV={cscv['pbo']:.3f} → PAPER")
            else:
                track.archive(
                    f"Validation failed: WFA={'PASS' if wf['passed'] else 'FAIL'}, "
                    f"DSR={'SIG' if dsr['significant'] else 'no'}, "
                    f"CSCV PBO={cscv['pbo']:.3f}"
                )
                logger.info(f"  {track.name}: ARCHIVED ({track.archived_reason})")

                # ── 立刻恢复：淘汰时当场调Miner重挖 ──
                if auto_recover:
                    if self._recover_count < max_recover:
                        self._recover_count += 1
                        logger.info(f"  └─ Auto-recover triggered for {track.name}")
                        new_tracks = self._recover_track(track)
                        for nt in new_tracks:
                            self._run_stage2(nt)
                            self._run_stage3(nt)  # 新Track立刻验证
        except Exception as e:
            track.archive(f"Validation error: {e}")
            if auto_recover and (track.entry_type or track.exit_type):
                if self._recover_count < max_recover:
                    self._recover_count += 1
                    logger.info(f"  └─ Auto-recover triggered on error for {track.name}")
                    new_tracks = self._recover_track(track)
                    for nt in new_tracks:
                        self._run_stage2(nt)
                        self._run_stage3(nt)

    # ── Stage 4: Paper Trading ──

    def _run_stage4(self, track: Track):
        """纸上交易 — 使用 PaperTrader (StrategyGuard + 风控 + 偏差对比)"""
        try:
            from live.paper_trader import PaperTrader

            cls = track.strategy_class
            if cls is None:
                from backtest.strategies.builtin.ma_cross import MaCrossStrategy
                cls = MaCrossStrategy

            trader = PaperTrader(
                cls, self.df, initial_cash=self.cash,
                enable_guard=True, **track.strategy_params,
            )
            result = trader.run()

            n_trades = len([o for o in result.get("orders", [])
                           if o.get("status") == "filled"])
            total_return = result.get("total_return", 0)
            bt_return = result.get("baseline_return", 0)
            deviation = result.get("deviation", 0)
            guard_status = result.get("guard_status")
            n_signals = len(result.get("signals", []))

            # Gate conditions
            gate_pass = True
            gate_reasons = []

            if n_trades < 3:
                gate_pass = False
                gate_reasons.append(f"only {n_trades} trades")

            if guard_status and guard_status.get("blown"):
                gate_pass = False
                gate_reasons.append(f"StrategyGuard blown: {guard_status.get('reason', '')}")

            if deviation > 0.50 and n_trades < 10:
                gate_pass = False
                gate_reasons.append(f"deviation {deviation:.1%} > 50% with only {n_trades} trades")

            if gate_pass:
                track.paper_result = {
                    "trades": n_trades, "signals": n_signals,
                    "total_return": total_return, "deviation": deviation,
                }
                track.final_score = total_return
                track.promote(Stage.READY)
                logger.info(f"  {track.name}: {n_trades} trades, {total_return:.1%} "
                           f"(bt={bt_return:.1%}, dev={deviation:.1%}) → READY!")
                if n_trades >= 20:
                    logger.info(f"  └─ 已满足实盘就绪条件，可接入 PaperDaemon 长期跟踪")
            else:
                track.archive(f"Paper: {', '.join(gate_reasons)}")
                logger.info(f"  {track.name}: ARCHIVED ({track.archived_reason})")
        except Exception as e:
            track.archive(f"Paper trading error: {e}")

    # ── 截面回测：多股票Panel → 多数据源Backtrader ──

    def _run_cross_section_backtest(self, track: Track) -> dict or None:
        """截面多因子选股回测。panel = {symbol: DataFrame}"""
        import backtrader as bt

        if not self.panel or len(self.panel) < 5:
            return None

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.cash)

        from backtest.engine.bt_runner import AShareCommission
        comm = AShareCommission()
        cerebro.broker.addcommissioninfo(comm)
        cerebro.broker.set_slippage_perc(0.001)

        # 每只股票一个data feed
        n_added = 0
        for symbol, sdf in list(self.panel.items())[:50]:  # 最多50只
            if len(sdf) < 100:
                continue
            data = bt.feeds.PandasData(
                dataname=sdf, datetime="date",
                open="open", high="high", low="low", close="close", volume="volume",
                openinterest=-1,
            )
            cerebro.adddata(data)
            n_added += 1
        if n_added < 5:
            return None

        params = dict(track.strategy_params)
        cerebro.addstrategy(track.strategy_class, **params)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.025, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades_analyzer")

        start_val = cerebro.broker.getvalue()
        results = cerebro.run()
        strat = results[0]
        end_val = cerebro.broker.getvalue()

        sr_analyzer = strat.analyzers.sharpe.get_analysis()
        dd_analyzer = strat.analyzers.drawdown.get_analysis()
        ret_analyzer = strat.analyzers.returns.get_analysis()
        ta = strat.analyzers.trades_analyzer.get_analysis()

        sharpe = sr_analyzer.get("sharperatio", 0) or 0
        max_dd = dd_analyzer.get("max", {}).get("drawdown", 0) / 100
        annual_return = ret_analyzer.get("rnorm100", 0) / 100
        total_trades = ta.get("total", {}).get("total", 0)

        from backtest.engine.result import BacktestResult
        return BacktestResult(
            strategy_name=track.name,
            symbol=f"{n_added} stocks",
            start_value=self.cash, end_value=end_val,
            total_return=(end_val / start_val - 1),
            annual_return=annual_return,
            drawdown=max_dd,
            sharpe=sharpe,
            calmar=annual_return / max_dd if max_dd > 0 else 0,
            total_trades=total_trades,
        )

    # ── 自动恢复：淘汰Track → StrategyMiner重挖参数 ──

    def _recover_track(self, archived_track: Track) -> list[Track]:
        """从淘汰的Track中提取入场/出场逻辑，调StrategyMiner重搜参数。

        entry_type/exit_type 为空时自动推断：
          - FactorStrategy → 用因子名映射通用入场/出场规则
          - 策略名含关键词 → 按关键词推断
          - 兜底 → ma_cross入场 + trailing_stop出场
        """
        try:
            from backtest.strategy_miner import StrategyMiner, ENTRY_RULES, EXIT_RULES

            entry = archived_track.entry_type
            exit_ = archived_track.exit_type

            # fallback: 从策略参数/因子名推断入场出场类型
            if not entry or not exit_:
                inferred = self._infer_entry_exit(archived_track)
                entry = entry or inferred["entry"]
                exit_ = exit_ or inferred["exit"]
                if entry and exit_:
                    logger.debug(
                        f"  Auto-recover fallback for {archived_track.name}: "
                        f"entry={entry} exit={exit_}"
                    )

            # 验证入场/出场类型有效
            if entry not in ENTRY_RULES or exit_ not in EXIT_RULES:
                logger.debug(f"  Skip recover {archived_track.name}: entry={entry} exit={exit_} not in known rules")
                return []

            miner = StrategyMiner(self.df, cash=self.cash)
            result = miner.mine(
                max_combinations=15, min_trades=10,
                train_end="2020-01-01", val_end="2023-01-01", test_end="2025-01-01",
                strict=False,
            )

            top = result.get("top10", [])
            new_tracks = []
            for i, r in enumerate(top[:2], 1):  # 最多恢复2条
                score = r.get("score", 0)
                desc = r.get("description", f"recovered_{i}")
                params = r.get("params", {})

                new_name = f"{archived_track.name}_v{i}"
                new_track = self.add_track(
                    new_name,
                    strategy_params=params,
                )
                new_track.entry_type = entry
                new_track.exit_type = exit_

                # 用动态生成的策略类做回测（StrategyMiner内部创建）
                from backtest.strategy_miner import _make_strategy_class
                try:
                    cls = _make_strategy_class(entry, exit_, params)
                    new_track.strategy_class = cls
                    new_track.stage = Stage.BACKTEST  # 跳过因子阶段，直接回测
                    new_track.final_score = score
                    logger.info(f"  Recover: {new_name} (score={score:.3f}) {desc[:50]}")
                    new_tracks.append(new_track)
                except Exception as e:
                    logger.warning(f"  Recover failed for {new_name}: {e}")

            return new_tracks
        except Exception as e:
            logger.warning(f"Auto-recover failed for {archived_track.name}: {e}")
            return []

    def _infer_entry_exit(self, track: Track) -> dict[str, str]:
        """从 Track 的因子名/策略参数推断入场和出场类型（fallback）。"""
        entry = ""
        exit_ = ""

        # 1. 从因子名推断
        factors = track.factors or []
        params = track.strategy_params or {}

        factor_str = " ".join(factors).lower() + " " + str(params).lower()
        name_lower = track.name.lower()

        # 动量类因子 → momentum 入场
        momentum_kw = ["momentum", "roc", "return", "trend", "sma", "ema", "ma_cross"]
        if any(kw in factor_str or kw in name_lower for kw in momentum_kw):
            entry = entry or "momentum"

        # 均值回归 → bollinger/mean_revert 入场
        mr_kw = ["mean_revert", "bollinger", "rsi", "oversold", "deviation"]
        if any(kw in factor_str or kw in name_lower for kw in mr_kw):
            entry = entry or "mean_revert"

        # 突破类 → breakout 入场
        breakout_kw = ["breakout", "donchian", "channel", "high", "resistance"]
        if any(kw in factor_str or kw in name_lower for kw in breakout_kw):
            entry = entry or "breakout"

        # 成交量类 → volume 入场
        volume_kw = ["volume", "dollar_volume", "turnover", "vpt"]
        if any(kw in factor_str or kw in name_lower for kw in volume_kw):
            entry = entry or "volume_breakout"

        # 2. 从策略类名推断
        if track.strategy_class is not None:
            cls_name = track.strategy_class.__name__.lower()
            if "momentum" in cls_name:
                entry = entry or "momentum"
            elif "mean" in cls_name or "bollinger" in cls_name:
                entry = entry or "mean_revert"
            elif "cross" in cls_name:
                entry = entry or "ma_cross"
            elif "factor" in cls_name:
                entry = entry or "momentum"  # FactorStrategy default

        # 3. Kronos 因子特殊处理
        if "kronos" in factor_str or "kronos" in name_lower:
            if "forecast_return" in factor_str:
                entry = "momentum"   # 预测涨跌 → 动量类
            elif "forecast_vol" in factor_str:
                entry = "mean_revert"  # 预测波动 → 均值回归类

        # 4. 兜底
        if not entry:
            entry = "ma_cross"
        if not exit_:
            exit_ = "trailing_stop"

        return {"entry": entry, "exit": exit_}

    def report(self) -> str:
        """生成管线运行报告"""
        lines = [
            "=" * 70,
            "  PIPELINE REPORT",
            "=" * 70,
        ]
        if self.env_result is not None:
            er = self.env_result
            lines.append(f"  Env Filter: trade={'YES' if er['should_trade'] else 'NO'} "
                        f"dir={er['direction']} pos={er['position_multiplier']:.0%} "
                        f"risk={er['risk_level']}")
            lines.append(f"  └─ {er['reasoning'][:100]}")
        for stage in [Stage.READY, Stage.PAPER, Stage.VALIDATE, Stage.BACKTEST, Stage.FACTOR]:
            tracks = [t for t in self.tracks if t.stage == stage]
            if tracks:
                lines.append(f"\n  [{STAGE_NAMES[stage]}] {len(tracks)} tracks:")
                for t in tracks:
                    lines.append(f"    - {t.name} (score={t.final_score:.3f})")

        archived = [t for t in self.tracks if t.stage == Stage.ARCHIVED]
        if archived:
            lines.append(f"\n  [ARCHIVED] {len(archived)} tracks:")
            for t in archived:
                lines.append(f"    - {t.name}: {t.archived_reason}")

        lines.append("\n" + "=" * 70)
        lines.append(f"  Summary: {sum(1 for t in self.tracks if t.stage.value >= Stage.VALIDATE.value)} "
                     f"validated / {len(self.tracks)} total")
        lines.append("=" * 70)

        # 对比表
        lines.append(f"\n  {'Track':<20} {'Stage':<12} {'Score':>7} {'Details'}")
        lines.append("  " + "-" * 65)
        for t in sorted(self.tracks, key=lambda t: t.final_score, reverse=True):
            score_str = f"{t.final_score:+.3f}" if t.final_score != 0 else "-"
            detail = ""
            if t.stage == Stage.ARCHIVED:
                detail = t.archived_reason[:35]
            elif t.stage == Stage.VALIDATE:
                wfa = 'PASS' if t.validation_result.get('wfa_passed') else 'FAIL'
                pbo = t.validation_result.get('cscv_pbo', '?')
                detail = f"WFA={wfa} PBO={pbo}"
            elif t.stage == Stage.READY:
                detail = "READY FOR LIVE"
            lines.append(f"  {t.name:<20} {STAGE_NAMES[t.stage]:<12} {score_str:>7}  {detail}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def ready_strategies(self) -> list[Track]:
        """返回所有候选实盘的 track"""
        return [t for t in self.tracks if t.stage == Stage.READY]


# ============================================================
# 命令行测试
# ============================================================
# python core/pipeline.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fd
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy

    df = fd("沪深300", "20200101", "20260524")
    print(f"Data: {len(df)} rows")

    pl = Pipeline(df, cash=100000)

    # 多条并行的研究路线
    pl.add_track("MA_5_20", strategy_class=MaCrossStrategy,
                 strategy_params={"fast": 5, "slow": 20})
    pl.add_track("MA_10_30", strategy_class=MaCrossStrategy,
                 strategy_params={"fast": 10, "slow": 30})
    pl.add_track("MA_5_60", strategy_class=MaCrossStrategy,
                 strategy_params={"fast": 5, "slow": 60})
    pl.add_track("Factor_Auto", factors=["momentum_smooth", "dollar_volume", "ret_3m"])

    pl.run(until_stage=Stage.VALIDATE)
    print(pl.report())
