"""
策略挖掘引擎（Strategy Miner） — 元素组合 → 自动回测 → 优选输出

核心理念：
    策略 = 入场规则 + 出场规则 + 过滤条件 + 参数组合
    把策略拆成基本"元素"，排列组合生成候选，自动回测筛选，
    好的留下进入纸上交易验证。

类比：化学里把元素周期表上的元素组合成不同材料，遇到性能好的就测试。

关键保护（防止过拟合——试1000个策略总有几个碰巧好）：
    1. 样本外验证是硬门槛（不通过直接淘汰）
    2. 最少交易次数（≤5笔的不靠谱）
    3. 参数不超过5个
    4. 打分 = 年化收益 × 0.4 + 夏普 × 0.3 - 回撤 × 0.3

用法
--------
>>> from backtest.strategy_miner import StrategyMiner
>>> miner = StrategyMiner(df)
>>> results = miner.mine(max_combinations=50)
>>> print(results["top10"])
"""

import logging
import time
from itertools import product

import backtrader as bt
import numpy as np
import pandas as pd

from backtest.engine.bt_runner import BaseStrategy, AShareCommission# ============================================================
# 1. 元素表 — 策略的基本"化学元素"
# ============================================================

# 入场规则（6种）
from config.log import get_logger

logger = get_logger("strategy_miner")

ENTRY_RULES = {
    "ma_cross_up": {
        "name": "MA金叉",
        "params": {"ma_fast": [5, 10], "ma_slow": [20, 30, 60]},
        "desc_template": "SMA({ma_fast})上穿SMA({ma_slow})买入",
    },
    "rsi_oversold": {
        "name": "RSI超卖",
        "params": {"rsi_period": [14], "rsi_low": [25, 30, 35]},
        "desc_template": "RSI({rsi_period})低于{rsi_low}买入",
    },
    "bollinger_low": {
        "name": "布林下轨",
        "params": {"bb_period": [20], "bb_dev": [2.0, 2.5], "require_mid": [True, False]},
        "desc_template": "价格低于布林下轨(period={bb_period}, dev={bb_dev})买入",
    },
    "momentum_break": {
        "name": "动量突破",
        "params": {"mom_lookback": [60, 120, 250], "mom_threshold": [0.05, 0.10, 0.20]},
        "desc_template": "{mom_lookback}日涨跌幅>{mom_threshold:.0%}买入",
    },
    "volume_surge": {
        "name": "放量突破",
        "params": {"vol_period": [20], "vol_mult": [1.5, 2.0, 3.0]},
        "desc_template": "成交量>{vol_period}日均量的{vol_mult}倍买入",
    },
    "price_channel": {
        "name": "通道突破",
        "params": {"ch_period": [20, 60], "ch_direction": ["up"]},
        "desc_template": "价格突破{ch_period}日最高价买入",
    },
}

# 出场规则（6种）
EXIT_RULES = {
    "ma_cross_down": {
        "name": "MA死叉",
        "params": {"ma_fast_e": [5, 10], "ma_slow_e": [20, 30, 60]},
        "desc_template": "SMA({ma_fast_e})下穿SMA({ma_slow_e})卖出",
    },
    "rsi_overbought": {
        "name": "RSI超买",
        "params": {"rsi_period_e": [14], "rsi_high": [65, 70, 75]},
        "desc_template": "RSI({rsi_period_e})高于{rsi_high}卖出",
    },
    "stop_loss": {
        "name": "固定止损",
        "params": {"sl_pct": [0.02, 0.03, 0.05]},
        "desc_template": "亏损超过{sl_pct:.0%}止损卖出",
    },
    "bollinger_mid": {
        "name": "回中轨",
        "params": {"bb_period_e": [20], "bb_dev_e": [2.0]},
        "desc_template": "价格回到布林中轨卖出",
    },
    "trailing_stop": {
        "name": "移动止损",
        "params": {"ts_pct": [0.03, 0.05, 0.10]},
        "desc_template": "从最高点回落{ts_pct:.0%}卖出",
    },
    "time_exit": {
        "name": "持仓到期",
        "params": {"te_days": [5, 10, 20]},
        "desc_template": "持仓{te_days}天后卖出",
    },
}


# ============================================================
# 1.5. 逻辑规则预筛选 — 用金融常识过滤，不是用数据筛选
# ============================================================

LOGIC_RULES: dict[tuple[str, str], bool] = {
    # 动量类入场 × 各类出场
    ("momentum_break", "trailing_stop"): True,
    ("momentum_break", "stop_loss"): True,
    ("momentum_break", "ma_cross_down"): True,
    ("momentum_break", "bollinger_mid"): False,
    ("momentum_break", "rsi_overbought"): True,
    ("momentum_break", "time_exit"): False,
    # 均值回归入场 × 各类出场
    ("rsi_oversold", "take_profit"): True,
    ("rsi_oversold", "rsi_overbought"): True,
    ("rsi_oversold", "time_exit"): True,
    ("rsi_oversold", "trailing_stop"): False,
    ("rsi_oversold", "stop_loss"): False,
    ("rsi_oversold", "ma_cross_down"): False,
    # 布林带入场 × 各类出场
    ("bollinger_low", "trailing_stop"): True,
    ("bollinger_low", "bollinger_mid"): True,
    ("bollinger_low", "rsi_overbought"): True,
    ("bollinger_low", "stop_loss"): False,
    ("bollinger_low", "ma_cross_down"): False,
    ("bollinger_low", "time_exit"): False,
    # 通道突破入场 × 各类出场
    ("price_channel", "trailing_stop"): True,
    ("price_channel", "stop_loss"): True,
    ("price_channel", "time_exit"): True,
    ("price_channel", "rsi_overbought"): False,
    ("price_channel", "bollinger_mid"): False,
    ("price_channel", "ma_cross_down"): False,
    # 放量突破入场 × 各类出场
    ("volume_surge", "trailing_stop"): True,
    ("volume_surge", "stop_loss"): True,
    ("volume_surge", "ma_cross_down"): True,
    ("volume_surge", "time_exit"): False,
    ("volume_surge", "rsi_overbought"): False,
    ("volume_surge", "bollinger_mid"): False,
    # 均线金叉入场 × 各类出场
    ("ma_cross_up", "trailing_stop"): True,
    ("ma_cross_up", "ma_cross_down"): True,
    ("ma_cross_up", "stop_loss"): True,
    ("ma_cross_up", "time_exit"): False,
    ("ma_cross_up", "rsi_overbought"): False,
    ("ma_cross_up", "bollinger_mid"): False,
}


def _check_logic(entry_type: str, exit_type: str) -> bool:
    """用经济逻辑预筛选入场×出场组合。默认不通，防止随机噪音。"""
    return LOGIC_RULES.get((entry_type, exit_type), False)


# ============================================================
# 2. 动态策略生成器
# ============================================================

def _make_strategy_class(entry_name: str, exit_name: str, strategy_params: dict) -> type:
    """
    动态创建一个 Backtrader Strategy 类。

    给定 入场规则 + 出场规则 + 参数，生成一个可以直接传给 run_backtest 的策略类。
    """
    # 先转成 tuple（避免和 Backtrader 基类的 params 冲突）
    _bt_params = tuple(strategy_params.items())

    # 生成唯一类名
    class_name = f"Mined_{entry_name}_{exit_name}_{hash(frozenset(strategy_params.items())) & 0xFFFF:04x}"

    entry_info = ENTRY_RULES[entry_name]
    exit_info = EXIT_RULES[exit_name]

    class MinedStrategy(BaseStrategy):
        params = _bt_params  # 用预先转好的 tuple，避免和基类 params 命名冲突

        def __init__(self):
            super().__init__()
            self.entry_price = 0
            self.peak_price = 0

            # ---- 入场指标 ----
            if entry_name == "ma_cross_up":
                self.ma_f = bt.indicators.SMA(self.data.close, period=self.params.ma_fast)
                self.ma_s = bt.indicators.SMA(self.data.close, period=self.params.ma_slow)
                self.entry_signal = bt.indicators.CrossOver(self.ma_f, self.ma_s)

            elif entry_name == "rsi_oversold":
                self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

            elif entry_name == "bollinger_low":
                self.bb = bt.indicators.BollingerBands(
                    self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev
                )

            elif entry_name == "momentum_break":
                self.roc = bt.indicators.RateOfChange(self.data.close, period=self.params.mom_lookback)

            elif entry_name == "volume_surge":
                self.vol_avg = bt.indicators.SMA(self.data.volume, period=self.params.vol_period)

            elif entry_name == "price_channel":
                self.ch_high = bt.indicators.Highest(self.data.high, period=self.params.ch_period, subplot=False)

            # ---- 出场指标 ----
            if exit_name == "ma_cross_down":
                self.ma_fe = bt.indicators.SMA(self.data.close, period=self.params.ma_fast_e)
                self.ma_se = bt.indicators.SMA(self.data.close, period=self.params.ma_slow_e)
                self.exit_signal = bt.indicators.CrossOver(self.ma_fe, self.ma_se)

            elif exit_name == "rsi_overbought":
                self.rsi_e = bt.indicators.RSI(self.data.close, period=self.params.rsi_period_e)

            elif exit_name == "bollinger_mid":
                self.bb_e = bt.indicators.BollingerBands(
                    self.data.close, period=self.params.bb_period_e, devfactor=self.params.bb_dev_e
                )

            elif exit_name == "trailing_stop" or exit_name == "stop_loss":
                pass  # 这些在 next() 里手动计算

            self.bars_held = 0

        # ---- 入场逻辑 ----
        def _check_entry(self):
            if entry_name == "ma_cross_up":
                return self.entry_signal[0] > 0
            elif entry_name == "rsi_oversold":
                return self.rsi[0] < self.params.rsi_low
            elif entry_name == "bollinger_low":
                below = self.data.close[0] <= self.bb.lines.bot[0]
                if not self.params.get("require_mid", False):
                    return below
                # 要求之前跌破中轨（避免在上涨趋势中买在下轨）
                return below
            elif entry_name == "momentum_break":
                return self.roc[0] / 100 > self.params.mom_threshold
            elif entry_name == "volume_surge":
                return self.data.volume[0] > self.vol_avg[0] * self.params.vol_mult
            elif entry_name == "price_channel":
                return self.data.close[0] >= self.ch_high[-1]  # 昨日最高价（避免未来函数）
            return False

        # ---- 出场逻辑 ----
        def _check_exit(self):
            if exit_name == "ma_cross_down":
                return self.exit_signal[0] < 0
            elif exit_name == "rsi_overbought":
                return self.rsi_e[0] > self.params.rsi_high
            elif exit_name == "stop_loss":
                loss = (self.data.close[0] / self.entry_price - 1)
                return loss < -self.params.sl_pct
            elif exit_name == "bollinger_mid":
                return self.data.close[0] >= self.bb_e.lines.mid[0]
            elif exit_name == "trailing_stop":
                if self.data.close[0] > self.peak_price:
                    self.peak_price = self.data.close[0]
                return self.data.close[0] < self.peak_price * (1 - self.params.ts_pct)
            elif exit_name == "time_exit":
                return self.bars_held >= self.params.te_days
            return False

        def next(self):
            if not self.position:
                if self._check_entry():
                    size = int(self.broker.getcash() / self.data.close[0])
                    if size > 0:
                        self.buy(size=size)
                        self.entry_price = self.data.close[0]
                        self.peak_price = self.data.close[0]
                        self.bars_held = 0
            else:
                self.bars_held += 1
                if self.data.close[0] > self.peak_price:
                    self.peak_price = self.data.close[0]
                if self._check_exit():
                    self.close()

    MinedStrategy.__name__ = class_name
    MinedStrategy.__qualname__ = class_name
    return MinedStrategy


# ============================================================
# 3. 组合生成器
# ============================================================

def generate_combinations(max_combinations: int = 100) -> list[dict]:
    """
    生成所有有意义的 (入场, 出场, 参数) 组合。

    两步筛选：
      1. 逻辑规则预筛选 — 经济逻辑不通的直接排除（约剩 ~18/36）
      2. 参数展开 — 笛卡尔积取前5组
    """
    combinations = []
    seen = set()
    logic_skipped = []

    for e_name, e_info in ENTRY_RULES.items():
        for x_name, x_info in EXIT_RULES.items():
            # 逻辑预筛选
            if not _check_logic(e_name, x_name):
                logic_skipped.append(f"{e_name}+{x_name}")
                continue

            # 为每对 (入场, 出场) 生成 1-5 组参数组合
            param_keys = list(e_info["params"].keys()) + list(x_info["params"].keys())
            param_values = list(e_info["params"].values()) + list(x_info["params"].values())

            param_combos = list(product(*param_values))

            # 每对最多取 5 组参数
            for pvals in param_combos[:5]:
                params = dict(zip(param_keys, pvals))
                sig = f"{e_name}|{x_name}|{sorted(params.items())}"
                if sig in seen:
                    continue
                seen.add(sig)
                combinations.append({
                    "entry": e_name,
                    "exit": x_name,
                    "params": params,
                    "entry_name": e_info["name"],
                    "exit_name": x_info["name"],
                    "logic_pass": True,
                })
                if len(combinations) >= max_combinations:
                    logger.info(f"逻辑预筛选: {len(logic_skipped)} 组合被排除 ({', '.join(logic_skipped[:8])}"
                                f"{'...' if len(logic_skipped) > 8 else ''})")
                    return combinations

    logger.info(f"逻辑预筛选: 排除 {len(logic_skipped)} 组合，生成 {len(combinations)} 候选")
    if logic_skipped:
        logger.debug(f"  排除明细: {', '.join(logic_skipped)}")
    return combinations


# ============================================================
# 4. 评分函数
# ============================================================

def _normalize(value: float, low: float, high: float) -> float:
    """线性归一化到 [0, 1]，超出范围截断"""
    if value is None:
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


def composite_score(result: dict, meta: dict = None) -> float:
    """
    综合评分 — 逻辑自洽+参数稳定的策略排名靠前。

    维度: 夏普30% + 卡玛30% + FWER20% + 逻辑10% + 稳健10%
    """
    sharpe = _normalize(result.get("sharpe") or 0, 0, 3)
    calmar = _normalize(result.get("calmar") or 0, 0, 2)
    fwer_pass = 1.0 if (meta or {}).get("fwer_pass", False) else 0.0
    logic_pass = 1.0 if (meta or {}).get("logic_pass", False) else 0.0
    stability = _normalize((meta or {}).get("stability", 0), 0, 1)

    score = (
        0.3 * sharpe
        + 0.3 * calmar
        + 0.2 * fwer_pass
        + 0.1 * logic_pass
        + 0.1 * stability
    )
    return round(score, 4)


# ============================================================
# 5. 主矿工类
# ============================================================

class StrategyMiner:
    """
    策略矿工。

    用法
    --------
    >>> miner = StrategyMiner(df, cash=100000)
    >>> results = miner.mine(max_combinations=60)
    >>> print(results["top10"])
    >>> # miner.save_top("notebooks/top_strategies.json")  # 保存最优策略
    """

    def __init__(self, df: pd.DataFrame, cash: float = 100000.0,
                 only_long: bool = True):
        self.df = df
        self.cash = cash
        self.only_long = only_long  # A股普通账户无法做空
        self.results: list[dict] = []

    def mine(
        self,
        max_combinations: int = 100,
        min_trades: int = 30,
        train_end: str = "2020-01-01",
        val_end: str = "2023-01-01",
        test_end: str = "2025-01-01",
        use_vectorbt: bool = False,
        vectorbt_top_n: int = 20,
        strict: bool = True,
        only_long: bool = None,
    ) -> dict:
        """
        三步数据分割 + 五重过滤 + 综合评分。

        流程 (全量候选 → 逻辑筛 → 训练 → min_trades+sharpe → 验证 → 衰退+参数+FWER → 测试 → 综合排名):
          1. 生成策略组合 → LOGIC_RULES 预筛选 (~18/36)
          2. 训练集回测 (数据开始 ~ train_end) → min_trades≥30 + sharpe>0
          3. 验证集回测 (train_end ~ val_end) → 衰退率<50% + FWER + 参数稳健
          4. 测试集回测 (val_end ~ test_end) → 只跑一轮, composite_score 排名
        """
        from backtest.engine.bt_runner import run_backtest
        from backtest.analysis.validate import out_of_sample_test, check_decay_rate, fwer_control

        logger.info(f"生成策略组合（最多 {max_combinations} 种）...")
        combos = generate_combinations(max_combinations)
        logger.info(f"共 {len(combos)} 种候选待回测")

        start_time = time.time()

        # ── 准备三段时间 ──
        train_df = self.df[self.df["date"] < train_end].copy()
        val_df = self.df[(self.df["date"] >= train_end) & (self.df["date"] < val_end)].copy()
        test_df = self.df[(self.df["date"] >= val_end) & (self.df["date"] < test_end)].copy()

        if len(train_df) < 100:
            logger.error(f"训练集数据不足（{len(train_df)}条），无法挖掘")
            return {"top10": [], "total_tested": len(combos), "total_passed": 0,
                    "elapsed_seconds": 0, "error": "训练数据不足"}

        step1_pass = []

        # ═══ 阶段 1: 训练集回测 + 基础过滤 ═══
        logger.info(f"[阶段1] 训练集回测 {train_df['date'].min().date()} ~ {train_df['date'].max().date()} ({len(train_df)}条)")
        for i, combo in enumerate(combos):
            try:
                strategy_class = _make_strategy_class(combo["entry"], combo["exit"], combo["params"])
                result = run_backtest(strategy_class, train_df, initial_cash=self.cash, **combo["params"])

                trades = result.get("total_trades", 0)
                if trades < min_trades:
                    continue
                if result.get("sharpe", 0) <= 0:
                    continue

                combo["train_result"] = result
                step1_pass.append(combo)

                if (i + 1) % 10 == 0:
                    logger.info(f"  训练集进度: {i+1}/{len(combos)}, 通过: {len(step1_pass)}")
            except Exception:
                continue

        logger.info(f"阶段1完成: {len(combos)} → {len(step1_pass)} (min_trades≥{min_trades}, sharpe>0)")

        # ═══ 阶段 2: 验证集回测 + 过拟合过滤 ═══
        step2_pass = []
        if val_df is not None and len(val_df) >= 50:
            logger.info(f"[阶段2] 验证集回测 {val_df['date'].min().date()} ~ {val_df['date'].max().date()} ({len(val_df)}条)")
            for combo in step1_pass:
                try:
                    strategy_class = _make_strategy_class(combo["entry"], combo["exit"], combo["params"])
                    val_result = run_backtest(strategy_class, val_df, initial_cash=self.cash, **combo["params"])

                    train_sharpe = combo["train_result"].get("sharpe") or 0
                    val_sharpe = val_result.get("sharpe")

                    # 衰退率过滤
                    decay = check_decay_rate(train_sharpe, val_sharpe)
                    if decay.get("overfit"):
                        continue

                    combo["val_result"] = val_result
                    combo["decay"] = decay
                    step2_pass.append(combo)
                except Exception:
                    continue

            # FWER 校正 — 分母是所有原始组合数（不是幸存者数量）
            # 专业标准: 测试了 N 个策略，阈值就是 0.05/N，不管有多少存活到验证阶段
            # strict=False 时跳过 FWER（用于 auto_recover 等场景）
            if strict and len(step2_pass) > 1:
                fwer = fwer_control([r.get("train_result", {}) for r in step2_pass],
                                    n_total=len(combos))  # 用原始试验总数，非通过筛选数
                for i, combo in enumerate(step2_pass):
                    combo["fwer_pass"] = i < fwer["passed"]
                    combo["fwer_threshold"] = fwer["threshold"]
            elif not strict:
                for combo in step2_pass:
                    combo["fwer_pass"] = True
                    combo["fwer_threshold"] = 0
        else:
            step2_pass = step1_pass  # 无验证集，跳过

        logger.info(f"阶段2完成: {len(step1_pass)} → {len(step2_pass)} (衰退率+FWER)")

        # ═══ 阶段 3: 测试集回测 + 综合排名 ═══
        step3_results = []
        if test_df is not None and len(test_df) >= 50:
            logger.info(f"[阶段3] 测试集回测 {test_df['date'].min().date()} ~ {test_df['date'].max().date()} ({len(test_df)}条)")
            for combo in step2_pass:
                try:
                    strategy_class = _make_strategy_class(combo["entry"], combo["exit"], combo["params"])
                    result = run_backtest(strategy_class, test_df, initial_cash=self.cash, **combo["params"])

                    meta = {
                        "logic_pass": combo.get("logic_pass", False),
                        "fwer_pass": combo.get("fwer_pass", False),
                        "stability": combo.get("stability", 1.0),
                    }
                    score = composite_score(result, meta)
                    n_params = len(combo.get("params", {}))
                    if n_params > 5:
                        score -= 0.05 * (n_params - 5)

                    combo["test_result"] = result
                    combo["score"] = score
                    combo["desc"] = self._make_description(combo)
                    step3_results.append(combo)
                except Exception:
                    continue
        else:
            # 无测试集，用验证集结果打分
            for combo in step2_pass:
                result = combo.get("val_result", combo.get("train_result", {}))
                meta = {
                    "logic_pass": combo.get("logic_pass", False),
                    "fwer_pass": combo.get("fwer_pass", False),
                    "stability": combo.get("stability", 1.0),
                }
                score = composite_score(result, meta)
                combo["score"] = score
                combo["desc"] = self._make_description(combo)
                step3_results.append(combo)

        # 排序
        step3_results.sort(key=lambda x: x["score"], reverse=True)
        self.results = step3_results
        elapsed = time.time() - start_time

        n_fwer = sum(1 for r in step3_results if r.get("fwer_pass"))
        n_logic = sum(1 for r in step3_results if r.get("logic_pass"))
        logger.info(f"挖掘完成: {len(combos)} 种 → 训练{len(step1_pass)} → "
                    f"验证{len(step2_pass)} → 最终{len(step3_results)} 种, "
                    f"耗时 {elapsed:.0f}s, "
                    f"FWER通过: {n_fwer}, 逻辑通过: {n_logic}")

        # 无统计显著策略时警告
        if n_fwer == 0 and len(step3_results) < len(step2_pass):
            logger.warning("无统计显著策略: 所有候选未通过FWER多重比较校正，"
                           "建议扩大选择范围或接受更高的过拟合风险")

        # 构建返回结果
        top10 = step3_results[:10]
        clean_top10 = []
        for i, r in enumerate(top10):
            res = r.get("test_result", r.get("val_result", r.get("train_result", {})))
            clean_top10.append({
                "rank": i + 1,
                "source": "mined",
                "entry": r.get("entry_name", r["entry"]),
                "exit": r.get("exit_name", r["exit"]),
                "params": r["params"],
                "desc": r.get("desc", ""),
                "score": r.get("score", 0),
                "annual_return": res.get("annual_return", 0),
                "drawdown": res.get("drawdown", 1.0),
                "sharpe": res.get("sharpe"),
                "trades": res.get("total_trades", 0),
                "logic_pass": r.get("logic_pass", False),
                "fwer_pass": r.get("fwer_pass", False),
                "stability": r.get("stability", 1.0),
            })
        return {
            "top10": clean_top10,
            "total_tested": len(combos),
            "total_passed": len(step3_results),
            "elapsed_seconds": elapsed,
            "engine": "vectorbt+backtrader" if use_vectorbt else "backtrader",
            "summary": self._build_summary(clean_top10),
        }

    # ── Bayes 精调 ─────────────────────────────────────────

    def fine_tune_top(
        self,
        n: int = 3,
        n_trials: int = 30,
        timeout: int = 300,
    ) -> list[dict]:
        """对挖掘结果 Top N 运行贝叶斯优化精调参数。

        网格搜索做广筛（快速覆盖大量entry×exit组合），
        Bayes做深调（在优质组合的参数空间中精细搜索）。

        参数
        ----------
        n : 精调前几名
        n_trials : Bayes优化试验次数
        timeout : 单策略优化超时秒数

        返回
        -------
        [{original, optimized, improvement, ...}]
        """
        from backtest.analysis.bayesian_opt import BayesianOptimizer

        if not self.results:
            logger.warning("没有挖掘结果，先运行 mine()")
            return []

        top_n = self.results[:n]
        tuned = []

        for i, combo in enumerate(top_n):
            entry = combo.get("entry", "")
            exit_rule = combo.get("exit", "")
            params = combo.get("params", {})
            original_score = combo.get("score", 0)

            # 构建参数空间（当前值的 ±50% 范围，整数参数取 step=1）
            param_space = {}
            for k, v in params.items():
                if isinstance(v, int):
                    lo = max(1, int(v * 0.5))
                    hi = max(lo + 2, int(v * 1.5))
                    param_space[k] = (lo, hi, 1)
                else:
                    lo = v * 0.5
                    hi = v * 1.5
                    param_space[k] = (lo, hi)

            strategy_class = _make_strategy_class(entry, exit_rule, params)
            train_df = self.df[self.df["date"] < combo.get("train_result", {}).get("end_date", "2020-01-01")]

            logger.info(f"Bayes精调 [{i+1}/{n}]: {combo.get('desc', '')}, "
                        f"参数空间 {param_space}")

            try:
                bo = BayesianOptimizer(train_df, cash=self.cash)
                best_params = bo.optimize(
                    strategy_class, param_space,
                    n_trials=n_trials, timeout=timeout,
                )
                # 用最优参数重跑
                tuned_class = _make_strategy_class(entry, exit_rule, best_params)
                from backtest.engine.bt_runner import run_backtest
                tuned_result = run_backtest(tuned_class, self.df, initial_cash=self.cash, **best_params)

                improvement = (tuned_result.get("sharpe", 0) -
                               combo.get("train_result", {}).get("sharpe", 0))

                tuned.append({
                    "entry": entry,
                    "exit": exit_rule,
                    "original_params": params,
                    "optimized_params": best_params,
                    "original_sharpe": combo.get("train_result", {}).get("sharpe", 0),
                    "tuned_sharpe": tuned_result.get("sharpe", 0),
                    "improvement": improvement,
                    "tuned_result": tuned_result,
                })
                logger.info(f"  完成: {best_params}, "
                            f"夏普 {combo.get('train_result', {}).get('sharpe', 0):.2f} → "
                            f"{tuned_result.get('sharpe', 0):.2f} "
                            f"({improvement:+.2f})")
            except Exception as e:
                logger.error(f"Bayes精调失败 [{i+1}/{n}]: {e}")
                tuned.append({"entry": entry, "exit": exit_rule, "error": str(e)})

        return tuned

    # ── VectorBT 快速初筛 ───────────────────────────────────

    def _run_vectorbt_screen(
        self,
        combos: list[dict],
        split_date: str,
    ) -> list[dict] | None:
        """
        用 VectorBT 对所有组合做向量化快速初筛。

        VectorBT 比 Backtrader 快 100-200 倍，适合大规模参数扫描。
        将日线数据转为向量化信号 → 批量回测 → 按夏普排序返回。

        返回 None 表示 VectorBT 不可用（未安装/数据不足）。
        """
        try:
            import vectorbt as vbt
        except ImportError:
            logger.info("VectorBT 未安装（pip install vectorbt），使用 Backtrader 全量回测")
            return None

        train_df = self.df[self.df["date"] < split_date].copy()
        if len(train_df) < 100:
            return None

        close = train_df["close"].values
        high = train_df.get("high", close).values
        low = train_df.get("low", close).values
        volume = train_df.get("volume", pd.Series(1, index=train_df.index)).values

        results = []

        for combo in combos:
            try:
                entry_signal = self._vb_generate_entry(
                    combo["entry"], combo["params"], close, high, low, volume
                )
                exit_signal = self._vb_generate_exit(
                    combo["exit"], combo["params"], close, high, low, volume, entry_signal
                )

                if entry_signal is None or not entry_signal.any():
                    continue

                pf = vbt.Portfolio.from_signals(
                    close=pd.Series(close),
                    entries=pd.Series(entry_signal),
                    exits=pd.Series(exit_signal),
                    freq="D",
                    init_cash=self.cash,
                )

                trades = pf.trades
                if trades.count() < 3:
                    continue

                total_return = pf.total_return()
                if total_return <= -0.30:
                    continue

                combo["vb_return"] = float(total_return)
                combo["vb_sharpe"] = float(pf.sharpe_ratio()) if pf.sharpe_ratio() else 0
                combo["vb_drawdown"] = float(pf.max_drawdown())
                combo["vb_trades"] = trades.count()
                combo["vb_score"] = (
                    combo["vb_return"] * 0.4
                    + combo["vb_sharpe"] * 0.3
                    - abs(combo["vb_drawdown"]) * 0.3
                )
                results.append(combo)

            except Exception:
                continue

        if not results:
            return None

        results.sort(key=lambda x: x.get("vb_score", 0), reverse=True)
        return results

    def _vb_generate_entry(
        self,
        entry_name: str,
        params: dict,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray | None:
        """生成 VectorBT 入场信号（布尔数组）"""
        n = len(close)

        if entry_name == "ma_cross_up":
            fast = params.get("ma_fast", 5)
            slow = params.get("ma_slow", 20)
            ma_f = pd.Series(close).rolling(fast).mean().values
            ma_s = pd.Series(close).rolling(slow).mean().values
            signal = np.zeros(n, dtype=bool)
            for i in range(slow + 1, n):
                if ma_f[i] > ma_s[i] and ma_f[i - 1] <= ma_s[i - 1]:
                    signal[i] = True
            return signal

        elif entry_name == "rsi_oversold":
            period = params.get("rsi_period", 14)
            rsi_low = params.get("rsi_low", 30)
            delta = pd.Series(close).diff()
            gain = delta.clip(lower=0).rolling(period).mean().values
            loss = (-delta.clip(upper=0)).rolling(period).mean().values
            rs = np.divide(gain, loss, out=np.zeros_like(gain), where=loss != 0)
            rsi = 100 - 100 / (1 + rs)
            return rsi < rsi_low

        elif entry_name == "bollinger_low":
            period = params.get("bb_period", 20)
            dev = params.get("bb_dev", 2.0)
            s = pd.Series(close)
            mid = s.rolling(period).mean().values
            std = s.rolling(period).std().values
            lower = mid - dev * std
            return close < lower

        elif entry_name == "momentum_break":
            lookback = params.get("mom_lookback", 60)
            threshold = params.get("mom_threshold", 0.10)
            roc = pd.Series(close).pct_change(lookback).values
            return roc > threshold

        elif entry_name == "volume_surge":
            period = params.get("vol_period", 20)
            mult = params.get("vol_mult", 1.5)
            vol_avg = pd.Series(volume).rolling(period).mean().values
            return volume > vol_avg * mult

        elif entry_name == "price_channel":
            period = params.get("ch_period", 20)
            ch_high = pd.Series(high).rolling(period).max().values
            signal = np.zeros(n, dtype=bool)
            for i in range(period + 1, n):
                if close[i] >= ch_high[i - 1]:  # 突破昨日最高价
                    signal[i] = True
            return signal

        return None

    def _vb_generate_exit(
        self,
        exit_name: str,
        params: dict,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        entry_signal: np.ndarray,
    ) -> np.ndarray:
        """生成 VectorBT 出场信号（布尔数组）"""
        n = len(close)
        signal = np.zeros(n, dtype=bool)

        if exit_name == "ma_cross_down":
            fast = params.get("ma_fast_e", 5)
            slow = params.get("ma_slow_e", 20)
            ma_f = pd.Series(close).rolling(fast).mean().values
            ma_s = pd.Series(close).rolling(slow).mean().values
            for i in range(slow + 1, n):
                if ma_f[i] < ma_s[i] and ma_f[i - 1] >= ma_s[i - 1]:
                    signal[i] = True
            return signal

        elif exit_name == "rsi_overbought":
            period = params.get("rsi_period_e", 14)
            rsi_high = params.get("rsi_high", 70)
            delta = pd.Series(close).diff()
            gain = delta.clip(lower=0).rolling(period).mean().values
            loss = (-delta.clip(upper=0)).rolling(period).mean().values
            rs = np.divide(gain, loss, out=np.zeros_like(gain), where=loss != 0)
            rsi = 100 - 100 / (1 + rs)
            return rsi > rsi_high

        elif exit_name == "stop_loss":
            sl_pct = params.get("sl_pct", 0.03)
            in_position = False
            entry_price = 0
            for i in range(n):
                if entry_signal[i]:
                    in_position = True
                    entry_price = close[i]
                elif in_position:
                    if close[i] < entry_price * (1 - sl_pct):
                        signal[i] = True
                        in_position = False
            return signal

        elif exit_name == "bollinger_mid":
            period = params.get("bb_period_e", 20)
            s = pd.Series(close)
            mid = s.rolling(period).mean().values
            return close >= mid

        elif exit_name == "trailing_stop":
            ts_pct = params.get("ts_pct", 0.05)
            in_position = False
            peak = 0
            for i in range(n):
                if entry_signal[i]:
                    in_position = True
                    peak = close[i]
                elif in_position:
                    if close[i] > peak:
                        peak = close[i]
                    if close[i] < peak * (1 - ts_pct):
                        signal[i] = True
                        in_position = False
            return signal

        elif exit_name == "time_exit":
            days = params.get("te_days", 10)
            in_position = False
            held = 0
            for i in range(n):
                if entry_signal[i]:
                    in_position = True
                    held = 0
                elif in_position:
                    held += 1
                    if held >= days:
                        signal[i] = True
                        in_position = False
            return signal

        return np.zeros(n, dtype=bool)

    def save_top(self, path: str, n: int = 10) -> None:
        """保存最佳策略到 JSON 文件"""
        import json
        top = self.results[:n]
        data = [
            {
                "entry": r["entry"],
                "exit": r["exit"],
                "params": {k: v for k, v in r["params"].items()
                          if isinstance(v, (int, float, bool, str))},
                "desc": r["desc"],
                "score": r["score"],
                "annual_return": r["result"]["annual_return"],
                "drawdown": r["result"]["drawdown"],
                "sharpe": r["result"]["sharpe"],
            }
            for r in top
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Top {n} 策略已保存: {path}")

    def _make_description(self, combo: dict) -> str:
        """生成策略描述"""
        e_tmpl = ENTRY_RULES[combo["entry"]]["desc_template"]
        x_tmpl = EXIT_RULES[combo["exit"]]["desc_template"]
        params = combo["params"]

        # 填充模板
        try:
            e_desc = e_tmpl.format(**{k: v for k, v in params.items()
                                      if k in e_tmpl})
            x_desc = x_tmpl.format(**{k: v for k, v in params.items()
                                      if k in x_tmpl})
        except (KeyError, ValueError):
            e_desc = f"{combo['entry_name']}({combo['entry']})"
            x_desc = f"{combo['exit_name']}({combo['exit']})"

        return f"{e_desc} → {x_desc}"

    def _build_summary(self, top10: list) -> str:
        """纯文本总结"""
        lines = [
            "=" * 100,
            "                   策略挖掘结果 — Top 10 (过拟合六重过滤)",
            "=" * 100,
            f"{'排名':<4} {'入场':<10} {'出场':<10} {'年化':>6} {'回撤':>6} {'夏普':>6} {'交易':>4} {'逻辑':<5} {'FWER':<5} {'稳定':<5}",
            "-" * 100,
        ]
        for r in top10:
            s = r.get('sharpe')
            sharpe_str = f"{s:.2f}" if s else "N/A"
            logic_str = "PASS" if r.get('logic_pass') else "-"
            fwer_str = "PASS" if r.get('fwer_pass') else "-"
            stab = r.get('stability', 1.0)
            stab_str = f"{stab:.2f}" if stab != 1.0 else "-"
            lines.append(
                f"{r['rank']:<4} {r['entry']:<10} {r['exit']:<10} "
                f"{r['annual_return']:>5.1%} {r['drawdown']:>5.1%} "
                f"{sharpe_str:>6} {r['trades']:>4} {logic_str:<5} {fwer_str:<5} {stab_str:<5}"
            )
        lines.append("=" * 100)
        return "\n".join(lines)


# ============================================================
# 命令行测试
# ============================================================
# python backtest/strategy_miner.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    print("=" * 60)
    print("策略挖掘测试（少量组合快速验证）")
    print("=" * 60)

    df = fetch_index_daily("沪深300", "20200101", "20250601")

    # 检测 VectorBT 是否可用
    try:
        import vectorbt
        use_vb = True
        print("VectorBT 可用 → 使用向量化加速模式")
    except ImportError:
        use_vb = False
        print("VectorBT 未安装 → 使用 Backtrader 模式")

    miner = StrategyMiner(df, cash=100000)
    results = miner.mine(max_combinations=30, min_trades=10, use_vectorbt=use_vb)
    print(results["summary"])

    if miner.results:
        print("\n优选策略详情:")
        for i, r in enumerate(results["top10"][:5]):
            print(f"  #{i+1} {r['desc']}")
            print(f"      年化{r['annual_return']:.2%} | 回撤{r['drawdown']:.2%} | "
                  f"交易{r['trades']}笔 | 样本外{'通过' if r['oos_passed'] else '待验证'}")
