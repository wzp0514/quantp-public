"""
因子选股策略 — 多因子综合评分 → 交易信号

与策略挖掘器（随机组合技术指标）不同：因子策略基于 IC 分析的因子，
有经济逻辑支撑，每个因子都通过统计显著性检验。

逻辑：
  - 每日计算因子综合评分(0-1)
  - 评分 > buy_threshold → 买入（加仓信号）
  - 评分 < sell_threshold → 卖出（减仓信号）
  - 阈值可配，默认买入>0.7、卖出<0.3

用法
--------
>>> from backtest.analysis.factor_miner import FactorMiner, compute_factors
>>> from backtest.strategies.experimental.factor_strategy import FactorStrategy
>>> fm = FactorMiner(df)
>>> fm.mine()
>>> scores = fm.multi_factor_score(deduplicated=True)
>>> # 将 scores 传入 strategy 做回测
"""

import logging

import backtrader as bt
import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("factor_strategy")


class FactorStrategy(bt.Strategy):
    """
    基于多因子评分的策略。

    原理：当因子综合评分高于买入阈值时做多，低于卖出阈值时空仓。
    评分在 0-1 之间，0.5 为中性。
    """

    params = (
        ("buy_threshold", 0.65),    # 评分高于此值买入
        ("sell_threshold", 0.35),   # 评分低于此值卖出
        ("a_share_mode", True),
        ("stop_loss_pct", 0.05),
        ("_score_series", None),    # 外部传入的评分 Series
        ("position_mode", "full"),  # "full"=全仓 | "vol_target"=波动率目标
        ("target_vol", 0.15),       # 目标年化波动率（vol_target模式）
        ("vol_lookback", 20),       # 波动率回看窗口
        ("max_position_pct", 0.20), # 单策略仓位上限
        ("vol_cut_mode", "continuous"),  # "continuous"=连续缩放 | "stepwise"=阶梯缩放
        ("vol_step_thresholds", (0.8, 1.2, 2.0)),  # 阶梯阈值：低/中/高波动边界(vol_ratio)
        ("vol_step_factors", (1.5, 1.0, 0.5)),     # 阶梯仓位乘数：低/中/高波动
        ("verbose", False),
    )

    def __init__(self):
        self.score = self.p._score_series
        if self.score is None:
            raise ValueError("_score_series must be passed")

        self.entry_price = 0
        self.peak_price = 0
        self.trades = []
        self._returns = []  # 用于波动率目标计算
        self._score_values = self.score.values if hasattr(self.score, 'values') else self.score

        if len(self.score) > 20:
            self._buy_threshold = float(self.score.quantile(0.75))
            self._sell_threshold = float(self.score.quantile(0.25))
        else:
            self._buy_threshold = 0.65
            self._sell_threshold = 0.35

    def _vol_target_size(self) -> int:
        """波动率目标仓位: 高波动缩仓，低波动扩仓"""
        cash = self.broker.getcash()
        price = self.data.close[0]
        base_size = int(cash * self.p.max_position_pct / price)

        if self.p.position_mode != "vol_target" or len(self._returns) < self.p.vol_lookback:
            return base_size

        recent_ret = np.array(self._returns[-self.p.vol_lookback:])
        realized_vol = np.std(recent_ret) * np.sqrt(252)
        if realized_vol < 0.01:
            return base_size

        if self.p.vol_cut_mode == "stepwise":
            return self._vol_target_size_stepwise(base_size, realized_vol)
        else:
            return self._vol_target_size_continuous(base_size, realized_vol)

    def _vol_target_size_continuous(self, base_size: int, realized_vol: float) -> int:
        """连续缩放模式：线性调节仓位"""
        scale = self.p.target_vol / realized_vol
        scale = np.clip(scale, 0.3, 2.0)
        return int(base_size * scale)

    def _vol_target_size_stepwise(self, base_size: int, realized_vol: float) -> int:
        """阶梯缩放模式：三段式阈值调节仓位"""
        vol_ratio = realized_vol / self.p.target_vol
        thresholds = sorted(self.p.vol_step_thresholds)
        factors = self.p.vol_step_factors
        n_tiers = min(len(thresholds), len(factors))
        for i in range(n_tiers):
            if vol_ratio <= thresholds[i]:
                scale = factors[i]
                break
        else:
            scale = factors[-1]
        scale = np.clip(scale, 0.3, 2.0)
        return int(base_size * scale)

    def log(self, msg: str):
        if self.p.verbose:
            try:
                dt = self.datas[0].datetime.date(0)
                logger.info(f"[{dt}] Factor: {msg}")
            except Exception:
                pass

    def _get_score(self) -> float:
        """获取当前 Bar 的因子评分（按 bar 序号对齐）"""
        try:
            i = len(self) - 1  # backtrader len(self) is 1-based at first bar
            if 0 <= i < len(self._score_values):
                s = self._score_values[i]
                if not np.isnan(s):
                    return float(s)
        except Exception:
            pass
        return 0.5

    def next(self):
        s = self._get_score()

        if not self.position:
            if s > self._buy_threshold:
                size = self._vol_target_size()
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    self.peak_price = self.data.close[0]
                    self.log(f"BUY {size}股 @ {self.data.close[0]:.2f} (score={s:.3f})")
            # 记录收益用于波动率计算（无论是否持仓）
            if len(self) > 1:
                prev_close = self.data.close[-1]
                if prev_close > 0:
                    self._returns.append(self.data.close[0] / prev_close - 1)
        else:
            # 记录收益
            if len(self) > 1 and self.data.close[-1] > 0:
                self._returns.append(self.data.close[0] / self.data.close[-1] - 1)

            loss = self.data.close[0] / self.entry_price - 1
            if loss < -self.p.stop_loss_pct:
                self.close()
                self.log(f"STOP LOSS @ {self.data.close[0]:.2f}")
                return

            if self.data.close[0] > self.peak_price:
                self.peak_price = self.data.close[0]

            if s < self._sell_threshold:
                self.close()
                self.log(f"SELL @ {self.data.close[0]:.2f} (score={s:.3f})")


# ============================================================
# 命令行测试
# ============================================================
# python backtest/strategies/factor_strategy.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.analysis.factor_miner import FactorMiner
    from backtest.engine.bt_runner import run_backtest

    print("=" * 60)
    print("因子选股策略 — 端到端测试")
    print("=" * 60)

    df = fetch_index_daily("沪深300", "20200101", "20260524")

    # 1. 因子挖掘
    fm = FactorMiner(df)
    fm.mine()
    fm.ic_decay_report(top_n=5)

    # 2. 生成评分
    scores = fm.multi_factor_score(deduplicated=True)

    # 3. 回测
    print("\n--- 因子策略回测 ---")
    result = run_backtest(FactorStrategy, df, _score_series=scores)
    s = result.get("sharpe") or 0
    wr = result.get("win_rate", 0) or 0
    print(f"年化={result['annual_return']*100:.1f}% | "
          f"回撤={result['drawdown']*100:.1f}% | "
          f"夏普={s:.2f} | "
          f"交易={result['total_trades']}笔 | "
          f"胜率={wr*100:.0f}%")
