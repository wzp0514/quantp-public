"""
共振策略 — 多信号确认 + 市场区制过滤

专业量化减少噪声的三种方式合一：
  1. 多信号共振 — N 个独立信号中至少 M 个同时触发才交易
  2. 区制过滤 — 只在牛市/震荡市做多，熊市自动空仓
  3. 因子嵌入 — 可接入 factor_miner 评分作为额外共振条件

对比：普通策略（单信号触发→交易）vs 共振策略（≥2信号共鸣才交易）
共振策略大幅减少假信号，代价是可能错过一些真信号。

用法
--------
>>> from backtest.strategies.experimental.resonance import ResonanceStrategy
>>> # 自动择时：多信号共振 + 区制过滤 + 因子评分
"""

import backtrader as bt
import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("resonance")


class ResonanceStrategy(bt.Strategy):
    """
    共振策略：至少 require_signals 个独立信号同时触发才入场。

    信号清单（4个独立信号源）：
      1. MA金叉（趋势信号）
      2. RSI超卖（反转信号）
      3. 放量突破（量能确认）
      4. 因子评分 > 阈值（基本面确认，需传入 _factor_scores）

    另加：
      5. 区制过滤（regime != BEAR）

    默认 require_signals=3（5个信号中至少3个同时满足）。
    """

    params = (
        # Signal parameters
        ("ma_fast", 10),
        ("ma_slow", 30),
        ("rsi_period", 14),
        ("rsi_low", 35),
        ("vol_mult", 1.5),
        ("vol_period", 20),
        # Resonance
        ("require_signals", 3),        # 至少几个信号共振
        # Regime filter
        ("regime_enabled", True),      # 启用区制过滤
        ("regime_lookback", 60),
        # Factor
        ("factor_enabled", False),     # 启用因子评分
        ("_factor_scores", None),      # 外部传入的因子评分 Series
        # Standard
        ("a_share_mode", True),
        ("stop_loss_pct", 0.05),
        ("verbose", False),
    )

    def __init__(self):
        # Trend signal: MA crossover
        self.ma_f = bt.indicators.SMA(self.data.close, period=self.p.ma_fast)
        self.ma_s = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.crossover = bt.indicators.CrossOver(self.ma_f, self.ma_s)

        # Reversal signal: RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # Volume signal
        self.vol_avg = bt.indicators.SMA(self.data.volume, period=self.p.vol_period)

        self.entry_price = 0
        self.peak_price = 0
        self.trades = []

        # Factor scores
        self._factor_scores = self.p._factor_scores
        if self._factor_scores is not None:
            self._fscore_values = self._factor_scores.values if hasattr(self._factor_scores, 'values') else self._factor_scores
        else:
            self._fscore_values = None

        # Regime filter: pre-compute at init time
        self._regime_allowed = None
        if self.p.regime_enabled:
            self._init_regime()

    def _init_regime(self):
        """预计算区制信号（避免 next() 里反复调 heavy 计算）"""
        try:
            close_vals = self.data.close.array
            # Simple regime detection: use 60d rolling return
            ret_60d = pd.Series(close_vals).pct_change(self.p.regime_lookback).fillna(0).values
            self._regime_allowed = ret_60d > -0.05  # 60日跌幅 < 5% = 非熊市
        except Exception:
            self._regime_allowed = None

    def _is_regime_ok(self) -> bool:
        """当前是否允许交易（非熊市）"""
        if not self.p.regime_enabled or self._regime_allowed is None:
            return True
        try:
            i = len(self) - 1
            if 0 <= i < len(self._regime_allowed):
                return bool(self._regime_allowed[i])
        except Exception:
            pass
        return True

    def _get_factor_score(self) -> float:
        """获取当前 bar 的因子评分"""
        if self._fscore_values is None:
            return 0.5
        try:
            i = len(self) - 1
            if 0 <= i < len(self._fscore_values):
                return float(self._fscore_values[i])
        except Exception:
            pass
        return 0.5

    def log(self, msg: str):
        if self.p.verbose:
            try:
                dt = self.datas[0].datetime.date(0)
                logger.info(f"[{dt}] Resonance: {msg}")
            except Exception:
                pass

    def _count_signals(self) -> tuple[int, list[str]]:
        """统计当前有多少个信号触发"""
        signals = []
        # 1. MA金叉
        if self.crossover[0] > 0:
            signals.append("MA_golden")
        # 2. RSI超卖
        if self.rsi[0] < self.p.rsi_low:
            signals.append("RSI_oversold")
        # 3. 放量
        if self.data.volume[0] > self.vol_avg[0] * self.p.vol_mult:
            signals.append("Volume_surge")
        # 4. 因子评分（可选）
        if self.p.factor_enabled and self._fscore_values is not None:
            if self._get_factor_score() > 0.6:
                signals.append("Factor_strong")
        # 5. 区制过滤（非熊市）
        if self._is_regime_ok():
            signals.append("Regime_OK")
        else:
            # 区制不通过 → 减分（算作缺失的一个信号）
            return 0, ["Regime_BEAR(rejected)"]

        return len(signals), signals

    def next(self):
        n_signals, active = self._count_signals()

        if not self.position:
            # 入场：多信号共振
            if n_signals >= self.p.require_signals:
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    self.peak_price = self.data.close[0]
                    self.log("BUY (resonance=%d: %s)" % (n_signals, ",".join(active)))
        else:
            # 止损
            loss = self.data.close[0] / self.entry_price - 1
            if loss < -self.p.stop_loss_pct:
                self.close()
                self.log("STOP LOSS")
                return

            if self.data.close[0] > self.peak_price:
                self.peak_price = self.data.close[0]

            # 出场：信号衰减（不足共振时出场）
            if n_signals < max(1, self.p.require_signals - 1):
                self.close()
                self.log("EXIT (signals=%d < threshold)" % n_signals)


# ============================================================
# 命令行测试
# ============================================================
# python backtest/strategies/resonance.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fd
    from backtest.engine.bt_runner import run_backtest
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    from backtest.analysis.factor_miner import FactorMiner

    df = fd("沪深300", "20200101", "20260524")

    # Factor scores for resonance
    fm = FactorMiner(df)
    fm.mine()
    scores = fm.multi_factor_score(deduplicated=True)

    print()
    print("=" * 65)
    print("  Resonance Strategy Comparison")
    print("=" * 65)

    # Baseline
    baseline = run_backtest(MaCrossStrategy, df, fast=5, slow=20)
    bs = baseline.get("sharpe") or 0
    print("MA Cross (single signal): AR=%.1f%% DD=%.1f%% SR=%.2f T=%d" % (
        baseline["annual_return"] * 100, baseline["drawdown"] * 100, bs, baseline["total_trades"]))

    # Resonance (3 signals)
    r3 = run_backtest(ResonanceStrategy, df, require_signals=3, regime_enabled=True,
                      factor_enabled=False, verbose=False)
    s3 = r3.get("sharpe") or 0
    print("Resonance(3-signal+regime): AR=%.1f%% DD=%.1f%% SR=%.2f T=%d" % (
        r3["annual_return"] * 100, r3["drawdown"] * 100, s3, r3["total_trades"]))

    # Resonance (2 signals, easier to trigger)
    r2 = run_backtest(ResonanceStrategy, df, require_signals=2, regime_enabled=True,
                      factor_enabled=False, verbose=False)
    s2 = r2.get("sharpe") or 0
    print("Resonance(2-signal+regime): AR=%.1f%% DD=%.1f%% SR=%.2f T=%d" % (
        r2["annual_return"] * 100, r2["drawdown"] * 100, s2, r2["total_trades"]))

    # Resonance + Factor
    rf = run_backtest(ResonanceStrategy, df, require_signals=2, regime_enabled=True,
                      factor_enabled=True, _factor_scores=scores, verbose=False)
    sf = rf.get("sharpe") or 0
    print("Resonance(2-signal+regime+FACTOR): AR=%.1f%% DD=%.1f%% SR=%.2f T=%d" % (
        rf["annual_return"] * 100, rf["drawdown"] * 100, sf, rf["total_trades"]))
