"""
StrategyGuard — 策略级熔断器

监控单个策略的实时行为，检测异常并独立熔断。
与账户级 risk_engine 互补：那里管"账户会不会爆仓"，
这里管"这个策略是不是跑偏了"。

检测维度：
  1. 信号频率漂移 — 信号数量偏离回测期望 > 阈值 → 疑似过拟合失效
  2. 成交价偏离 — 实盘成交价与理论信号价偏离 > 阈值 → 滑点异常/流动性问题
  3. 单策略回撤 — 策略单独回撤超限 → 独立熔断
  4. 连续亏损笔数 — 该策略连亏超记录 → 暂停

用法
--------
>>> guard = StrategyGuard("布林带回归", max_drawdown=0.15)
>>> guard.check_signal(bar_index=150, backtest_expected_count=30)
>>> guard.check_trade(price=10.5, signal_price=10.2)
>>> if guard.is_blown():
...     print(f"熔断: {guard.reason}")
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from config.log import get_logger

logger = get_logger("strategy_guard")


@dataclass
class StrategyMetrics:
    """策略实时指标（滚动更新）"""
    total_signals: int = 0
    total_trades: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    total_pnl: float = 0.0
    avg_signal_price: float = 0.0
    avg_trade_price: float = 0.0
    price_deviation_sum: float = 0.0
    price_deviation_count: int = 0


class StrategyGuard:
    """
    策略级熔断器。

    参数
    ----------
    name : str — 策略名
    max_drawdown : float — 单策略最大回撤（0.15=15%）
    max_consecutive_losses : int — 连续亏损上限
    signal_freq_drift_pct : float — 信号频率允许偏差（0.5=50%）
    price_slippage_pct : float — 成交价允许偏离（0.02=2%）
    """

    def __init__(
        self,
        name: str,
        max_drawdown: float = 0.15,
        max_consecutive_losses: int = 5,
        signal_freq_drift_pct: float = 0.50,
        price_slippage_pct: float = 0.02,
    ):
        self.name = name
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.signal_freq_drift_pct = signal_freq_drift_pct
        self.price_slippage_pct = price_slippage_pct

        self._metrics = StrategyMetrics()
        self._blown = False
        self._blown_reason = ""
        self._created_at = time.time()
        self._backtest_baseline: Optional[dict] = None

    # ── 公开接口 ──

    def set_backtest_baseline(self, total_signals: int, avg_signal_interval: float):
        """设置回测基准（用于信号频率漂移检测）"""
        self._backtest_baseline = {
            "total_signals": total_signals,
            "avg_interval": avg_signal_interval,
        }

    def set_liquidity_guard(self, guardian) -> None:
        """设置 Guardian 实例以获取流动性状态（可选）"""
        self._guardian = guardian

    def check_signal(self, bar_index: int) -> bool:
        """
        信号频率漂移检测。
        在回放/实盘中每产生一个信号时调用。
        bar_index: 当前bar序号（从开始算起）
        """
        if self._blown:
            return False

        self._metrics.total_signals += 1

        if self._backtest_baseline:
            expected_sigs = self._backtest_baseline["total_signals"]
            if expected_sigs > 0:
                ratio = self._metrics.total_signals / (expected_sigs * bar_index / max(bar_index, 1))
                # 简化为：信号密度 vs 时间
                expected_at_this_point = expected_sigs * (bar_index / self._backtest_baseline.get("total_bars", bar_index + 1))
                if expected_at_this_point > 5:
                    drift = abs(self._metrics.total_signals - expected_at_this_point) / expected_at_this_point
                    if drift > self.signal_freq_drift_pct:
                        self._blow(f"信号频率漂移: 实际{self._metrics.total_signals} vs 预期{expected_at_this_point:.0f}")
                        return False
        return True

    def check_trade(self, price: float, signal_price: float = 0.0) -> bool:
        """成交价偏离检测"""
        if self._blown:
            return False

        self._metrics.total_trades += 1
        self._metrics.avg_trade_price = (
            (self._metrics.avg_trade_price * (self._metrics.total_trades - 1) + price)
            / self._metrics.total_trades
        )

        if signal_price > 0:
            deviation = abs(price - signal_price) / signal_price
            self._metrics.price_deviation_sum += deviation
            self._metrics.price_deviation_count += 1

            if deviation > self.price_slippage_pct:
                avg_dev = self._metrics.price_deviation_sum / self._metrics.price_deviation_count
                if avg_dev > self.price_slippage_pct:
                    self._blow(f"成交价持续偏离: 平均{avg_dev:.2%} > {self.price_slippage_pct:.2%}")
                    return False
        return True

    def update_equity(self, equity: float, trade_pnl: float = 0.0):
        """更新权益曲线 + 回撤检测 + 连续亏损检测"""
        if self._blown:
            return

        # 连续亏损
        if trade_pnl < 0:
            self._metrics.consecutive_losses += 1
            self._metrics.max_consecutive_losses = max(
                self._metrics.max_consecutive_losses, self._metrics.consecutive_losses
            )
        else:
            self._metrics.consecutive_losses = 0

        if self._metrics.consecutive_losses >= self.max_consecutive_losses:
            self._blow(f"连续亏损{self._metrics.consecutive_losses}笔 ≥ {self.max_consecutive_losses}")
            return

        # 回撤
        self._metrics.peak_equity = max(self._metrics.peak_equity, equity)
        if self._metrics.peak_equity > 0:
            dd = (self._metrics.peak_equity - equity) / self._metrics.peak_equity
            self._metrics.current_drawdown = dd
            self._metrics.max_drawdown = max(self._metrics.max_drawdown, dd)

            if dd > self.max_drawdown:
                self._blow(f"策略回撤{dd:.1%} > {self.max_drawdown:.1%}")
                return

    def is_blown(self) -> bool:
        return self._blown

    @property
    def reason(self) -> str:
        return self._blown_reason

    @property
    def metrics(self) -> StrategyMetrics:
        return self._metrics

    def status(self) -> dict:
        """返回策略当前状态摘要"""
        return {
            "name": self.name,
            "active": not self._blown,
            "blown_reason": self._blown_reason,
            "signals": self._metrics.total_signals,
            "trades": self._metrics.total_trades,
            "consecutive_losses": self._metrics.consecutive_losses,
            "max_drawdown": self._metrics.max_drawdown,
            "avg_price_deviation": (
                self._metrics.price_deviation_sum / self._metrics.price_deviation_count
                if self._metrics.price_deviation_count > 0 else 0
            ),
        }

    def reset(self):
        """重置熔断（谨慎使用——只在手动审查后调用）"""
        self._metrics = StrategyMetrics()
        self._blown = False
        self._blown_reason = ""

    def _blow(self, reason: str):
        self._blown = True
        self._blown_reason = reason
        logger.warning(f"[{self.name}] 策略熔断: {reason}")


# ============================================================
# 命令行测试
# ============================================================
# python live/risk/strategy_guard.py

if __name__ == "__main__":
    print("=" * 60)
    print("StrategyGuard Demo")
    print("=" * 60)

    guard = StrategyGuard("test_strategy", max_drawdown=0.10, max_consecutive_losses=3)

    # 模拟正常交易
    print("Simulating trades...")
    guard.update_equity(100000, 500)
    print(f"  After win:  blown={guard.is_blown()}, consec={guard.metrics.consecutive_losses}")

    guard.update_equity(101000, -2000)
    guard.update_equity(100000, -1500)
    guard.update_equity(98000, -1200)
    print(f"  After 3 losses: blown={guard.is_blown()}, reason={guard.reason}")
    print(f"  Status: {guard.status()}")
