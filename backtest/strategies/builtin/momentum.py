"""
动量策略（Momentum）

逻辑：
  - 计算过去 N 个月的累计收益率
  - 收益率 > 阈值（正动量）→ 买入，持有 M 个月
  - 收益率 < 阈值 → 卖出

动量效应：过去涨得好的资产，未来一段时间会继续涨。
这在全球金融市场中被广泛验证有效，但存在"动量崩溃"风险——
市场风格突然切换时，动量因子会一次性大亏。

参数：
  - lookback: 回看周期（交易日），默认 126（约 6 个月）
  - hold_period: 持有周期（交易日），默认 21（约 1 个月）
  - threshold: 动量阈值，默认 0.03（过去 6 个月涨超 3% 才买入）

回测重点观察：
  - 动量崩溃时的一次性大亏
  - 震荡市中频繁进出导致的交易成本累积
"""

import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class MomentumStrategy(BaseStrategy):
    params = (
        ("lookback", 126),
        ("hold_period", 21),
        ("threshold", 0.03),
    )

    def __init__(self):
        super().__init__()

        # 计算过去 N 个交易日的收益率
        # RateOfChange = (今天收盘价 / N天前收盘价 - 1)
        self.roc = bt.indicators.RateOfChange(
            self.data.close, period=self.params.lookback
        )

        # 持仓计数器（买入后持有了几天）
        self.bars_held = 0

        self.log(
            f"动量策略初始化: lookback={self.params.lookback}天, "
            f"hold={self.params.hold_period}天, threshold={self.params.threshold:.1%}"
        )

    def next(self):
        momentum = self.roc[0] / 100  # RateOfChange 返回百分比，除以 100 转小数

        if not self.position:
            # 动量超过阈值 → 买入
            if momentum > self.params.threshold:
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
                    self.bars_held = 0
                    self.log(f"动量信号 → 买入 {size} 股 (动量: {momentum:.2%})")

        else:
            self.bars_held += 1

            # 持仓时间到 → 卖出（不管盈亏）
            if self.bars_held >= self.params.hold_period:
                self.close()
                self.log(f"持仓到期 → 卖出 (持有了 {self.bars_held} 天)")

            # 动量转负 → 提早卖出
            elif momentum < 0:
                self.close()
                self.log(f"动量转负 → 提前卖出 (动量: {momentum:.2%})")
