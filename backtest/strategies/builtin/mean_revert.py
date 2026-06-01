"""
均值回归策略（Mean Reversion）

逻辑：
  - 价格跌到远低于均线（超跌）→ 买入，赌它会反弹回均线
  - 价格回到均线附近 → 卖出

核心假设：价格短期偏离均值后会回归。这和动量策略相反——
动量赌趋势继续，均值回归赌趋势反转。

风险：如果跌是因为基本面恶化（价值陷阱），可能不会回归，
一跌再跌。

参数：
  - period: 均线周期，默认 20
  - threshold: 偏离多少标准差算"超跌"，默认 2.0
               （价格低于均线 2 个标准差以上才买入）

回测重点观察：
  - 不回归的情况（跌了继续跌）
  - 大熊市中过早抄底
"""

import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class MeanRevertStrategy(BaseStrategy):
    params = (
        ("period", 20),
        ("threshold", 2.0),
    )

    def __init__(self):
        super().__init__()

        # 计算均线和标准差
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.period
        )
        self.std = bt.indicators.StandardDeviation(
            self.data.close, period=self.params.period
        )

        # Z-score = (当前价 - 均价) / 标准差
        # Z < -threshold → 价格远低于均值（超跌，买入信号）
        # Z > 0 → 价格回到均值以上（卖出信号）
        self.zscore = (self.data.close - self.sma) / self.std

        self.entry_price = 0

        self.log(f"均值回归初始化: period={self.params.period}, threshold={self.params.threshold}")

    def next(self):
        price = self.data.close[0]
        z = self.zscore[0]

        if not self.position:
            # Z-score 低于负阈值 → 超跌，买入
            if z < -self.params.threshold:
                size = int(self.broker.getcash() / price)
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = price
                    self.log(
                        f"超跌信号 → 买入 {size} 股 @ {price:.2f} "
                        f"(Z-score: {z:.2f}, 均线: {self.sma[0]:.2f})"
                    )

        else:
            # Z-score 回到 0 以上（回到均值）→ 卖出
            if z > 0:
                self.close()
                profit_pct = (price / self.entry_price - 1) * 100
                self.log(
                    f"回归均线信号 → 卖出 @ {price:.2f} "
                    f"(盈亏: {profit_pct:.2f}%, Z-score: {z:.2f})"
                )
