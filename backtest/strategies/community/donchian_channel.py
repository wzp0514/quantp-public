"""Donchian 通道突破策略。"""
import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class DonchianChannelStrategy(BaseStrategy):
    """
    来源: Backtrader 社区文档（backtrader.com）
    突破 N 日最高价买入，跌破 N 日最低价卖出
    """
    params = (("period", 20),)

    def __init__(self):
        super().__init__()
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.period)
        self.log(f"Donchian通道: period={self.p.period}")

    def next(self):
        price = self.data.close[0]
        if not self.position and price >= self.highest[-1]:
            size = int(self.broker.getcash() / price)
            if size > 0:
                self.buy(size=size)
        elif self.position and price <= self.lowest[-1]:
            self.close()
