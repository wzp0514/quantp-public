"""海龟交易法则策略。"""
import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class TurtleTradingStrategy(BaseStrategy):
    """
    来源: 海龟交易法则（Richard Dennis, 1983）
    突破20日高点买入，跌破10日低点卖出，盈利后每0.5ATR加仓（最多4次）
    参数: entry_period=20, exit_period=10, atr_period=20
    """
    params = (("entry_period", 20), ("exit_period", 10), ("atr_period", 20))

    def __init__(self):
        super().__init__()
        self.entry_high = bt.indicators.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.indicators.Lowest(self.data.low, period=self.p.exit_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.add_count = 0
        self.last_add_price = 0
        self.log(f"海龟交易: entry={self.p.entry_period}日高点, exit={self.p.exit_period}日低点")

    def next(self):
        price = self.data.close[0]
        if not self.position:
            if price >= self.entry_high[-1]:
                size = int(self.broker.getcash() * 0.5 / price)
                if size > 0:
                    self.buy(size=size)
                    self.last_add_price = price
                    self.add_count = 1
        else:
            if self.add_count < 4 and price >= self.last_add_price + 0.5 * self.atr[0]:
                size = int(self.broker.getcash() * 0.25 / price)
                if size > 0:
                    self.buy(size=size)
                    self.last_add_price = price
                    self.add_count += 1
            if price <= self.exit_low[-1]:
                self.close()
