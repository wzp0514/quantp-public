"""ATR 移动止损策略。"""
import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class ATRTrailingStopStrategy(BaseStrategy):
    """
    来源: 量化交易常见模式
    均线金叉入场 + ATR动态止损（波动大放宽，波动小收紧）
    """
    params = (("ma_fast", 10), ("ma_slow", 30), ("atr_period", 14), ("atr_mult", 2.0))

    def __init__(self):
        super().__init__()
        self.ma_f = bt.indicators.SMA(self.data.close, period=self.p.ma_fast)
        self.ma_s = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.cross = bt.indicators.CrossOver(self.ma_f, self.ma_s)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.peak = 0
        self.log(f"ATR止损: ma({self.p.ma_fast},{self.p.ma_slow}), atr({self.p.atr_period})x{self.p.atr_mult}")

    def next(self):
        price = self.data.close[0]
        if not self.position and self.cross > 0:
            size = int(self.broker.getcash() / price)
            if size > 0:
                self.buy(size=size)
                self.peak = price
        elif self.position:
            if price > self.peak:
                self.peak = price
            if price <= self.peak - self.p.atr_mult * self.atr[0]:
                self.close()
