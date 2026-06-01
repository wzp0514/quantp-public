"""Freqtrade RSI 均值回归策略。"""
import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class FreqtradeRSIStrategy(BaseStrategy):
    """
    来源: Freqtrade 社区（48K Stars）
    RSI 均值回归：RSI < oversold 超卖买入，RSI > overbought 超买卖出
    参数: rsi_period=14, oversold=30, overbought=70
    """
    params = (("rsi_period", 14), ("oversold", 30), ("overbought", 70))

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.log(f"Freqtrade RSI: period={self.p.rsi_period}, range=[{self.p.oversold},{self.p.overbought}]")

    def next(self):
        if not self.position and self.rsi[0] < self.p.oversold:
            size = int(self.broker.getcash() / self.data.close[0])
            if size > 0:
                self.buy(size=size)
        elif self.position and self.rsi[0] > self.p.overbought:
            self.close()
