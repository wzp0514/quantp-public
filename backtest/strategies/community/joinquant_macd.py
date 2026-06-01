"""聚宽 MACD 金叉死叉策略。"""
import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class JoinQuantMACDStrategy(BaseStrategy):
    """
    来源: 聚宽社区（joinquant.com）
    MACD 金叉买入，死叉卖出
    参数: fast=12, slow=26, signal=9
    """
    params = (("fast", 12), ("slow", 26), ("signal", 9))

    def __init__(self):
        super().__init__()
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal,
        )
        self.cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.log(f"聚宽MACD: ({self.p.fast},{self.p.slow},{self.p.signal})")

    def next(self):
        if not self.position and self.cross > 0:
            size = int(self.broker.getcash() / self.data.close[0])
            if size > 0:
                self.buy(size=size)
        elif self.position and self.cross < 0:
            self.close()
