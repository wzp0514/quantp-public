"""vnpy 经典双均线策略。"""
import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class VnpyDoubleMaStrategy(BaseStrategy):
    """
    来源: vnpy 社区（国内最知名的开源量化框架）
    经典双均线策略：金叉全仓买入，死叉全部卖出
    参数: fast=5, slow=20
    """
    params = (("fast", 5), ("slow", 20))

    def __init__(self):
        super().__init__()
        self.ma_f = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.ma_s = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.cross = bt.indicators.CrossOver(self.ma_f, self.ma_s)
        self.log(f"vnpy双均线: fast={self.p.fast}, slow={self.p.slow}")

    def next(self):
        if not self.position and self.cross > 0:
            size = int(self.broker.getcash() / self.data.close[0])
            if size > 0:
                self.buy(size=size)
        elif self.position and self.cross < 0:
            self.close()
