"""
Freqtrade→Backtrader 策略移植 — 3个代表性策略。

从 176 个 Freqtrade 指标中挑选 3 个最有代表性的，改写为 Backtrader 策略，
注册到策略总表。
"""

import backtrader as bt
import numpy as np
from backtest.engine.bt_runner import BaseStrategy


class VIDYACrossStrategy(BaseStrategy):
    """
    VIDYA 自适应均线交叉。

    用 VIDYA（Variable Index Dynamic Average）替代固定均线，
    CMO 动态调节平滑度，趋势中更敏感、震荡中更平滑。
    """
    params = (("vidya_period", 14), ("smoothing", 9))

    def __init__(self):
        super().__init__()
        from backtest.freqtrade_indicators import VIDYA
        self.vidya = VIDYA(np.array([self.data.close[i] for i in range(len(self.data))]),
                           self.p.vidya_period, self.p.smoothing)
        self._idx = 0
        self._signal = 0

    def next(self):
        super().next()
        if self._idx < self.p.vidya_period + 1:
            self._idx += 1
            return
        price = self.data.close[0]
        vidya_val = self.vidya[self._idx]
        if np.isnan(vidya_val):
            self._idx += 1
            return
        if price > vidya_val and self._signal <= 0:
            self.buy(size=self._calc_size(price), reason="VIDYA金叉")
            self._signal = 1
        elif price < vidya_val and self._signal >= 0:
            self.sell(size=self.position.size, reason="VIDYA死叉")
            self._signal = -1
        self._idx += 1


class STCStrategy(BaseStrategy):
    """
    Schaff Trend Cycle 策略。

    STC > 75 超买→卖出，STC < 25 超卖→买入。
    比 MACD 反应更快，减少滞后。
    """
    params = (("stc_fast", 23), ("stc_slow", 50), ("stc_cycle", 10))

    def __init__(self):
        super().__init__()
        from backtest.freqtrade_indicators import STC
        close_arr = np.array([self.data.close[i] for i in range(len(self.data))])
        self.stc_vals = STC(close_arr, self.p.stc_fast, self.p.stc_slow, self.p.stc_cycle)
        self._idx = 0

    def next(self):
        super().next()
        if self._idx < self.p.stc_slow + self.p.stc_cycle:
            self._idx += 1
            return
        stc = self.stc_vals[self._idx]
        if np.isnan(stc):
            self._idx += 1
            return
        if stc < 25 and self.position.size == 0:
            self.buy(size=self._calc_size(self.data.close[0]), reason="STC超卖")
        elif stc > 75 and self.position.size > 0:
            self.sell(size=self.position.size, reason="STC超买")
        self._idx += 1


class KelterBreakoutStrategy(BaseStrategy):
    """
    凯尔特纳通道突破策略。

    TKE (Triple Keltner) 上轨突破买入，下轨跌破卖出。
    """
    params = (("ke_period", 20), ("ke_atr", 10), ("ke_mult", 1.5))

    def __init__(self):
        super().__init__()
        self._idx = 0
        self._channel = None

    def next(self):
        super().next()
        if self._idx < self.p.ke_period:
            self._idx += 1
            return
        from backtest.freqtrade_indicators import TKE
        high = np.array([self.data.high[i] for i in range(self._idx + 1)])
        low = np.array([self.data.low[i] for i in range(self._idx + 1)])
        close = np.array([self.data.close[i] for i in range(self._idx + 1)])
        channel = TKE(high, low, close, self.p.ke_period, self.p.ke_atr, self.p.ke_mult)

        price = self.data.close[0]
        upper = channel["upper"][-1]
        lower = channel["lower"][-1]

        if price > upper and self.position.size == 0:
            self.buy(size=self._calc_size(price), reason="Keltner上轨突破")
        elif price < lower and self.position.size > 0:
            self.sell(size=self.position.size, reason="Keltner下轨跌破")
        self._idx += 1
