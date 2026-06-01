"""
布林带回归策略（Bollinger Bands）

逻辑：
  - 价格触及下轨（超卖）→ 买入
  - 价格回到中轨 → 卖出

布林带由三条线组成：
  - 中轨 = N 日移动均线
  - 上轨 = 中轨 + K × 标准差
  - 下轨 = 中轨 - K × 标准差

价格在上下轨之间运行是常态，触及上下轨意味着"过度偏离"，
大概率会回归到中轨附近。但如果市场进入单边趋势，
价格会沿着轨道走，连续触及上/下轨而不回归——这就是风险。

参数：
  - period: 均线周期，默认 20
  - devfactor: 标准差倍数，默认 2.0

回测重点观察：
  - 单边趋势中不回归导致的连续亏损
  - 低位抄底后是否及时反弹
"""

import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class BollingerStrategy(BaseStrategy):
    params = (
        ("period", 20),
        ("devfactor", 2.0),
    )

    def __init__(self):
        super().__init__()

        # 布林带指标
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.period,
            devfactor=self.params.devfactor,
        )
        # 三条线
        self.mid = self.boll.lines.mid    # 中轨（均线）
        self.top = self.boll.lines.top    # 上轨
        self.bot = self.boll.lines.bot    # 下轨

        # 记录上次买入价（用于判断何时卖出）
        self.entry_price = 0

        self.log(f"布林带初始化: period={self.params.period}, devfactor={self.params.devfactor}")

    def next(self):
        price = self.data.close[0]

        if not self.position:
            # 价格跌破下轨 → 超卖，买入
            if price <= self.bot[0]:
                size = int(self.broker.getcash() / price)
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = price
                    self.log(
                        f"触下轨信号 → 买入 {size} 股 @ {price:.2f} "
                        f"(下轨: {self.bot[0]:.2f}, 中轨: {self.mid[0]:.2f})"
                    )

        else:
            # 价格回到中轨以上 → 卖出
            if price >= self.mid[0]:
                self.close()
                profit_pct = (price / self.entry_price - 1) * 100
                self.log(
                    f"回中轨信号 → 卖出 @ {price:.2f} "
                    f"(盈亏: {profit_pct:.2f}%, 中轨: {self.mid[0]:.2f})"
                )
