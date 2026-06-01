"""
双均线交叉策略（Moving Average Crossover）

逻辑：
  - 快线（短期均线）上穿慢线（长期均线）→ 买入（金叉）
  - 快线下穿慢线 → 卖出（死叉）

这是最基础的趋势跟踪策略。适合趋势明显的市场，
在震荡市中会反复打脸（连续小亏）。

参数：
  - fast: 快线周期，默认 5（5日均线）
  - slow: 慢线周期，默认 20（20日均线）

回测重点观察：
  - 震荡市中连续亏损次数（最大连亏）
  - 牛市中是否能抓住主升浪
"""

import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class MaCrossStrategy(BaseStrategy):
    params = (
        ("fast", 5),
        ("slow", 20),
    )

    def __init__(self):
        super().__init__()

        # 计算两条移动平均线
        self.ma_fast = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast
        )
        self.ma_slow = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow
        )

        # 记录交叉信号
        # crossover = 1 表示快线上穿慢线（金叉）
        # crossover = -1 表示快线下穿慢线（死叉）
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

        self.log(f"双均线初始化完成: fast={self.params.fast}, slow={self.params.slow}")

    def next(self):
        """每个交易日调用一次"""

        # 检查当前是否已有持仓
        if not self.position:

            # 无持仓 + 金叉（快线上穿慢线）→ 全仓买入
            if self.crossover > 0:
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
                    self.log(f"金叉信号 → 买入 {size} 股")

        else:
            # 有持仓 + 死叉（快线下穿慢线）→ 全部卖出
            if self.crossover < 0:
                self.close()
                self.log("死叉信号 → 全部卖出")
