"""
网格交易策略（Grid Trading）

逻辑：
  - 设定一个价格区间（上限/下限）和网格数量
  - 价格每跌破一个网格线 → 买入一份
  - 价格每涨回上一个网格线 → 卖出一份

本质是"低买高卖赚波动"。在横盘震荡市场中非常好用，
因为价格反复穿越网格线，每穿一次就赚一次差价。

最大风险：单边大跌。价格跌破下限后，所有网格都会成交，
变成满仓被套。没有止损的话会一直持有。

参数：
  - grid_count: 网格层数，默认 10
  - upper_pct: 上限（相对当前价的百分比），默认 10%
  - lower_pct: 下限（相对当前价的百分比），默认 -10%

回测重点观察：
  - 单边大跌时满仓套牢
  - 大牛市中过早卖出所有仓位（踏空）
"""

import backtrader as bt
from backtest.engine.bt_runner import BaseStrategy


class GridStrategy(BaseStrategy):
    params = (
        ("grid_count", 10),
        ("upper_pct", 0.10),
        ("lower_pct", -0.10),
    )

    def __init__(self):
        super().__init__()

        # 网格是基于回测开始时的价格计算的
        # 实际运行时应该用启动时的价格，这里用第一个 Bar 的价格
        self.base_price = None  # 基准价（第一个 Bar 确定）
        self.grids = []         # 网格线价格（从高到低）
        self.last_cross = 0     # 上一次在哪个网格

        self.log(
            f"网格策略初始化: grids={self.params.grid_count}, "
            f"range=[{self.params.lower_pct:.1%}, {self.params.upper_pct:.1%}]"
        )

    def next(self):
        price = self.data.close[0]

        # 第一个 Bar：用收盘价作为基准价，建立网格
        if self.base_price is None:
            self.base_price = price
            step = (self.params.upper_pct - self.params.lower_pct) / self.params.grid_count
            # 从下到上：下轨、下轨+step、...、上轨
            self.grids = [
                self.base_price * (1 + self.params.lower_pct + i * step)
                for i in range(self.params.grid_count + 1)
            ]
            self.last_cross = self._find_grid(price)
            self.log(
                f"基准价: {self.base_price:.2f}, "
                f"网格区间: [{self.grids[0]:.2f}, {self.grids[-1]:.2f}], "
                f"当前网格: {self.last_cross}"
            )
            return

        current_grid = self._find_grid(price)

        # 价格向上穿越网格线 → 卖出（低买高卖）
        if current_grid > self.last_cross and self.position:
            sell_size = self.position.size // (self.params.grid_count - self.last_cross + 1)
            if sell_size > 0:
                self.sell(size=sell_size)
                self.log(
                    f"上穿网格 {current_grid} → 卖出 {sell_size} 股 @ {price:.2f}"
                )

        # 价格向下穿越网格线 → 买入
        elif current_grid < self.last_cross:
            buy_size = int(self.broker.getcash() / price / (self.params.grid_count - current_grid))
            if buy_size > 0:
                self.buy(size=buy_size)
                self.log(
                    f"下穿网格 {current_grid} → 买入 {buy_size} 股 @ {price:.2f}"
                )

        self.last_cross = current_grid

    def _find_grid(self, price: float) -> int:
        """找到价格落在哪个网格（0 = 最底层，grid_count = 最顶层）"""
        for i in range(len(self.grids) - 1):
            if self.grids[i] <= price < self.grids[i + 1]:
                return i
        if price < self.grids[0]:
            return 0       # 跌破下限
        return self.params.grid_count  # 涨破上限
