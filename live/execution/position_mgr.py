"""
仓位管理系统

跟踪当前持有哪些标的、持仓成本、盈亏情况。

核心概念：
  - 持仓（Position）：你持有一个标的的数量和成本信息
  - 平均成本：多次买入时，加权平均成本
  - 浮动盈亏（Unrealized P&L）：按当前市价计算的盈亏（还没卖，不算真的盈亏）
  - 已实现盈亏（Realized P&L）：卖出后锁定的盈亏

示例
--------
>>> pm = PositionManager()
>>> pm.update_position("sh000300", "buy", 100, 4000.0)
>>> pm.get_position("sh000300")
{'symbol': 'sh000300', 'size': 100, 'avg_cost': 4000.0, ...}
"""

from datetime import date, datetime
from typing import Optional

from config.log import get_logger
logger = get_logger("position_mgr")
class PositionManager:
    """
    仓位管理器。

    每次成交后调用 update_position() 更新持仓。
    """

    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, dict] = {}  # symbol → position dict
        self.trade_history: list[dict] = []   # 交易历史

    def update_position(
        self,
        symbol: str,
        side: str,
        size: int,
        price: float,
        trade_date: date = None,
    ) -> dict:
        """
        更新持仓（一次买或卖）

        参数
        ----------
        symbol : str
            标的代码
        side : str
            "buy" 或 "sell"
        size : int
            数量
        price : float
            成交价
        trade_date : date
            交易日期

        返回
        -------
        dict : 更新后的持仓信息
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "size": 0,
                "avg_cost": 0.0,
                "total_cost": 0.0,
                "realized_pnl": 0.0,
            }

        pos = self.positions[symbol]
        trade_value = price * size

        if side == "buy":
            # 买入：更新平均成本（加权平均）
            old_total_cost = pos["total_cost"]
            pos["size"] += size
            pos["total_cost"] += trade_value
            pos["avg_cost"] = pos["total_cost"] / pos["size"] if pos["size"] > 0 else 0
            self.cash -= trade_value

            logger.info(
                f"买入 {size} {symbol} @ {price:.2f} | "
                f"持仓: {pos['size']} | "
                f"均价: {pos['avg_cost']:.2f} | "
                f"剩余现金: {self.cash:,.2f}"
            )

        elif side == "sell":
            if size > pos["size"]:
                logger.error(
                    f"卖出数量 {size} > 持仓 {pos['size']} {symbol}！"
                    f"这是实盘不允许的操作——不能卖空。已调整为卖出 {pos['size']}。"
                )
                size = pos["size"]
                trade_value = price * size

            # 计算已实现盈亏
            realized = (price - pos["avg_cost"]) * size
            pos["realized_pnl"] += realized
            pos["size"] -= size
            pos["total_cost"] = pos["avg_cost"] * pos["size"] if pos["size"] > 0 else 0
            self.cash += trade_value

            logger.info(
                f"卖出 {size} {symbol} @ {price:.2f} | "
                f"盈亏: {realized:+.2f} | "
                f"剩余持仓: {pos['size']} | "
                f"剩余现金: {self.cash:,.2f}"
            )

        # 记录交易
        self.trade_history.append({
            "date": trade_date or date.today(),
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "value": trade_value,
        })

        return pos

    def get_position(self, symbol: str) -> Optional[dict]:
        """获取某个标的的持仓"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> list[dict]:
        """获取所有有持仓的标的"""
        return [p for p in self.positions.values() if p["size"] > 0]

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """
        计算浮动盈亏（按当前市价，持有仓位值多少钱 - 成本）

        示例
        --------
        >>> pnl = pm.get_unrealized_pnl("sh000300", 4100.0)
        >>> print(f"浮动盈亏: {pnl:+,.2f}")
        """
        pos = self.positions.get(symbol)
        if not pos or pos["size"] == 0:
            return 0.0
        return (current_price - pos["avg_cost"]) * pos["size"]

    def get_total_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """
        所有持仓的浮动盈亏总和

        参数
        ----------
        prices : dict
            symbol → 当前价格
        """
        total = 0.0
        for symbol, price in prices.items():
            total += self.get_unrealized_pnl(symbol, price)
        return total

    def get_total_value(self, prices: dict[str, float]) -> float:
        """当前总资产 = 现金 + 持仓市值"""
        holding_value = sum(
            pos["size"] * prices.get(symbol, pos["avg_cost"])
            for symbol, pos in self.positions.items()
            if pos["size"] > 0
        )
        return self.cash + holding_value

    def get_total_return(self, prices: dict[str, float]) -> float:
        """总收益率"""
        total_value = self.get_total_value(prices)
        return (total_value / self.initial_cash) - 1

    def get_exposure(self, prices: dict[str, float]) -> float:
        """总风险敞口（持仓市值 / 总资产）"""
        total_value = self.get_total_value(prices)
        if total_value <= 0:
            return 0.0
        holding_value = total_value - self.cash
        return holding_value / total_value

    def get_summary(self, prices: dict[str, float]) -> str:
        """仓位摘要（可打印的文本）"""
        if not self.get_all_positions():
            return "当前无持仓"

        lines = [f"{'标的':<12} {'数量':>8} {'均价':>10} {'现价':>10} {'盈亏':>12}"]
        lines.append("-" * 55)

        for pos in self.get_all_positions():
            symbol = pos["symbol"]
            current_price = prices.get(symbol, pos["avg_cost"])
            pnl = self.get_unrealized_pnl(symbol, current_price)
            lines.append(
                f"{symbol:<12} {pos['size']:>8} "
                f"{pos['avg_cost']:>10.2f} {current_price:>10.2f} {pnl:>+12.2f}"
            )

        total_pnl = sum(
            self.get_unrealized_pnl(p["symbol"], prices.get(p["symbol"], p["avg_cost"]))
            for p in self.get_all_positions()
        )
        lines.append("-" * 55)
        lines.append(f"现金: {self.cash:,.2f}  浮动盈亏: {total_pnl:+,.2f}  "
                     f"总资产: {self.get_total_value(prices):,.2f}")
        return "\n".join(lines)
