"""
订单管理系统

管理从"策略发出信号"到"交易所确认成交"的整个过程。

订单生命周期：
  CREATED → SUBMITTED → FILLED（成交）
                       → REJECTED（被交易所拒绝）
                       → CANCELLED（用户取消）
                       → EXPIRED（超时未成交）

真实交易 vs 纸上交易：
  - 纸上交易：所有订单立即按次日开盘价模拟成交
  - 真实交易：订单交给经纪商 API，实际状态由 API 回调更新

订单以字典而不是类存储，简单直接。
"""

import logging
from datetime import date, datetime
from typing import Optional
# 订单状态常量
STATUS_CREATED = "created"       # 刚创建
STATUS_SUBMITTED = "submitted"   # 已提交给交易所
STATUS_FILLED = "filled"         # 已成交
STATUS_PARTIAL = "partial"       # 部分成交
STATUS_CANCELLED = "cancelled"   # 已取消
STATUS_REJECTED = "rejected"     # 被拒绝
STATUS_EXPIRED = "expired"       # 已过期

# 订单方向
BUY = "buy"
SELL = "sell"

from config.log import get_logger
logger = get_logger("order_mgr")
class OrderManager:
    """
    订单管理器：创建、跟踪、管理所有订单。

    示例
    --------
    >>> om = OrderManager()
    >>> order = om.create_market_order("sh000300", "buy", 100)
    >>> om.fill_order(order["id"], fill_price=4000.0)
    >>> orders = om.get_orders(status="filled")
    """

    def __init__(self):
        self.orders: list[dict] = []
        self._next_id = 1

    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: int,
        order_date: Optional[date] = None,
    ) -> dict:
        """
        创建市价单（按当前市价立刻成交）

        参数
        ----------
        symbol : str
            交易标的，如 "sh000300" 或 "600519"
        side : str
            "buy" 或 "sell"
        size : int
            数量（股/张）
        order_date : date
            下单日期，默认今天

        返回
        -------
        dict : 订单记录
        """
        if side not in (BUY, SELL):
            raise ValueError(f"side 必须是 'buy' 或 'sell'，收到了: {side}")
        if size <= 0:
            raise ValueError(f"size 必须 > 0，收到了: {size}")

        order = {
            "id": self._next_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "filled_size": 0,
            "price": None,         # 成交价（成交后才填写）
            "status": STATUS_CREATED,
            "created_at": order_date or date.today(),
            "filled_at": None,
            "note": "",
        }
        self._next_id += 1
        self.orders.append(order)

        logger.info(
            f"订单 #{order['id']} 创建: {side.upper()} {size} {symbol}"
        )
        return order

    def submit(self, order_id: int) -> dict:
        """提交订单到交易所（纸上交易中直接成交，真实交易中异步等待）"""
        order = self._find(order_id)
        order["status"] = STATUS_SUBMITTED
        logger.debug(f"订单 #{order_id} 已提交")
        return order

    def fill(self, order_id: int, fill_price: float, fill_date: Optional[date] = None) -> dict:
        """
        订单完全成交

        参数
        ----------
        order_id : int
        fill_price : float
            成交价格
        fill_date : date
            成交日期
        """
        order = self._find(order_id)
        order["status"] = STATUS_FILLED
        order["price"] = fill_price
        order["filled_size"] = order["size"]
        order["filled_at"] = fill_date or date.today()

        value = fill_price * order["size"]
        logger.info(
            f"订单 #{order_id} 成交: {order['side'].upper()} "
            f"{order['size']} {order['symbol']} @ {fill_price:.2f} "
            f"(金额: {value:,.2f})"
        )
        return order

    def partial_fill(self, order_id: int, fill_size: int, fill_price: float) -> dict:
        """部分成交"""
        order = self._find(order_id)
        order["filled_size"] += fill_size
        order["price"] = fill_price
        if order["filled_size"] >= order["size"]:
            order["status"] = STATUS_FILLED
        else:
            order["status"] = STATUS_PARTIAL
        logger.info(
            f"订单 #{order_id} 部分成交: {fill_size}/{order['size']} @ {fill_price:.2f}"
        )
        return order

    def cancel(self, order_id: int) -> dict:
        """取消订单"""
        order = self._find(order_id)
        if order["status"] in (STATUS_FILLED, STATUS_CANCELLED):
            logger.warning(f"订单 #{order_id} 已是 {order['status']}，无法取消")
            return order
        order["status"] = STATUS_CANCELLED
        logger.info(f"订单 #{order_id} 已取消")
        return order

    def reject(self, order_id: int, reason: str = "") -> dict:
        """订单被拒绝（资金不足/风控拦截等）"""
        order = self._find(order_id)
        order["status"] = STATUS_REJECTED
        order["note"] = reason
        logger.warning(f"订单 #{order_id} 被拒绝: {reason}")
        return order

    def get_orders(self, status: str = "", symbol: str = "") -> list[dict]:
        """查询订单"""
        orders = self.orders
        if status:
            orders = [o for o in orders if o["status"] == status]
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
        return orders

    def get_order(self, order_id: int) -> Optional[dict]:
        """查询单个订单"""
        try:
            return self._find(order_id)
        except ValueError:
            return None

    def get_pending(self) -> list[dict]:
        """获取所有待处理订单（已提交但未成交）"""
        return [o for o in self.orders if o["status"] in (STATUS_CREATED, STATUS_SUBMITTED, STATUS_PARTIAL)]

    def _find(self, order_id: int) -> dict:
        for o in self.orders:
            if o["id"] == order_id:
                return o
        raise ValueError(f"订单 #{order_id} 不存在")
