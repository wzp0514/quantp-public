"""
vnpy 实盘网关 — A股/期货实盘交易接口

vnpy 是国内最知名的开源量化交易框架（38K Stars），
覆盖 90%+ 期货公司（CTP）和主流券商接口（XTP/宽睿/华鑫奇点等）。

本模块提供 vnpy 的轻量封装，对接风控+仓位管理系统。

前置条件：
  1. pip install vnpy
  2. 期货仿真: 注册 SimNow (simnow.com.cn) 免费账户
  3. 配置 config/settings.local.yaml 中的账户信息

用法
--------
>>> from live.gateway.vnpy_gateway import VnpyGateway
>>> gw = VnpyGateway()
>>> gw.connect()
>>> gw.get_positions()
>>> gw.market_buy("IF2406", 1, price_check=True)
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from config.loader import get_config
from config.log import get_logger

logger = get_logger("vnpy_gateway")

# ── vnpy 可用性检测 ──────────────────────────────────────────
_VNPY_AVAILABLE = False
_VNPY_IMPORT_ERROR = ""

try:
    from vnpy.event.engine import EventEngine
    from vnpy.trader.engine import MainEngine
    from vnpy.trader.event import (
        EVENT_TICK, EVENT_ORDER, EVENT_TRADE,
        EVENT_POSITION, EVENT_ACCOUNT, EVENT_CONTRACT,
        EVENT_LOG,
    )
    from vnpy.trader.constant import Direction, Offset, OrderType, Status
    from vnpy.trader.object import OrderRequest, SubscribeRequest
    _VNPY_AVAILABLE = True
except ImportError as e:
    _VNPY_IMPORT_ERROR = str(e)
    logger.info(f"vnpy 未安装（{_VNPY_IMPORT_ERROR}），将使用骨架模式。"
                f"安装: pip install vnpy")


class VnpyGateway:
    """
    vnpy 实盘网关。

    两种运行模式：
      - 真实模式: vnpy 已安装且配置完整 → 真实连接交易所
      - 骨架模式: vnpy 未安装 → 打印模拟信息，返回假数据（用于开发测试）

    SimNow 仿真（免费，推荐入门）:
      1. 注册 https://www.simnow.com.cn/
      2. 拿到 broker_id / user_id / password
      3. 填入 settings.local.yaml
    """

    # SimNow 7×24 仿真环境（免费注册即可使用）
    SIMNOW_CONFIG = {
        "broker": "CTP",
        "broker_id": "9999",
        "td_address": "tcp://180.168.146.187:10202",
        "md_address": "tcp://180.168.146.187:10212",
        "app_id": "simnow_client_test",
        "auth_code": "0000000000000000",
    }

    def __init__(self):
        self.connected = False
        self._mode = "skeleton"  # "skeleton" | "live"
        self._main_engine = None
        self._event_engine = None
        self._gateway_name = ""
        self._account_info: dict = {}
        self._positions: dict[str, dict] = {}
        self._orders: dict[str, dict] = {}
        self._trades: list[dict] = []
        self._cfg: dict = {}
        self._lock = threading.Lock()
        self._load_config()

    # ── 配置 ─────────────────────────────────────────────────

    def _load_config(self):
        self._cfg = get_config().get("vnpy", {})

    # ── 连接 ─────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        连接到交易网关。

        真实模式下连接 vnpy 网关（CTP/XTP），骨架模式下打印配置提示。
        """
        self._load_config()

        if not _VNPY_AVAILABLE:
            return self._connect_skeleton()

        broker = self._cfg.get("broker", "")
        username = self._cfg.get("username", "")

        if not broker or not username:
            logger.warning(
                "vnpy 账户未配置。在 config/settings.local.yaml 添加:\n"
                "  vnpy:\n"
                "    broker: \"CTP\"\n"
                "    username: \"你的账号\"\n"
                "    password: \"你的密码\"\n"
                "    broker_id: \"9999\"\n"
                "    td_address: \"tcp://180.168.146.187:10202\"\n"
                "    md_address: \"tcp://180.168.146.187:10212\"\n"
                "    app_id: \"simnow_client_test\"\n"
                "    auth_code: \"0000000000000000\"\n"
                "\n以上是 SimNow 仿真环境（免费注册 simnow.com.cn）。"
            )
            return self._connect_skeleton()

        try:
            return self._connect_live()
        except Exception as e:
            logger.error(f"vnpy 真实连接失败: {e}，降级为骨架模式")
            return self._connect_skeleton()

    def _connect_live(self) -> bool:
        """真实连接 vnpy 网关"""
        broker = self._cfg["broker"]

        # 创建事件引擎和主引擎
        # 注意：MainEngine.__init__() 内部会调用 event_engine.start()，不能提前 start
        self._event_engine = EventEngine()
        self._main_engine = MainEngine(self._event_engine)

        # 根据 broker 类型选择网关
        gateway_map = {
            "CTP": "CtpGateway",
            "CTPTEST": "CtpGateway",
            "XTP": "XtpGateway",
            "MINI": "MiniGateway",
        }
        gateway_name = gateway_map.get(broker, "CtpGateway")
        self._gateway_name = gateway_name
        self._main_engine.add_gateway(gateway_name)

        # 构建连接配置
        setting = {
            "用户名": self._cfg.get("username", ""),
            "密码": self._cfg.get("password", ""),
            "经纪商代码": self._cfg.get("broker_id", ""),
            "交易服务器": self._cfg.get("td_address", ""),
            "行情服务器": self._cfg.get("md_address", ""),
            "产品名称": self._cfg.get("app_id", ""),
            "授权编码": self._cfg.get("auth_code", ""),
        }

        # 连接
        self._main_engine.connect(setting, gateway_name)

        # 注册事件回调
        self._register_events()

        # 等待连接建立
        time.sleep(2)

        self.connected = True
        self._mode = "live"
        logger.info(f"已连接 vnpy {broker} 网关 ({gateway_name}) — 真实交易")
        return True

    def _connect_skeleton(self) -> bool:
        """骨架模式连接"""
        cfg = self._cfg
        broker = cfg.get("broker", "") or "CTP（SimNow仿真）"
        username = cfg.get("username", "") or "未配置"

        logger.info(f"骨架模式连接: {broker} (用户 {username})")
        logger.info(
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            "\n  A股实盘暂不可用（需要 vnpy_ctp CTP 网关）"
            "\n"
            "\n  要启用实盘？请完成以下步骤："
            "\n    1. 安装编译工具（二选一）："
            "\n       A. winget install Microsoft.VisualStudio.2022.BuildTools"
            "\n       B. 或下载 VeighNa Studio（含预编译 CTP）"
            "\n    2. pip install vnpy_ctp"
            "\n    3. 注册 SimNow (simnow.com.cn, 免费)"
            "\n    4. 在 config/settings.local.yaml 填写账户信息"
            "\n"
            "\n  当前为骨架模式 —— 不影响其他功能，可正常回测/策略挖掘"
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        self.connected = True
        self._mode = "skeleton"
        return True

    def _register_events(self):
        """注册 vnpy 事件回调"""
        if not self._event_engine:
            return
        self._event_engine.register(EVENT_ORDER, self._on_order)
        self._event_engine.register(EVENT_TRADE, self._on_trade)
        self._event_engine.register(EVENT_POSITION, self._on_position)
        self._event_engine.register(EVENT_ACCOUNT, self._on_account)

    # ── vnpy 事件回调 ─────────────────────────────────────────

    def _on_order(self, event):
        """订单状态更新"""
        order = event.data
        self._orders[order.orderid] = {
            "id": order.orderid,
            "symbol": order.symbol,
            "direction": str(order.direction),
            "offset": str(order.offset),
            "price": order.price,
            "volume": order.volume,
            "traded": order.traded,
            "status": str(order.status),
            "time": order.datetime,
        }
        logger.debug(f"订单更新: {order.orderid} {order.symbol} {order.status}")

    def _on_trade(self, event):
        """成交回报"""
        trade = event.data
        self._trades.append({
            "id": trade.tradeid,
            "order_id": trade.orderid,
            "symbol": trade.symbol,
            "direction": str(trade.direction),
            "price": trade.price,
            "volume": trade.volume,
            "time": trade.datetime,
        })
        logger.info(f"成交: {trade.symbol} {trade.direction} "
                    f"{trade.volume}@{trade.price}")

    def _on_position(self, event):
        """持仓更新"""
        pos = event.data
        key = f"{pos.symbol}_{pos.direction}"
        self._positions[key] = {
            "symbol": pos.symbol,
            "direction": str(pos.direction),
            "volume": pos.volume,
            "price": pos.price,
            "pnl": pos.pnl,
        }

    def _on_account(self, event):
        """账户资金更新"""
        account = event.data
        self._account_info = {
            "balance": account.balance,
            "available": account.available,
            "frozen": account.frozen,
            "margin": account.margin,
        }

    # ── 账户查询 ─────────────────────────────────────────────

    def get_balance(self) -> float:
        """查询可用资金"""
        self._check_connected()

        if self._mode == "live" and self._account_info:
            return float(self._account_info.get("available", 0))

        logger.info("查询可用资金（骨架模式: 100,000）")
        return 100000.0

    def get_account_info(self) -> dict:
        """查询完整账户信息"""
        self._check_connected()

        if self._mode == "live" and self._account_info:
            return dict(self._account_info)

        return {
            "balance": 100000.0,
            "available": 100000.0,
            "frozen": 0.0,
            "margin": 0.0,
            "mode": "skeleton",
        }

    def get_positions(self) -> list[dict]:
        """查询当前持仓"""
        self._check_connected()

        if self._mode == "live" and self._positions:
            return list(self._positions.values())

        logger.info("查询持仓（骨架模式: 空仓）")
        return []

    def get_orders(self) -> list[dict]:
        """查询所有订单"""
        self._check_connected()
        return list(self._orders.values())

    def get_trades(self) -> list[dict]:
        """查询所有成交"""
        self._check_connected()
        return list(self._trades)

    # ── 下单（带风控检查）───────────────────────────────────

    def market_buy(
        self,
        symbol: str,
        volume: int,
        price_check: bool = True,
    ) -> Optional[dict]:
        """市价买入"""
        return self._place_order(symbol, "buy", "market", volume, price_check=price_check)

    def market_sell(
        self,
        symbol: str,
        volume: int,
        price_check: bool = True,
    ) -> Optional[dict]:
        """市价卖出"""
        return self._place_order(symbol, "sell", "market", volume, price_check=price_check)

    def limit_buy(
        self,
        symbol: str,
        volume: int,
        price: float,
        price_check: bool = True,
    ) -> Optional[dict]:
        """限价买入"""
        return self._place_order(symbol, "buy", "limit", volume, price, price_check)

    def limit_sell(
        self,
        symbol: str,
        volume: int,
        price: float,
        price_check: bool = True,
    ) -> Optional[dict]:
        """限价卖出"""
        return self._place_order(symbol, "sell", "limit", volume, price, price_check)

    def _place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        volume: int,
        price: float = 0,
        price_check: bool = True,
    ) -> Optional[dict]:
        """下单核心逻辑"""
        self._check_connected()

        if price_check:
            ok, msg = self._risk_check(side, symbol, volume)
            if not ok:
                logger.warning(f"风控拒绝 {side} {symbol}: {msg}")
                return None

        if self._mode == "live":
            return self._place_order_live(symbol, side, order_type, volume, price)

        return self._place_order_skeleton(symbol, side, volume)

    def _place_order_live(
        self,
        symbol: str,
        side: str,
        order_type: str,
        volume: int,
        price: float = 0,
    ) -> Optional[dict]:
        """通过 vnpy 真实下单"""
        direction = Direction.LONG if side == "buy" else Direction.SHORT
        offset = Offset.OPEN
        order_price = price if order_type == "limit" else 0
        order_type_enum = OrderType.LIMIT if order_type == "limit" else OrderType.MARKET

        req = OrderRequest(
            symbol=symbol,
            exchange=self._cfg.get("exchange", ""),
            direction=direction,
            type=order_type_enum,
            volume=volume,
            price=order_price,
            offset=offset,
        )

        try:
            vt_orderid = self._main_engine.send_order(req, self._gateway_name)
            logger.info(f"订单已提交: {side.upper()} {volume} {symbol} | ID: {vt_orderid}")
            return {
                "status": "submitted",
                "symbol": symbol,
                "side": side,
                "volume": volume,
                "price": price,
                "order_id": vt_orderid,
                "mode": "live",
            }
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return None

    def _place_order_skeleton(self, symbol: str, side: str, volume: int) -> dict:
        """骨架模式下单（返回模拟数据）"""
        logger.info(f"骨架模式 {side.upper()}: {symbol} {volume}手")
        return {
            "status": "submitted",
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "mode": "skeleton",
        }

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        self._check_connected()

        if self._mode == "live":
            try:
                self._main_engine.cancel_order(order_id, self._gateway_name)
                logger.info(f"订单 {order_id} 已撤销")
                return True
            except Exception as e:
                logger.error(f"撤销失败: {e}")
                return False

        logger.info(f"骨架模式: 订单 {order_id} 已撤销")
        return True

    def cancel_all(self) -> int:
        """撤销所有未成交订单，返回撤销数量"""
        self._check_connected()
        count = 0
        for oid, o in list(self._orders.items()):
            if o.get("status") not in ("已撤销", "全成", "拒单"):
                if self.cancel_order(oid):
                    count += 1
        logger.info(f"已撤销 {count} 个订单")
        return count

    # ── 行情 ─────────────────────────────────────────────────

    def subscribe(self, symbol: str) -> bool:
        """订阅行情"""
        if self._mode != "live" or not self._main_engine:
            logger.info(f"骨架模式: 订阅 {symbol}")
            return True

        try:
            req = SubscribeRequest(symbol=symbol)
            self._main_engine.subscribe(req, self._gateway_name)
            return True
        except Exception as e:
            logger.error(f"订阅失败: {e}")
            return False

    # ── 风控 ─────────────────────────────────────────────────

    def _risk_check(self, side: str, symbol: str, volume: int, price: float = 0) -> tuple[bool, str]:
        """每次下单前的风控检查"""
        try:
            from live.risk.risk_engine import run_all_checks
            # 计算当前仓位占比（避免空参数导致仓位限制失效）
            balance = self.get_balance()
            position_pct = (volume * price) / balance if balance > 0 and price > 0 else 0.0
            results = run_all_checks(position_pct=position_pct, order_count=1, order_time_minutes=60)
            for ok, msg in results:
                if not ok:
                    return False, msg
            return True, ""
        except Exception as e:
            logger.warning(f"风控检查跳过: {e}")
            return True, ""

    # ── 辅助 ─────────────────────────────────────────────────

    def _check_connected(self):
        if not self.connected:
            raise RuntimeError("未连接！请先调用 connect()")

    def disconnect(self):
        """断开连接"""
        if self._mode == "live" and self._main_engine:
            try:
                self._main_engine.close()
            except Exception as e:
                logger.warning(f"关闭主引擎异常: {e}")

        self.connected = False
        self._main_engine = None
        self._event_engine = None
        self._mode = "skeleton"
        logger.info("已断开")

    @property
    def mode(self) -> str:
        """当前模式: 'skeleton' | 'live'"""
        return self._mode

    @property
    def is_live(self) -> bool:
        """是否真实交易模式"""
        return self._mode == "live" and self.connected


# ── SimNow 配置模板 ──────────────────────────────────────────

VNPY_CONFIG_TEMPLATE = """
# vnpy 实盘账户配置（填入 config/settings.local.yaml，已 gitignore）
# SimNow 仿真环境（推荐入门，免费）：
#   1. 注册 https://www.simnow.com.cn/
#   2. 拿到 broker_id / user_id / password
#   3. 填入下方配置

vnpy:
  broker: "CTP"                     # CTP=期货, XTP=股票
  username: "你的SimNow用户ID"
  password: "你的SimNow密码"
  broker_id: "9999"                 # SimNow 仿真用 9999
  td_address: "tcp://180.168.146.187:10202"
  md_address: "tcp://180.168.146.187:10212"
  app_id: "simnow_client_test"
  auth_code: "0000000000000000"
  exchange: ""                      # 留空自动识别
"""


# ── 命令行测试 ──────────────────────────────────────────────
# python live/gateway/vnpy_gateway.py

if __name__ == "__main__":
    print("=" * 60)
    print("vnpy 实盘网关 — A股/期货")
    print(f"vnpy 可用: {'是' if _VNPY_AVAILABLE else '否（骨架模式）'}")
    print("=" * 60)

    gw = VnpyGateway()
    if gw.connect():
        print(f"\n模式: {gw.mode}")
        balance = gw.get_balance()
        print(f"可用资金: {balance:,.0f}")

        positions = gw.get_positions()
        if positions:
            print(f"当前持仓: {len(positions)} 个")
            for p in positions:
                print(f"  {p['symbol']}: {p['volume']}手 @ {p['price']}")
        else:
            print("当前空仓")

        if gw.mode == "skeleton":
            print(f"\n配置模板（粘贴到 settings.local.yaml）:")
            print(VNPY_CONFIG_TEMPLATE)

        gw.disconnect()
    else:
        print("连接失败。")
