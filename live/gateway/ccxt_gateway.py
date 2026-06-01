"""
CCXT 交易所网关 — 加密货币实盘接口

CCXT 是一个统一了 100+ 加密货币交易所 API 的库。
一行代码切换 Binance/OKX/Bybit，不用学每个交易所的 API。

当前实现：Binance Testnet（测试网，完全免费，API 和真实一模一样）
真实交易前必须先在 Testnet 上验证所有流程。

Testnet 注册: https://testnet.binance.vision/
  - 注册后拿到 API Key 和 Secret
  - 放在 config/settings.local.yaml 中（已 gitignore）
  - Testnet 里的 BTC/USDT 价格和真实市场一样，但用的是假钱

依赖: pip install ccxt（已在 requirements.txt 中）

参考资料:
  CCXT 文档: https://docs.ccxt.com/
  Binance Testnet: https://testnet.binance.vision/
"""

import logging
from datetime import datetime
from typing import Optional

import ccxt

from config.log import get_logger
logger = get_logger("ccxt_gateway")
class CCXTGateway:
    """
    CCXT 交易所网关。

    支持的操作：
      - 连接交易所（真实 or 测试网）
      - 查询余额和持仓
      - 下市价单/限价单
      - 撤销订单
      - 拉取 K 线数据

    用法
    --------
    >>> gw = CCXTGateway(testnet=True)
    >>> gw.connect(api_key="...", secret="...")
    >>> balance = gw.get_balance()
    >>> order = gw.market_buy("BTC/USDT", 0.001)  # 买 0.001 BTC
    """

    def __init__(self, exchange: str = "binance", testnet: bool = True):
        """
        参数
        ----------
        exchange : str
            交易所名称，如 "binance", "okx", "bybit"
        testnet : bool
            True = 连接测试网（免费假钱），False = 真实交易（真金白银）
        """
        self.exchange_name = exchange
        self.testnet = testnet
        self.connected = False
        self._api_errors = 0  # 连续API错误计数器（熔断用）
        self._max_api_errors = 5  # 连续5次错误触发熔断

        # CCXT 交易所实例（懒加载，connect 时才创建）
        self._exchange = None

        if testnet:
            logger.info(f"CCXT 网关初始化: {exchange} 测试网（免费假钱）")
        else:
            logger.warning(f"CCXT 网关初始化: {exchange} 真实交易！！！")

    def connect(self, api_key: str = "", secret: str = "", password: str = "") -> bool:
        """
        连接到交易所。

        api_key 和 secret 从 config/settings.local.yaml 读取，不写在代码里。
        """
        exchange_class = getattr(ccxt, self.exchange_name)
        if exchange_class is None:
            logger.error(f"不支持的交易所: {self.exchange_name}")
            return False

        config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,  # CCXT 自动控制请求频率（避免被封 IP）
            "options": {"defaultType": "spot"},  # 现货交易（不做合约）
        }

        if self.testnet and self.exchange_name == "binance":
            config["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api/v3",
                    "private": "https://testnet.binance.vision/api/v3",
                }
            }

        self._exchange = exchange_class(config)

        try:
            self._exchange.load_markets()
            self.connected = True
            logger.info(f"已连接到 {self.exchange_name}" +
                       (" 测试网" if self.testnet else " 真实交易"))
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    # ============================================================
    # 账户查询
    # ============================================================

    def _retry_call(self, fn, max_retries: int = 3, backoff: float = 1.0):
        """带指数退避的API重试 + 连续错误熔断。参照05文档失败5(API死亡500万)。"""
        import time
        last_err = None
        for attempt in range(max_retries):
            try:
                result = fn()
                self._api_errors = 0  # 成功→重置计数器
                return result
            except Exception as e:
                last_err = e
                self._api_errors += 1
                wait = backoff * (2 ** attempt)
                logger.warning(f"API调用失败 (attempt {attempt+1}/{max_retries}): {e}，{wait}s后重试")
                if self._api_errors >= self._max_api_errors:
                    logger.error(f"连续{self._max_api_errors}次API错误，触发熔断！参照05文档失败5。")
                    self.connected = False
                    raise RuntimeError(f"API连续{self._max_api_errors}次错误，已熔断") from e
                time.sleep(wait)
        raise last_err

    def get_balance(self, currency: str = "USDT") -> float:
        """查询可用余额（默认查 USDT）"""
        if not self._check_connected():
            return 0.0
        try:
            balance = self._exchange.fetch_balance()
            free = balance.get(currency, {}).get("free", 0)
            logger.info(f"余额: {free} {currency}")
            return free
        except Exception as e:
            logger.error(f"查询余额失败: {e}")
            return 0.0

    def get_positions(self) -> list[dict]:
        """查询当前所有持仓（现货）"""
        if not self._check_connected():
            return []
        try:
            balance = self._exchange.fetch_balance()
            positions = []
            for currency, info in balance.get("total", {}).items():
                if info > 0 and currency != "USDT":
                    positions.append({"currency": currency, "amount": info})
            logger.info(f"持仓: {positions}")
            return positions
        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return []

    # ============================================================
    # 下单
    # ============================================================

    def market_buy(self, symbol: str, amount: float) -> Optional[dict]:
        """
        市价买入

        参数
        ----------
        symbol : str
            交易对，如 "BTC/USDT"
        amount : float
            买入数量（以 base currency 计，如 0.001 BTC）

        返回
        -------
        dict 或 None : 订单信息
        """
        return self._place_order(symbol, "buy", "market", amount)

    def market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """市价卖出"""
        return self._place_order(symbol, "sell", "market", amount)

    def limit_buy(self, symbol: str, amount: float, price: float) -> Optional[dict]:
        """限价买入（指定价格，可能不成交）"""
        return self._place_order(symbol, "buy", "limit", amount, price)

    def limit_sell(self, symbol: str, amount: float, price: float) -> Optional[dict]:
        """限价卖出"""
        return self._place_order(symbol, "sell", "limit", amount, price)

    def _place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float = None,
    ) -> Optional[dict]:
        """下单"""
        if not self._check_connected():
            return None

        def _do():
            params = {"symbol": symbol, "type": order_type, "side": side, "amount": amount}
            if price and order_type == "limit":
                params["price"] = price
            order = self._exchange.create_order(**params)
            logger.info(f"订单已提交: {side.upper()} {amount} {symbol} @ {order.get('price', '市价')} | ID: {order['id']}")
            return order

        try:
            return self._retry_call(_do)
        except Exception as e:
            logger.error(f"下单失败(已重试{3}次): {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """撤销订单"""
        if not self._check_connected():
            return False
        try:
            self._exchange.cancel_order(order_id, symbol)
            logger.info(f"订单 {order_id} 已撤销")
            return True
        except Exception as e:
            logger.error(f"撤销失败: {e}")
            return False

    def get_order(self, order_id: str, symbol: str) -> Optional[dict]:
        """查询订单状态"""
        if not self._check_connected():
            return None
        try:
            return self._exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return None

    # ============================================================
    # 行情数据
    # ============================================================

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """获取最新行情（买一/卖一/最新价/24h成交量）"""
        if not self._check_connected():
            return None
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            logger.debug(f"{symbol}: bid={ticker['bid']}, ask={ticker['ask']}, last={ticker['last']}")
            return ticker
        except Exception as e:
            logger.error(f"获取行情失败: {e}")
            return None

    def get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> list:
        """获取 K 线数据"""
        if not self._check_connected():
            return []
        try:
            return self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"获取 K 线失败: {e}")
            return []

    # ============================================================
    # 辅助
    # ============================================================

    def _check_connected(self) -> bool:
        if not self.connected:
            logger.error("未连接到交易所，请先调用 connect()")
            return False
        return True

    def disconnect(self):
        """断开连接"""
        self.connected = False
        self._exchange = None
        logger.info("已断开交易所连接")


# ============================================================
# 命令行测试
# ============================================================
# python live/gateway/ccxt_gateway.py

if __name__ == "__main__":
    print("=" * 60)
    print("CCXT 网关测试（Binance Testnet，免费假钱）")
    print("=" * 60)

    gw = CCXTGateway(exchange="binance", testnet=True)

    # Testnet 不需要真实 API key 也能查行情
    if gw.connect():
        ticker = gw.get_ticker("BTC/USDT")
        if ticker:
            print(f"\nBTC/USDT 最新价: ${ticker['last']:,.2f}")
            print(f"  买一: ${ticker['bid']:,.2f}")
            print(f"  卖一: ${ticker['ask']:,.2f}")
            print(f"  24h 涨跌: {ticker['percentage']:.2f}%")

        print("\n需要 API key 才能下单。注册地址: https://testnet.binance.vision/")
        gw.disconnect()
    else:
        print("连接失败（需要网络访问 Binance）")
