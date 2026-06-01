"""
加密货币佣金模型 — 各交易所 maker/taker 费率。

用于 CCXT 回测对接，默认 0.1%（taker）。

用法
--------
>>> from backtest.engine.crypto_commission import CryptoCommission
>>> cc = CryptoCommission("binance")
>>> cost = cc.calculate(1000, is_maker=False)  # 买入 1000 USDT
>>> print(cost)  # 1.0
"""

# 各交易所默认费率（taker/maker）
_FEE_SCHEDULE = {
    "binance": {"maker": 0.001, "taker": 0.001},
    "okx": {"maker": 0.0008, "taker": 0.001},
    "bybit": {"maker": 0.001, "taker": 0.001},
    "gate": {"maker": 0.002, "taker": 0.002},
    "coinbase": {"maker": 0.004, "taker": 0.006},
    "kraken": {"maker": 0.0016, "taker": 0.0026},
    "default": {"maker": 0.001, "taker": 0.001},
}


class CryptoCommission:
    """加密货币手续费计算器"""

    def __init__(self, exchange: str = "binance", maker_fee: float = 0, taker_fee: float = 0):
        fee = _FEE_SCHEDULE.get(exchange, _FEE_SCHEDULE["default"])
        self.maker_fee = maker_fee if maker_fee > 0 else fee["maker"]
        self.taker_fee = taker_fee if taker_fee > 0 else fee["taker"]
        self.exchange = exchange

    def calculate(self, trade_value: float, is_maker: bool = False) -> float:
        """
        计算单笔交易手续费。

        参数
        ----------
        trade_value : float
            成交金额（base_currency * price）
        is_maker : bool
            True=挂单(maker), False=吃单(taker)
        """
        rate = self.maker_fee if is_maker else self.taker_fee
        return trade_value * rate

    def round_trip(self, buy_value: float, sell_value: float) -> float:
        """往返手续费（买入taker + 卖出taker）"""
        return self.calculate(buy_value) + self.calculate(sell_value)

    def info(self) -> dict:
        return {"exchange": self.exchange, "maker": f"{self.maker_fee*100:.2f}%",
                "taker": f"{self.taker_fee*100:.2f}%"}
