"""
CCXT 加密数据管道 — OHLCV 获取→存储→回测对接。

保留但不默认启用（markets.crypto: false）。

依赖: pip install ccxt

用法
--------
>>> from data.fetchers.ccxt_fetch import fetch_ohlcv
>>> df = fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-12-31")
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config.log import get_logger

logger = get_logger("ccxt_fetch")

# 主流交易所
_EXCHANGE_PRIORITY = ["binance", "okx", "bybit", "gate"]


def _get_exchange(exchange_id: str = ""):
    """获取交易所实例（优先 binance）"""
    try:
        import ccxt
    except ImportError:
        logger.warning("ccxt 未安装，请执行: pip install ccxt")
        return None

    if exchange_id:
        return getattr(ccxt, exchange_id)()
    for ex_id in _EXCHANGE_PRIORITY:
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
            ex.load_markets()
            return ex
        except Exception:
            continue
    return None


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    start_date: str = "20200101",
    end_date: str = "",
    timeframe: str = "1d",
    exchange_id: str = "",
) -> Optional[pd.DataFrame]:
    """
    拉取加密货币 OHLCV 数据。

    参数
    ----------
    symbol : str
        交易对，如 "BTC/USDT"/"ETH/USDT"
    start_date : str
        起始日期 YYYYMMDD
    end_date : str
        结束日期 YYYYMMDD，默认今天
    timeframe : str
        K线周期: "1m"/"5m"/"15m"/"1h"/"4h"/"1d"/"1w"
    exchange_id : str
        指定交易所，空=自动选择

    返回
    -------
    DataFrame: date/open/high/low/close/volume
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    exchange = _get_exchange(exchange_id)
    if exchange is None:
        return None

    try:
        since = exchange.parse8601(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}T00:00:00Z")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not ohlcv:
            logger.debug(f"CCXT {symbol} 无数据")
            return None

        df = pd.DataFrame(ohlcv, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")

        end_dt = pd.to_datetime(end_date)
        df = df[df["date"] <= end_dt]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["symbol"] = symbol
        df["exchange"] = exchange.id

        logger.info(f"CCXT 拉取成功: {symbol}@{exchange.id}, {len(df)} 条")
        return df[["date", "open", "high", "low", "close", "volume", "symbol", "exchange"]]

    except Exception as e:
        logger.warning(f"CCXT {symbol} 拉取失败: {e}")
        return None


def list_symbols(quote: str = "USDT", exchange_id: str = "binance") -> list[str]:
    """
    列出可用交易对。

    参数
    ----------
    quote : str
        计价货币，默认 USDT
    exchange_id : str
        交易所

    返回
    -------
    list[str]: 交易对列表
    """
    exchange = _get_exchange(exchange_id)
    if exchange is None:
        return []
    try:
        exchange.load_markets()
        return [s for s in exchange.symbols if s.endswith(f"/{quote}")]
    except Exception as e:
        logger.warning(f"CCXT 获取交易对失败: {e}")
        return []


def is_crypto_symbol(symbol: str) -> bool:
    """判断是否为加密货币交易对格式"""
    return "/" in symbol and len(symbol.split("/")) == 2
