"""
通达信（mootdx）免费行情源 — 零成本零Key。

封装标准行情/板块/财务接口，纳入 fallback 链作为第一级（akshare 之上）。

依赖: pip install mootdx
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config.log import get_logger

logger = get_logger("mootdx_fetch")

# 指数名 → 通达信代码
_INDEX_MAP = {
    "沪深300": "000300",
    "中证500": "000905",
    "中证1000": "000852",
    "上证50": "000016",
    "创业板指": "399006",
    "科创50": "000688",
    "深证成指": "399001",
    "上证指数": "000001",
}

# 通达信代码 → 指数名
_CODE_TO_NAME = {v: k for k, v in _INDEX_MAP.items()}


def _get_client():
    """延迟导入 + 客户端创建"""
    from mootdx.quotes import Quotes
    return Quotes.factory(market="std", timeout=10)


def fetch_index_daily(
    name: str = "沪深300",
    start_date: str = "20200101",
    end_date: str = "",
) -> Optional[pd.DataFrame]:
    """
    拉取指数日线 OHLCV。

    参数
    ----------
    name : str
        指数名称，如"沪深300"/"中证500"/"创业板指"
    start_date : str
        起始日期 YYYYMMDD
    end_date : str
        结束日期 YYYYMMDD，默认今天

    返回
    -------
    DataFrame，列: date/open/high/low/close/volume/amount
    """
    code = _INDEX_MAP.get(name)
    if not code:
        logger.debug(f"mootdx 不支持的指数: {name}")
        return None

    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    try:
        client = _get_client()
        # 日线: frequency=9
        df = client.bars(symbol=code, frequency=9, start=0, count=2000)
        if df is None or df.empty:
            logger.debug(f"mootdx {name}({code}) 无数据")
            return None

        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "vol": "volume", "amount": "amount",
        })

        # 确保有 date 列
        if "date" not in df.columns:
            # mootdx 可能返回 datetime 作为 index
            if isinstance(df.index, pd.DatetimeIndex):
                df["date"] = df.index.strftime("%Y-%m-%d")
            else:
                logger.debug(f"mootdx {name}({code}) 无 date 列")
                return None

        # 转换为 datetime 便于过滤
        df["date"] = pd.to_datetime(df["date"])

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        if df.empty:
            logger.debug(f"mootdx {name}({code}) 日期范围无数据")
            return None

        # 统一列名和顺序
        keep_cols = ["date", "open", "high", "low", "close", "volume"]
        # 确保所有 OHLCV 列存在
        for c in keep_cols[1:]:
            if c not in df.columns:
                logger.debug(f"mootdx {name}({code}) 缺少列: {c}")
                return None

        df = df[keep_cols].reset_index(drop=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["name"] = name
        logger.info(f"mootdx 拉取成功: {name}({code}), {len(df)} 条")
        return df

    except ImportError:
        logger.debug("mootdx 未安装", exc_info=False)
        return None
    except Exception as e:
        logger.debug(f"mootdx {name}({code}) 拉取失败: {e}")
        return None


def fetch_stock_daily(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "",
) -> Optional[pd.DataFrame]:
    """
    拉取个股日线 OHLCV。

    参数
    ----------
    symbol : str
        股票代码（6位数字），如 "000001"
    start_date : str
        起始日期 YYYYMMDD
    end_date : str
        结束日期 YYYYMMDD

    返回
    -------
    DataFrame，列: date/open/high/low/close/volume
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    try:
        client = _get_client()
        df = client.bars(symbol=symbol, frequency=9, start=0, count=2000)
        if df is None or df.empty:
            return None

        # 处理日期列
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df["date"] = df.index.strftime("%Y-%m-%d")

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        if df.empty:
            return None

        keep_cols = ["date", "open", "high", "low", "close", "volume"]
        for c in keep_cols[1:]:
            if c not in df.columns:
                return None

        df = df[keep_cols].reset_index(drop=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        logger.info(f"mootdx 个股拉取成功: {symbol}, {len(df)} 条")
        return df

    except ImportError:
        logger.debug("mootdx 未安装", exc_info=False)
        return None
    except Exception as e:
        logger.debug(f"mootdx 个股 {symbol} 拉取失败: {e}")
        return None


def fetch_block() -> Optional[pd.DataFrame]:
    """
    获取板块分类数据。

    返回
    -------
    DataFrame，列: block_name/block_type/symbol/symbol_name
    """
    try:
        client = _get_client()
        df = client.block()
        if df is None or df.empty:
            return None
        logger.info(f"mootdx 板块数据: {len(df)} 条")
        return df
    except ImportError:
        logger.debug("mootdx 未安装", exc_info=False)
        return None
    except Exception as e:
        logger.debug(f"mootdx 板块拉取失败: {e}")
        return None


def fetch_finance(symbol: str) -> Optional[pd.DataFrame]:
    """
    获取个股财务数据。

    返回
    -------
    DataFrame
    """
    try:
        client = _get_client()
        df = client.finance(symbol=symbol)
        if df is None or df.empty:
            return None
        logger.info(f"mootdx 财务数据: {symbol}, {len(df)} 条")
        return df
    except ImportError:
        logger.debug("mootdx 未安装", exc_info=False)
        return None
    except Exception as e:
        logger.debug(f"mootdx 财务 {symbol} 拉取失败: {e}")
        return None


def fetch_quotes(symbols: list[str]) -> Optional[pd.DataFrame]:
    """
    获取实时行情（五档）。

    参数
    ----------
    symbols : list[str]
        股票代码列表，如 ["000001", "000002"]

    返回
    -------
    DataFrame
    """
    try:
        client = _get_client()
        df = client.quotes(symbol=symbols)
        if df is None or df.empty:
            return None
        logger.info(f"mootdx 实时行情: {len(symbols)} 只, {len(df)} 条")
        return df
    except ImportError:
        logger.debug("mootdx 未安装", exc_info=False)
        return None
    except Exception as e:
        logger.debug(f"mootdx 行情 {symbols} 拉取失败: {e}")
        return None
