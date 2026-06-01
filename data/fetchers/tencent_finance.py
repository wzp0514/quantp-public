"""
腾讯财经行情数据 — 免费公开 API，补估值维度。

获取 PE/PB/市值/换手率等基本面数据。

用法
--------
>>> from data.fetchers.tencent_finance import fetch_stock_valuation
>>> df = fetch_stock_valuation("000001")
"""

import logging
from typing import Optional

import pandas as pd
import requests

from config.log import get_logger

logger = get_logger("tencent_finance")

# 腾讯股票行情 API（免费，无需 Key）
_QUOTE_URL = "https://qt.gtimg.cn/q="
_BATCH_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query"


def _clean_tx_data(data: str) -> dict:
    """解析腾讯行情返回的 ~ 分隔字符串"""
    if not data or "~" not in data:
        return {}
    parts = data.split("~")
    if len(parts) < 50:
        return {}
    return {
        "name": parts[1],           # 股票名称
        "code": parts[2],           # 代码
        "price": parts[3],          # 当前价
        "change_pct": parts[32],    # 涨跌幅
        "pe": parts[39],            # 市盈率
        "pe_ttm": parts[39],        # PE(TTM)
        "pb": parts[46],            # 市净率
        "market_cap": parts[45],    # 总市值（亿）
        "circulating_cap": parts[44],  # 流通市值（亿）
        "volume": parts[6],         # 成交量（手）
        "amount": parts[37],        # 成交额（万）
        "turnover": parts[38],      # 换手率(%)
        "high": parts[33],          # 最高
        "low": parts[34],           # 最低
        "open": parts[5],           # 开盘
        "pre_close": parts[4],      # 昨收
        "amplitude": parts[43],     # 振幅(%)
        "volume_ratio": parts[49],  # 量比
    }


def _build_tx_code(symbol: str) -> str:
    """构建腾讯接口代码格式（市场前缀+代码）"""
    code = symbol.strip()
    if code.startswith("6"):
        return f"sh{code}"
    elif code.startswith(("0", "3")):
        return f"sz{code}"
    elif code.startswith("00") and len(code) == 5:
        return f"sz{code}"
    return code


def fetch_stock_valuation(symbol: str) -> Optional[dict]:
    """
    获取单只股票估值快照。

    参数
    ----------
    symbol : str
        股票代码，如 "000001"/"600519"

    返回
    -------
    dict: pe, pb, market_cap, turnover, price, name 等
    """
    tx_code = _build_tx_code(symbol)
    url = f"{_QUOTE_URL}{tx_code}"

    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = "gbk"
        text = resp.text
    except Exception as e:
        logger.warning(f"腾讯行情请求失败 {tx_code}: {e}")
        return None

    result = _clean_tx_data(text)
    if not result or not result.get("price") or result["price"] == "0":
        logger.debug(f"腾讯行情解析失败或停牌: {tx_code}")
        return None

    # 数值转换
    try:
        result["price"] = float(result["price"])
        result["pe"] = float(result["pe"]) if result["pe"] else None
        result["pb"] = float(result["pb"]) if result["pb"] else None
        result["market_cap"] = float(result["market_cap"]) if result["market_cap"] else None
        result["circulating_cap"] = float(result["circulating_cap"]) if result["circulating_cap"] else None
        result["turnover"] = float(result["turnover"]) if result["turnover"] else None
        result["change_pct"] = float(result["change_pct"]) if result["change_pct"] else None
        result["volume_ratio"] = float(result["volume_ratio"]) if result["volume_ratio"] else None
        result["symbol"] = symbol
    except (ValueError, TypeError) as e:
        logger.debug(f"腾讯行情数值转换失败: {e}")
        return None

    logger.debug(f"腾讯行情: {result.get('name')}({symbol}) PE={result.get('pe')} PB={result.get('pb')}")
    return result


def fetch_batch_valuation(symbols: list[str]) -> pd.DataFrame:
    """
    批量获取多只股票估值。

    参数
    ----------
    symbols : list[str]
        股票代码列表

    返回
    -------
    DataFrame: symbol/name/price/pe/pb/market_cap/turnover/change_pct
    """
    results = []
    for sym in symbols:
        try:
            val = fetch_stock_valuation(sym)
            if val:
                results.append(val)
        except Exception:
            continue

    if not results:
        logger.warning("腾讯行情批量获取无结果")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info(f"腾讯行情批量: {len(df)}/{len(symbols)} 只")
    return df


def fetch_index_valuation(index_code: str = "000300") -> Optional[dict]:
    """
    获取指数估值（PE/PB 分位）。

    使用腾讯财经的指数接口获取成分股加权估值。

    参数
    ----------
    index_code : str
        指数代码，如 "000300"（沪深300）

    返回
    -------
    dict: pe, pb, dividend_yield, pe_percentile, pb_percentile
    """
    # 腾讯指数行情
    tx_map = {"000300": "sh000300", "000016": "sh000016", "000905": "sh000905",
              "399006": "sz399006", "000688": "sh000688"}
    tx_code = tx_map.get(index_code, f"sh{index_code}")

    url = f"{_QUOTE_URL}{tx_code}"
    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = "gbk"
        text = resp.text
    except Exception as e:
        logger.warning(f"腾讯指数行情失败: {e}")
        return None

    result = _clean_tx_data(text)
    if not result:
        return None

    try:
        return {
            "index_code": index_code,
            "name": result.get("name", ""),
            "price": float(result.get("price", 0)),
            "change_pct": float(result.get("change_pct", 0)),
            "pe": float(result["pe"]) if result.get("pe") else None,
            "pb": float(result["pb"]) if result.get("pb") else None,
            "volume": result.get("volume", ""),
            "amount": result.get("amount", ""),
        }
    except (ValueError, TypeError):
        return None
