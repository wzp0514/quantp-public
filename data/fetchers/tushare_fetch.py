"""
Tushare Pro 数据获取 — A股分钟级数据（¥200/年）

Tushare Pro 是目前性价比最高的 A 股分钟级数据源。
与 AkShare（免费日频）互补：日线回测用 AkShare，分钟回测用 Tushare。

注册和获取 Token:
  1. 注册 https://tushare.pro/
  2. 个人中心 → 接口 Token → 复制
  3. 填入 config/settings.local.yaml:
     data_sources:
       tushare:
         token: "你的Token"

Tushare 积分规则:
  - 注册送 120 积分（够日线数据）
  - 分钟线需要 2000 积分（捐赠 ¥200 获得）
  - 详见: https://tushare.pro/document/1?doc_id=13

用法
--------
>>> from data.fetchers.tushare_fetch import fetch_minute_kline
>>> df = fetch_minute_kline("000001.SZ", freq="5min")  # 5分钟K线
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from config.log import get_logger
logger = get_logger("tushare_fetch")
def _get_token() -> str:
    """从配置文件读取 Tushare Token（通过统一配置加载器）"""
    from config.loader import get_tushare_token
    token = get_tushare_token()
    if token and token != "你的Tushare Token":
        return token
    return ""


def get_pro():
    """获取 Tushare Pro 连接（需要 Token）"""
    import tushare as ts
    token = _get_token()
    if not token:
        logger.warning(
            "Tushare Token 未配置！请:\n"
            "  1. 注册 https://tushare.pro/\n"
            "  2. 在 config/settings.local.yaml 填写 token\n"
            "  3. 日线数据够用(免费)，分钟线需要 ¥200/年"
        )
        return None

    ts.set_token(token)
    return ts.pro_api()


def fetch_minute_kline(
    symbol: str,
    freq: str = "5min",
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """
    拉取分钟级 K 线。

    参数
    ----------
    symbol : str
        Tushare 格式: "000001.SZ"（深圳）, "600519.SH"（上海）
    freq : str
        "1min" / "5min" / "15min" / "30min" / "60min"
    start_date : str
        "YYYYMMDD" 或 "YYYYMMDD HH:MM:SS"
    end_date : str
        同上

    返回
    -------
    DataFrame: ts_code, trade_time, open, high, low, close, vol, amount
    """
    pro = get_pro()
    if pro is None:
        return pd.DataFrame()

    if not start_date:
        start_date = (datetime.now().replace(hour=0, minute=0, second=0)
                      .strftime("%Y%m%d %H:%M:%S"))

    # 映射频率到 Tushare API 参数
    freq_map = {"1min": "1min", "5min": "5min", "15min": "15min",
                "30min": "30min", "60min": "60min"}
    ts_freq = freq_map.get(freq, "5min")

    logger.info(f"拉取 Tushare 分钟线: {symbol} {ts_freq} {start_date}~{end_date or '最新'}")

    try:
        if freq == "1min":
            df = pro.stk_mins(symbol, freq=ts_freq, start_date=start_date, end_date=end_date)
        else:
            # Tushare 的分钟线接口可能因版本而异
            df = pro.stk_mins(ts_code=symbol, freq=ts_freq,
                            start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            logger.info(f"拉取完成: {len(df)} 条")
            return df
        else:
            logger.warning(f"无数据: {symbol}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Tushare 拉取失败: {e}")
        return pd.DataFrame()


def fetch_daily_kline(
    symbol: str,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """
    拉取日线数据（Tushare 版本，比 AkShare 更稳定但需要 Token）。

    AkShare 免费且够用，这个函数主要用于数据校验，
    或者在 AkShare 抽风时作为备份。
    """
    pro = get_pro()
    if pro is None:
        return pd.DataFrame()

    if not start_date:
        start_date = "20200101"
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    logger.info(f"拉取 Tushare 日线: {symbol} {start_date}~{end_date}")

    try:
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            df = df.sort_values("trade_date").reset_index(drop=True)
            logger.info(f"拉取完成: {len(df)} 条")
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Tushare 拉取失败: {e}")
        return pd.DataFrame()


def fetch_stock_list() -> pd.DataFrame:
    """拉取 A 股全部股票列表（含代码、名称、行业、上市日期）"""
    pro = get_pro()
    if pro is None:
        return pd.DataFrame()

    try:
        df = pro.stock_basic(exchange="", list_status="L",
                            fields="ts_code,symbol,name,area,industry,list_date")
        logger.info(f"股票列表: {len(df)} 只")
        return df
    except Exception as e:
        logger.error(f"拉取失败: {e}")
        return pd.DataFrame()


def fetch_income(symbol: str, year: str = "") -> pd.DataFrame:
    """拉取利润表（用于财报分析）"""
    pro = get_pro()
    if pro is None:
        return pd.DataFrame()
    try:
        df = pro.income(ts_code=symbol, end_date=year) if year else pro.income(ts_code=symbol)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"拉取失败: {e}")
        return pd.DataFrame()


# ============================================================
# 命令行测试
# ============================================================
# python data/fetchers/tushare_fetch.py

if __name__ == "__main__":
    print("=" * 60)
    print("Tushare Pro 数据获取")
    print("=" * 60)

    token = _get_token()
    if not token:
        print("\nToken 未配置。")
        print("1. 注册 https://tushare.pro/")
        print("2. 在 config/settings.local.yaml 填写:")
        print("   data_sources:")
        print("     tushare:")
        print("       token: \"你的Token\"")
    else:
        print(f"Token 已配置: {token[:8]}...")

        # 测试拉取股票列表
        stocks = fetch_stock_list()
        if not stocks.empty:
            print(f"\nA股上市公司: {len(stocks)} 家")
            print(stocks.head())
