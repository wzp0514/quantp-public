"""
多品种截面数据 — 拉取 50-100 只股票日线，用于截面因子计算

用法
--------
>>> from data.fetchers.multi_stock import fetch_multi_stock_daily
>>> panel = fetch_multi_stock_daily(n_stocks=50, start="20200101")
>>> # panel 是 dict: {symbol: DataFrame, ...}
"""

import logging
import time
from typing import Optional

import pandas as pd

from config.log import get_logger

logger = get_logger("multi_stock")


def _get_hs300_constituents() -> list[str]:
    """获取沪深 300 成分股列表（通过 AkShare）"""
    try:
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex("000300")
        return df["成分券代码"].head(50).tolist()
    except Exception:
        pass
    # fallback: hardcoded top 50 by weight (as of 2025)
    return [
        "600519", "000858", "601318", "600036", "000333", "600900", "601166",
        "600030", "000651", "002415", "601398", "000568", "600276", "600887",
        "601288", "600309", "000002", "002594", "601939", "600585", "000725",
        "600809", "600690", "601668", "002142", "000001", "600031", "601899",
        "600406", "002304", "601328", "600000", "002352", "000625", "600048",
        "002714", "601088", "601601", "600016", "000338", "002475", "600104",
        "000063", "002050", "601857", "002271", "000157", "000895", "601818",
    ][:50]


def fetch_multi_stock_daily(
    symbols: list[str] = None,
    n_stocks: int = 50,
    start: str = "20200101",
    end: str = "",
) -> dict[str, pd.DataFrame]:
    """
    拉取多只股票日线数据，返回 {symbol: DataFrame}。

    每只股票拉取失败会静默跳过（某些股票可能已退市/停牌）。
    自动做前复权处理。
    """
    if symbols is None:
        symbols = _get_hs300_constituents()[:n_stocks]

    if not end:
        end = pd.Timestamp.now().strftime("%Y%m%d")

    result = {}
    failed = 0

    logger.info(f"Fetching {len(symbols)} stocks from AkShare...")
    t0 = time.time()

    for sym in symbols:
        try:
            import akshare as ak
            # 格式: sh600519 或 sz000001
            if sym.startswith("6") or sym.startswith("5"):
                code = f"sh{sym}"
            else:
                code = f"sz{sym}"

            df = ak.stock_zh_a_daily(symbol=code, start_date=start, end_date=end, adjust="qfq")
            if df is None or len(df) < 100:
                failed += 1
                continue

            df = df.rename(columns={
                "date": "date", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            })
            if "turnover" not in df.columns and "turnover_rate" in df.columns:
                df["turnover"] = df["turnover_rate"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df["symbol"] = sym
            result[sym] = df

        except Exception as e:
            failed += 1
            logger.debug(f"Failed {sym}: {e}")

    elapsed = time.time() - t0
    logger.info(f"Fetched {len(result)}/{len(symbols)} stocks ({failed} failed), {elapsed:.1f}s")
    return result


def build_panel(symbol_data: dict[str, pd.DataFrame],
                date_col: str = "date") -> pd.DataFrame:
    """
    将 {symbol: df} 转为标准化面板 DataFrame。

    columns: date, symbol, close, volume, ...
    每行 = 一个交易日 × 一只股票。
    """
    frames = []
    for sym, df in symbol_data.items():
        f = df[["date", "close", "volume", "open", "high", "low"]].copy()
        f["symbol"] = sym
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    从面板数据构建截面特征矩阵。

    每个 date × symbol 一行，计算基础时序特征：
      - 日收益率
      - 5日/20日/60日动量
      - 20日波动率
      - 成交量变化
      - 换手率（如有）
    """
    df = panel.copy()
    df = df.sort_values(["symbol", "date"])

    # 按股票分组计算时序特征
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
    df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["ret_20d"] = df.groupby("symbol")["close"].pct_change(20)
    df["ret_60d"] = df.groupby("symbol")["close"].pct_change(60)
    df["vol_20d"] = df.groupby("symbol")["ret_1d"].transform(lambda x: x.rolling(20).std())
    df["vol_chg"] = df.groupby("symbol")["volume"].transform(
        lambda x: x / x.rolling(20).mean() - 1)
    df["ma20_dev"] = df["close"] / df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(20).mean()) - 1
    if "turnover" in df.columns:
        df["turnover_5d"] = df.groupby("symbol")["turnover"].transform(
            lambda x: x.rolling(5).mean())

    return df.dropna()


# ============================================================
# 命令行测试
# ============================================================
# python data/fetchers/multi_stock.py

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Stock Data Fetch Test")
    print("=" * 60)

    data = fetch_multi_stock_daily(n_stocks=10, start="20240101")
    print(f"\nFetched {len(data)} stocks")

    if data:
        panel = build_panel(data)
        print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} stocks")
        print(f"Date range: {panel['date'].min().date()} ~ {panel['date'].max().date()}")

        features = build_feature_matrix(panel)
        print(f"Features: {len(features)} rows, {len(features.columns)} columns")
