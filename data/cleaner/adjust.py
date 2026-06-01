"""
复权处理 — 修正股票价格因分红送股导致的"价格断层"

一句话理解复权：
    假设一只股票昨天收盘 10 元，今天每 10 股送 10 股（股本翻倍），
    股价会变成 5 元（因为股票数量翻倍了，总市值不变）。
    如果你看 K 线图，会发现价格从 10 元"跳"到 5 元——这是假的跌幅。
    复权就是把昨天及之前的价格按比例缩放，消除这种"假跳空"。

三种复权方式：
  - 不复权：原始价格，有"假跳空"，不适合回测
  - 前复权：把历史价格按最新股本调整（推荐！回测标配）
    例子：如果现在 1 股 = 过去的 2 股（因为送股），那过去 10 元要先除 2，变成 5 元
  - 后复权：把最新价格按最初股本调整
    例子：现在价格膨胀回去，看"如果从来没送过股，现在该多少钱"

参考：
  前复权/后复权都保持了收益率计算正确。
  AkShare 已内置复权处理，拉取时传 adjust="qfq" 即可。
  本模块提供手动复权和价格连续性校验。
"""

import logging
from typing import Optional

import pandas as pd

from config.log import get_logger
logger = get_logger("adjust")
def check_price_gap(
    close_prices: pd.Series,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """
    检查收盘价序列中是否存在"价格跳空"——前后两天价格变化超过阈值

    跳空可能的原因：
      1. 未复权，股票发生了除权除息（分红/送股导致价格突降）
      2. 真实的大涨大跌（如涨停/跌停）

    回测时必须使用已复权的数据，否则"假跳空"会被当成策略盈亏。

    参数
    ----------
    close_prices : pd.Series
        收盘价序列
    threshold_pct : float
        跳空阈值（%），默认 5%。超过这个百分比就认为有跳空

    返回
    -------
    DataFrame，列出所有跳空日期和幅度

    示例
    --------
    >>> from data.fetchers.akshare_fetch import fetch_stock_daily
    >>> df = fetch_stock_daily("600519", adjust="")   # 不复权
    >>> gaps = check_price_gap(df["close"])
    >>> print(f"发现 {len(gaps)} 个价格跳空")
    """
    if len(close_prices) < 2:
        logger.warning("数据不足 2 行，无法检查跳空")
        return pd.DataFrame()

    # 计算每日收益率（%）：(今天 - 昨天) / 昨天 × 100
    returns = close_prices.pct_change() * 100

    # 找到变化超过阈值的行（排除第一行，因为 pct_change 第一行是 NaN）
    gap_mask = returns.abs() > threshold_pct
    gap_indices = gap_mask[gap_mask].index

    if len(gap_indices) == 0:
        logger.info(f"未发现价格跳空（阈值 ±{threshold_pct}%）")
        return pd.DataFrame()

    # 构建结果表
    result_rows = []
    for idx in gap_indices:
        pos = close_prices.index.get_loc(idx)
        if pos == 0:
            continue
        prev_idx = close_prices.index[pos - 1]
        result_rows.append({
            "date": idx,
            "prev_close": close_prices[prev_idx],
            "close": close_prices[idx],
            "change_pct": round(returns[idx], 2),
        })

    result = pd.DataFrame(result_rows)
    logger.warning(f"发现 {len(result)} 个价格跳空（阈值 ±{threshold_pct}%）：可能是未复权导致")
    return result


def forward_adjust_manual(
    df: pd.DataFrame,
    adjust_factor_col: str = "adjust_factor",
) -> pd.DataFrame:
    """
    手动前复权（如果 AkShare 返回的原始数据需要自行处理时使用）

    大多数情况下不需要这个函数——AkShare 的 adjust="qfq" 已经帮你复权了。
    这个函数留作备选，当数据源不提供复权功能时使用。

    前复权逻辑：
      从最新日期往前算，每个历史价格 × 累计复权因子。
      复权因子列如果不提供，无法计算（需要除权除息公告数据）。

    参数
    ----------
    df : DataFrame
        包含 open/high/low/close/volume 和复权因子列的数据
    adjust_factor_col : str
        复权因子列名

    返回
    -------
    DataFrame（新增 adj_open/adj_high/adj_low/adj_close 列）
    """
    if adjust_factor_col not in df.columns:
        logger.error(f"缺少复权因子列: {adjust_factor_col}，无法手动复权")
        raise ValueError(f"数据中无 {adjust_factor_col} 列，请使用 AkShare adjust='qfq' 参数自动复权")

    df = df.sort_values("date").reset_index(drop=True)

    # 从最近日期向前累积复权因子
    df["_cum_factor"] = df[adjust_factor_col][::-1].cumprod()[::-1]

    # 复权价格列
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[f"adj_{col}"] = df[col] * df["_cum_factor"]

    logger.info("手动前复权完成")
    return df


# ============================================================
# 命令行测试
# ============================================================
# python data/cleaner/adjust.py

if __name__ == "__main__":
    from data.fetchers.fallback import fetch_stock_daily_safe as fetch_stock_daily
    print("=" * 60)
    print("复权测试：对比不复权 vs 前复权的价格差异")
    print("=" * 60)

    # 拉取同一只股票的不复权和前复权数据
    print("\n拉取数据...")
    df_raw = fetch_stock_daily("600519", adjust="")    # 不复权
    df_qfq = fetch_stock_daily("600519", adjust="qfq")  # 前复权

    if not df_raw.empty and not df_qfq.empty:
        # 检查不复权数据的跳空
        print("\n不复权数据的价格跳空检查:")
        gaps = check_price_gap(df_raw["close"], threshold_pct=5.0)
        if not gaps.empty:
            print(gaps.head(10))

        print("\n前复权数据的价格跳空检查:")
        gaps_qfq = check_price_gap(df_qfq["close"], threshold_pct=5.0)
        if gaps_qfq.empty:
            print("前复权数据无跳空（正常）")

        # 对比最近5天的价格
        print("\n最近5天价格对比（不复权 vs 前复权）:")
        compare = pd.DataFrame({
            "date": df_raw["date"].tail(5).values,
            "raw_close": df_raw["close"].tail(5).values,
            "qfq_close": df_qfq["close"].tail(5).values,
        })
        print(compare)
