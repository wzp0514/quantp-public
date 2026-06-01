"""
A 股交易日历

作用：判断某一天能不能交易。
      A 股只在周一到周五交易（周末休市），且春节/国庆等法定节假日也不开市。
      回测时必须用交易日而非自然日，否则会算错收益率。

全部逻辑基于 AkShare 的官方交易日历数据。
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import akshare as ak
import pandas as pd# ============================================================
# 交易日历缓存（拉一次，一直用）
# ============================================================
# calendar 是一组 Python date 对象，放在 set 里查起来飞快
# 第一次调用时自动从 AkShare 拉取，之后的调用直接用缓存

_trade_dates: Optional[set] = None
_cache_year_range: tuple = (0, 0)  # 记录缓存覆盖的年份范围

from config.log import get_logger
logger = get_logger("trade_calendar")
def _ensure_calendar(year: int) -> None:
    """
    确保交易日历覆盖了指定年份。
    如果还没加载，或加载的范围不够，就重新拉取。
    """
    global _trade_dates, _cache_year_range

    min_year, max_year = _cache_year_range
    if _trade_dates is not None and min_year <= year <= max_year:
        return  # 缓存命中，不用重新拉

    # 拉取范围：传进来的 year 往前推 5 年，往后推 1 年
    start_year = min(year - 5, 2015)
    end_year = year + 1

    logger.info(f"加载 A 股交易日历: {start_year} ~ {end_year}")

    try:
        df = ak.tool_trade_date_hist_sina()
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        mask = (df["trade_date"].dt.year >= start_year) & \
               (df["trade_date"].dt.year <= end_year)
        dates = df[mask]["trade_date"].dt.date.tolist()

        _trade_dates = set(dates)
        _cache_year_range = (start_year, end_year)

        logger.info(f"交易日历加载完成: 共 {len(_trade_dates)} 个交易日")

    except Exception as e:
        logger.error(f"加载交易日历失败: {e}")
        raise


def is_trading_day(d: date) -> bool:
    """
    判断某个日期是不是交易日

    参数
    ----------
    d : datetime.date
        要检查的日期（年月日）

    返回
    -------
    bool : True = 能交易，False = 不能交易（周末或节假日）

    示例
    --------
    >>> from datetime import date
    >>> is_trading_day(date(2025, 1, 1))   # 元旦，休市 → False
    >>> is_trading_day(date(2025, 1, 2))   # 周四，正常交易 → True
    """
    _ensure_calendar(d.year)
    return d in _trade_dates


def next_trading_day(d: date) -> date:
    """
    找到 d 之后（含 d 自己）的第一个交易日

    如果你在周五发出信号但周五不是交易日（比如国庆调休），
    用这个函数找到下一个真正的交易日。

    示例
    --------
    >>> from datetime import date
    >>> next_trading_day(date(2025, 1, 1))   # 元旦，找下一个交易日
    """
    _ensure_calendar(d.year + 1)  # 确保覆盖到可能出现的最晚日期

    current = d
    # 最多找 14 天（春节期间可能连续休市 7 天+前后周末）
    for _ in range(14):
        if current in _trade_dates:
            return current
        current += timedelta(days=1)

    raise ValueError(f"在 {d} 之后的 14 天内找不到交易日，请检查交易日历")


def prev_trading_day(d: date) -> date:
    """
    找到 d 之前（含 d 自己）的第一个交易日

    示例
    --------
    >>> prev_trading_day(date(2025, 5, 1))   # 劳动节，找前一个交易日
    """
    _ensure_calendar(d.year)

    current = d
    for _ in range(14):
        if current in _trade_dates:
            return current
        current -= timedelta(days=1)

    raise ValueError(f"在 {d} 之前的 14 天内找不到交易日，请检查交易日历")


def count_trading_days(start: date, end: date) -> int:
    """
    计算两个日期之间有多少个交易日

    示例
    --------
    >>> from datetime import date
    >>> count_trading_days(date(2024, 1, 1), date(2024, 12, 31))
    242   # 2024年有242个交易日
    """
    _ensure_calendar(end.year)

    count = 0
    current = start
    while current <= end:
        if current in _trade_dates:
            count += 1
        current += timedelta(days=1)

    return count


# ============================================================
# 命令行测试
# ============================================================
# python data/cleaner/calendar.py

if __name__ == "__main__":
    today = date.today()
    print(f"今天是: {today}")
    print(f"是交易日吗？ {'是' if is_trading_day(today) else '否（休市）'}")

    if not is_trading_day(today):
        nxt = next_trading_day(today)
        print(f"下一个交易日: {nxt}")

    # 统计 2024 年交易日
    n_2024 = count_trading_days(date(2024, 1, 1), date(2024, 12, 31))
    print(f"2024 年共 {n_2024} 个交易日（全年约 250 个自然日工作日，假期扣除约 8 天）")
