"""
A 股数据获取 — 基于 AkShare（免费开源）

功能：
  - 拉取指数日线（沪深300 / 中证500 / 创业板指）
  - 拉取个股日线（含复权）
  - 拉取 A 股交易日历

⚠️ 幸存者偏差警告：
  AkShare 仅返回当前仍在交易的股票。已退市股票的历史数据不包含在内。
  这意味着回测结果会系统性高估策略收益（约 10-30%）。
  缓解措施：如果将来接入 Tushare Pro 或 Wind，建议使用
  point-in-time 成分股数据（包含退市股的历史记录）。

使用前确保 AkShare 已安装：pip install akshare

参考资料：
  AkShare 官方文档: https://akshare.akfamily.xyz/
"""

import logging
import sys
from datetime import datetime
from typing import Optional

import akshare as ak
import pandas as pd

from config.log import get_logger
logger = get_logger("akshare_fetch")
# ============================================================
# 日志配置（控制台 + 文件双通道）
# ============================================================
# 文件处理器：DEBUG 级别全记录（包括原始数据摘要），存到 notebooks/ 目录
try:
    file_handler = logging.FileHandler("notebooks/fetch.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
except Exception as e:
    logger.debug(f"无法创建文件日志: {e}")


# ============================================================
# 指数代码速查表（注释 = 帮助你理解每个指数的含义）
# ============================================================
# 新手建议：先用沪深300(sh000300)练手，它最稳定、数据最全
INDEX_CODE_MAP = {
    # 上海证券交易所（代码 sh 开头）
    "沪深300":    "sh000300",   # A股最大的300家公司，最常用的大盘基准
    "上证50":     "sh000016",   # 上海市场最大的50家，银行/保险占比高
    "中证500":    "sh000905",   # 排名301-800的中型公司，波动比沪深300大
    "上证指数":   "sh000001",   # 上海市场所有股票的综合指数（常说的"大盘"）

    # 深圳证券交易所（代码 sz 开头）
    "创业板指":   "sz399006",   # 深圳创业板，科技/成长型公司，风险最高
    "深证成指":   "sz399001",   # 深圳市场500家代表性公司
    "中小板指":   "sz399005",   # 深圳中小板公司（已合并到深市主板）
    "中证1000":   "sh000852",   # 排名301-1300的小型公司，波动最大
    "科创50":     "sh000688",   # 上交所科创板50家核心公司，硬科技
    "沪深300成长": "sh000918",   # 沪深300中成长性最强的100只
    "沪深300价值": "sh000919",   # 沪深300中估值最低的100只
}


# ============================================================
# 数据获取函数
# ============================================================

def fetch_index_daily(
    name: str = "沪深300",
    start_date: str = "20200101",
    end_date: str = "",
) -> pd.DataFrame:
    """
    拉取指数日线数据（每天的开盘价、最高价、最低价、收盘价、成交量）

    参数
    ----------
    name : str
        指数中文名。可选：沪深300、上证50、中证500、上证指数、创业板指、深证成指、中小板指
        不填默认沪深300
    start_date : str
        开始日期，格式 YYYYMMDD（例如 "20200101" 表示 2020年1月1日）
        不填默认从 2020 年开始
    end_date : str
        结束日期，格式同上。不填默认到今天

    返回
    -------
    pandas.DataFrame，包含以下列：
        date   — 交易日期
        open   — 开盘价（9:30 第一笔成交价）
        high   — 最高价（当天最高成交价）
        low    — 最低价（当天最低成交价）
        close  — 收盘价（15:00 最后一笔成交价）
        volume — 成交量（当天成交了多少股）

    示例
    --------
    >>> df = fetch_index_daily("沪深300", "20240101", "20241231")
    >>> print(df.head())  # 看前5行
    >>> print(f"共 {len(df)} 条数据")
    """
    # 1. 查代码
    code = INDEX_CODE_MAP.get(name)
    if code is None:
        supported = "、".join(INDEX_CODE_MAP.keys())
        logger.error(f"不支持的指数: {name}，可选: {supported}")
        raise ValueError(f"不支持的指数「{name}」，可选：{supported}")

    # 2. 如果没填结束日期，默认用今天
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    logger.info(f"开始拉取 {name}({code}) 日线数据，时间范围: {start_date} ~ {end_date}")

    try:
        # 3. 调用 AkShare API
        #    AkShare 这个函数名比较长，但参数很简单：
        #    symbol=指数代码，start_date=开始日期，end_date=结束日期
        #    返回的是一张表（DataFrame），每行是一个交易日
        df = ak.stock_zh_index_daily(symbol=code)

        logger.debug(f"API 返回原始数据: {len(df)} 条（未筛选日期前）")

        # 4. 按日期范围筛选
        #    AkShare 的 date 列是字符串类型（如 "2024-01-15"），需要先转成日期再比较
        df["date"] = pd.to_datetime(df["date"])

        # 筛选：只保留 start_date 到 end_date 之间的交易日
        mask = (df["date"] >= pd.to_datetime(start_date)) & \
               (df["date"] <= pd.to_datetime(end_date))
        df = df[mask].copy()
        df = df.reset_index(drop=True)  # 重新编号，从0开始

        # 5. 按日期升序排列（最旧的在前，最新的在后）
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(
            f"拉取完成: {name}({code})，"
            f"共 {len(df)} 条，"
            f"日期范围: {df['date'].min().strftime('%Y-%m-%d')} ~ "
            f"{df['date'].max().strftime('%Y-%m-%d')}"
        )

        return df

    except Exception as e:
        logger.error(f"拉取 {name}({code}) 失败: {type(e).__name__}: {e}")
        raise


def fetch_stock_daily(
    symbol: str,
    period: str = "daily",
    start_date: str = "20200101",
    end_date: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    拉取个股日线数据（单只股票）

    参数
    ----------
    symbol : str
        股票代码，例如 "000300" 或 "600519"
        注意：不要带 sh/sz 前缀，直接写数字
    period : str
        周期，"daily"=日线，"weekly"=周线，"monthly"=月线
    start_date : str
        开始日期，格式 YYYYMMDD
    end_date : str
        结束日期，格式 YYYYMMDD。不填默认到今天
    adjust : str
        复权方式：
        "qfq"  = 前复权（推荐，把历史价格按最新股本调整，回测首选）
        "hfq"  = 后复权（把最新价格按上市时股本调整）
        ""     = 不复权（原始价格，不推荐用于回测）

    返回
    -------
    pandas.DataFrame

    示例
    --------
    >>> df = fetch_stock_daily("600519", adjust="qfq")  # 贵州茅台，前复权
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    # AkShare 的 adjust 参数值：qfq=前复权, hfq=后复权, 空=不复权
    adjust_map = {
        "qfq": "qfq",
        "hfq": "hfq",
        "": "",
    }
    ak_adjust = adjust_map.get(adjust)
    if ak_adjust is None:
        raise ValueError(f"不支持的复权方式: {adjust}，可选: qfq/hfq/空字符串")

    logger.info(f"开始拉取个股 {symbol} ({period}, {adjust or '不复权'}): {start_date} ~ {end_date}")

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if df.empty:
            logger.warning(f"个股 {symbol} 返回空数据，可能代码错误或停牌中")
            return df

        # 统一列名（AkShare 返回的中文列名转英文，和指数数据保持一致）
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude", "涨跌幅": "pct_change",
            "涨跌额": "change", "换手率": "turnover",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        logger.info(
            f"拉取完成: {symbol}，共 {len(df)} 条，"
            f"日期: {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'} ~ "
            f"{df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}"
        )

        return df

    except Exception as e:
        logger.error(f"拉取个股 {symbol} 失败: {type(e).__name__}: {e}")
        raise


def fetch_trade_calendar(start_year: int = 2020, end_year: int = 2026) -> pd.DataFrame:
    """
    拉取 A 股交易日历（哪些日子是交易日，哪些不是）

    作用：回测时判断某天能不能交易——周末和节假日股市不交易，
         你不能在除夕那天买入股票。

    返回
    -------
    DataFrame，包含 trade_date 列（datetime 类型），每行是一个交易日

    示例
    --------
    >>> calendar = fetch_trade_calendar(2024, 2024)
    >>> print(f"2024年共有 {len(calendar)} 个交易日")  # 约 242-244 天
    """
    logger.info(f"拉取 A 股交易日历: {start_year}-{end_year}")

    try:
        df = ak.tool_trade_date_hist_sina()
        df = df.rename(columns={"trade_date": "trade_date"})  # 统一列名
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 筛选年份范围
        mask = (df["trade_date"].dt.year >= start_year) & \
               (df["trade_date"].dt.year <= end_year)
        df = df[mask].sort_values("trade_date").reset_index(drop=True)

        logger.info(f"交易日历: {start_year}-{end_year} 共 {len(df)} 个交易日")
        return df

    except Exception as e:
        logger.error(f"拉取交易日历失败: {type(e).__name__}: {e}")
        raise


# ============================================================
# 命令行快速测试
# ============================================================
# 你可以直接在终端运行这个文件来测试：
#   cd .
#   source venv/Scripts/activate
#   python data/fetchers/akshare_fetch.py
#
# 这会在 notebooks/ 目录下生成一个 CSV 文件，可以用 Excel 打开看

if __name__ == "__main__":
    print("=" * 60)
    print("AkShare 数据获取测试")
    print("=" * 60)

    # 测试1：拉取沪深300近2年日线
    print("\n[测试1] 沪深300 日线数据（2024-01-01 ~ 今天）")
    print("-" * 40)
    df_hs300 = fetch_index_daily("沪深300", "20240101")
    print(f"结果: {len(df_hs300)} 条")
    print(df_hs300.tail(5))  # 打印最近5天

    # 测试2：拉取交易日历
    print("\n[测试2] A股交易日历（2024年）")
    print("-" * 40)
    df_cal = fetch_trade_calendar(2024, 2024)
    print(f"2024年共 {len(df_cal)} 个交易日")

    # 保存到 CSV（方便用 Excel 打开看）
    save_path = "notebooks/hs300_test.csv"
    df_hs300.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n数据已保存到 {save_path}（可用 Excel 打开）")
