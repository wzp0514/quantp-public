"""
数据获取降级链 — AkShare → Tushare → Baostock 自动切换

当主数据源（AkShare 爬虫）因源站改版失效时，自动切换备用源，
确保数据不中断。每次切换都会记录日志和健康状态。

降级顺序：
  1. AkShare（免费，覆盖最广，优先使用）
  2. Tushare Pro（需 Token，更稳定，¥200/分钟级）
  3. Baostock（完全免费，仅历史日线，最后兜底）

每日数据完整性检查：
  - 检查最近 N 个交易日是否有缺失
  - 缺失 > 阈值 → 告警

拉取后自动质量检查（fetch_xxx_with_quality）：
  - 拉取 → 完整性检查 → 异常时自动推送通知（钉钉/飞书/Telegram/企微等）

用法
--------
>>> from data.fetchers.fallback import fetch_index_daily_safe, fetch_index_daily_with_quality
>>> df = fetch_index_daily_safe("沪深300", "20240101", "20250601")
>>> print(f"数据源: {df.attrs.get('source')}, {len(df)} 条")
>>>
>>> # 带质量检查 + 异常通知的版本
>>> df = fetch_index_daily_with_quality("沪深300", "20240101")
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

import pandas as pd

from config.log import get_logger

logger = get_logger("data_fallback")

# ── 数据源健康状态 ──────────────────────────────────────────
_source_health: dict[str, dict] = {
    "mootdx": {"status": "unknown", "last_check": None, "error": None},
    "akshare": {"status": "unknown", "last_check": None, "error": None},
    "tushare": {"status": "unknown", "last_check": None, "error": None},
    "baostock": {"status": "unknown", "last_check": None, "error": None},
}


def _update_health(source: str, ok: bool, error: str = ""):
    _source_health[source] = {
        "status": "ok" if ok else "down",
        "last_check": datetime.now().isoformat(),
        "error": error[:200] if error else None,
    }
    # 恢复后自动清除 error 和 down_since
    if ok:
        _source_health[source].pop("down_since", None)
        _source_health[source].pop("error", None)
    elif "down_since" not in _source_health[source]:
        _source_health[source]["down_since"] = datetime.now().isoformat()


def get_source_health() -> dict:
    """获取所有数据源健康状态"""
    return dict(_source_health)


# ── 自动恢复机制 ──────────────────────────────────────────

RESTORE_INTERVAL_MINUTES = 30  # 降级后每30分钟尝试恢复主源


def _should_retry(source: str) -> bool:
    """判断是否应该重试一个已降级的数据源"""
    info = _source_health.get(source, {})
    if info.get("status") != "down":
        return False
    last = info.get("last_check", "")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        elapsed = (datetime.now() - last_dt).total_seconds() / 60
        return elapsed >= RESTORE_INTERVAL_MINUTES
    except (ValueError, TypeError):
        return True


def try_restore(name: str = "沪深300", start: str = "20250101", end: str = "") -> dict:
    """
    尝试恢复已降级的数据源。每个源每30分钟最多重试一次。
    返回: {source: restored(bool), ...}
    """
    if not end:
        end = datetime.now().strftime("%Y%m%d")

    results = {}
    for src in ["mootdx", "akshare", "tushare", "baostock"]:
        if not _should_retry(src):
            continue
        logger.info(f"尝试恢复数据源: {src}...")
        try:
            if src == "mootdx":
                from data.fetchers.mootdx_fetch import fetch_index_daily
                df = fetch_index_daily(name, start, end)
                ok = df is not None and len(df) > 0
            elif src == "akshare":
                from data.fetchers.akshare_fetch import fetch_index_daily
                df = fetch_index_daily(name, start, end)
                ok = df is not None and len(df) > 0
            elif src == "tushare":
                from data.fetchers.tushare_fetch import fetch_daily_kline
                df = fetch_daily_kline("000300.SH", start, end)
                ok = df is not None and len(df) > 0
            elif src == "baostock":
                df = _fetch_baostock_index(name, start, end)
                ok = df is not None and len(df) > 0
            else:
                ok = False
        except Exception as e:
            ok = False
            logger.debug(f"恢复 {src} 失败: {e}")

        _update_health(src, ok)
        results[src] = ok
        if ok:
            logger.info(f"数据源 {src} 已恢复!")
        else:
            logger.debug(f"数据源 {src} 仍未恢复")

    return results


# ── 指数日线降级链 ──────────────────────────────────────────

def fetch_index_daily_safe(
    name: str = "沪深300",
    start_date: str = "20200101",
    end_date: str = "",
) -> pd.DataFrame:
    """
    安全拉取指数日线（自动降级）。

    依次尝试: mootdx → AkShare → Tushare → Baostock
    第一个成功的数据源结果直接返回。
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    # ── 第1层: mootdx（通达信免费行情）──
    if _should_retry("mootdx"):
        logger.info("尝试恢复 mootdx...")
    try:
        from data.fetchers.mootdx_fetch import fetch_index_daily as mootdx_fetch
        df = mootdx_fetch(name, start_date, end_date)
        if df is not None and len(df) > 0:
            _update_health("mootdx", True)
            df.attrs["source"] = "mootdx"
            logger.info(f"mootdx 拉取成功: {name}, {len(df)} 条")
            return df
    except Exception as e:
        _update_health("mootdx", False, str(e))
        logger.warning(f"mootdx 拉取 {name} 失败 → 降级到 AkShare: {e}")

    # ── 第2层: AkShare ──
    if _should_retry("akshare"):
        logger.info("尝试恢复 AkShare...")
    try:
        from data.fetchers.akshare_fetch import fetch_index_daily
        df = fetch_index_daily(name, start_date, end_date)
        if df is not None and len(df) > 0:
            _update_health("akshare", True)
            df.attrs["source"] = "akshare"
            logger.info(f"AkShare 拉取成功: {name}, {len(df)} 条")
            return df
    except Exception as e:
        _update_health("akshare", False, str(e))
        logger.warning(f"AkShare 拉取 {name} 失败 → 降级到 Tushare: {e}")

    # ── 第3层: Tushare Pro ──
    try:
        from data.fetchers.tushare_fetch import fetch_daily_kline
        ts_code = _index_name_to_ts_code(name)
        df = fetch_daily_kline(ts_code, start_date, end_date)
        if df is not None and len(df) > 0:
            _update_health("tushare", True)
            df.attrs["source"] = "tushare"
            logger.info(f"Tushare 拉取成功: {name}({ts_code}), {len(df)} 条")
            return df
    except Exception as e:
        _update_health("tushare", False, str(e))
        logger.warning(f"Tushare 拉取 {name} 失败 → 降级到 Baostock: {e}")

    # ── 第3层: Baostock ──
    try:
        df = _fetch_baostock_index(name, start_date, end_date)
        if df is not None and len(df) > 0:
            _update_health("baostock", True)
            df.attrs["source"] = "baostock"
            logger.info(f"Baostock 拉取成功: {name}, {len(df)} 条")
            return df
    except Exception as e:
        _update_health("baostock", False, str(e))
        logger.warning(f"Baostock 拉取 {name} 失败: {e}")

    # ── 全部失败 ──
    logger.error(f"所有数据源均失败: {name} {start_date}~{end_date}")
    return pd.DataFrame()


def fetch_stock_daily_safe(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    安全拉取个股日线（自动降级）。

    依次尝试: mootdx → AkShare → Tushare → Baostock
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    # ── 第1层: mootdx ──
    try:
        from data.fetchers.mootdx_fetch import fetch_stock_daily as mootdx_stock_fetch
        df = mootdx_stock_fetch(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            _update_health("mootdx", True)
            df.attrs["source"] = "mootdx"
            return df
    except Exception as e:
        _update_health("mootdx", False, str(e))
        logger.warning(f"mootdx 个股 {symbol} 失败 → AkShare: {e}")

    # ── 第2层: AkShare ──
    try:
        from data.fetchers.akshare_fetch import fetch_stock_daily
        df = fetch_stock_daily(symbol, "daily", start_date, end_date, adjust)
        if df is not None and len(df) > 0:
            _update_health("akshare", True)
            df.attrs["source"] = "akshare"
            return df
    except Exception as e:
        _update_health("akshare", False, str(e))
        logger.warning(f"AkShare 个股 {symbol} 失败 → Tushare: {e}")

    # ── 第3层: Tushare ──
    try:
        from data.fetchers.tushare_fetch import fetch_daily_kline
        ts_code = _symbol_to_ts_code(symbol)
        df = fetch_daily_kline(ts_code, start_date, end_date)
        if df is not None and len(df) > 0:
            _update_health("tushare", True)
            df.attrs["source"] = "tushare"
            return df
    except Exception as e:
        _update_health("tushare", False, str(e))
        logger.warning(f"Tushare 个股 {symbol} 失败 → Baostock: {e}")

    # ── 第3层: Baostock ──
    try:
        df = _fetch_baostock_stock(symbol, start_date, end_date, adjust)
        if df is not None and len(df) > 0:
            _update_health("baostock", True)
            df.attrs["source"] = "baostock"
            return df
    except Exception as e:
        _update_health("baostock", False, str(e))

    logger.error(f"所有数据源均失败: {symbol}")
    return pd.DataFrame()


# ── 数据完整性检查 ──────────────────────────────────────────

def check_data_completeness(
    df: pd.DataFrame,
    lookback_days: int = 5,
    max_missing: int = 2,
) -> dict:
    """
    检查最近 N 个交易日是否有缺失。

    参数
    ----------
    df : DataFrame
        需包含 'date' 列
    lookback_days : int
        检查最近多少个日历日
    max_missing : int
        缺失天数超过此值告警

    返回
    -------
    dict: {complete: bool, missing_days: int, last_date: str, warning: str}
    """
    if df is None or df.empty:
        return {
            "complete": False,
            "missing_days": lookback_days,
            "last_date": None,
            "warning": "数据为空",
        }

    if "date" not in df.columns:
        return {"complete": True, "missing_days": 0, "last_date": None, "warning": ""}

    dates = pd.to_datetime(df["date"]).sort_values()
    last_date = dates.max()
    today = datetime.now()

    # 统计最近 lookback_days 个日历日里有多少个交易日有数据
    expected_dates = _get_expected_trade_dates(last_date, today, lookback_days)
    actual_dates = set(d.date() for d in dates if d.date() >= (today - timedelta(days=lookback_days)).date())
    missing = sorted(set(expected_dates) - actual_dates)

    complete = len(missing) <= max_missing
    warning = ""

    if len(missing) > 0:
        missing_str = ", ".join(str(d) for d in missing[-3:])
        warning = f"缺失 {len(missing)} 个交易日（最近: {missing_str}）"
        if not complete:
            logger.warning(f"数据完整性告警: {warning}")

    return {
        "complete": complete,
        "missing_days": len(missing),
        "last_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
        "missing_dates": [str(d) for d in missing],
        "warning": warning,
    }


def get_latest_trade_date() -> Optional[str]:
    """获取最近一个有数据的交易日"""
    try:
        df = fetch_index_daily_safe("沪深300",
                                    start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                                    end_date=datetime.now().strftime("%Y%m%d"))
        if not df.empty and "date" in df.columns:
            return str(pd.to_datetime(df["date"]).max().date())
    except Exception:
        pass
    return None


# ── 带质量检查的数据获取（拉取 + 检查 + 通知）──────────────

def fetch_index_daily_with_quality(
    name: str = "沪深300",
    start_date: str = "20200101",
    end_date: str = "",
    notify: bool = True,
) -> pd.DataFrame:
    """
    拉取指数日线 + 自动质量检查 + 异常通知。

    在 fetch_index_daily_safe 基础上，拉取后自动跑 check_data_completeness，
    如果发现异常（缺失>2天、数据量为0等），自动通过通知渠道推送告警。
    """
    from config.loader import is_quality_check_enabled

    df = fetch_index_daily_safe(name, start_date, end_date)

    if not is_quality_check_enabled():
        return df

    # 质量检查
    result = check_data_completeness(df)

    if not result["complete"] and notify:
        source = df.attrs.get("source", "unknown") if not df.empty else "none"
        msg = _build_quality_alert(name, start_date, end_date, source, result)
        _notify_alert(msg)

    return df


def fetch_stock_daily_with_quality(
    symbol: str,
    start_date: str = "20200101",
    end_date: str = "",
    adjust: str = "qfq",
    notify: bool = True,
) -> pd.DataFrame:
    """拉取个股日线 + 自动质量检查 + 异常通知"""
    from config.loader import is_quality_check_enabled

    df = fetch_stock_daily_safe(symbol, start_date, end_date, adjust)

    if not is_quality_check_enabled():
        return df

    result = check_data_completeness(df)
    if not result["complete"] and notify:
        source = df.attrs.get("source", "unknown") if not df.empty else "none"
        msg = _build_quality_alert(symbol, start_date, end_date, source, result)
        _notify_alert(msg)

    return df


def _build_quality_alert(
    name: str, start: str, end: str, source: str, result: dict
) -> str:
    """构建数据质量告警消息"""
    from datetime import datetime
    return (
        f"[DATA QUALITY] 数据完整性告警\n"
        f"标的: {name}\n"
        f"范围: {start} ~ {end or '今天'}\n"
        f"数据源: {source}\n"
        f"最新日期: {result.get('last_date', 'N/A')}\n"
        f"缺失: {result.get('missing_days', 0)} 个交易日\n"
        f"详情: {result.get('warning', '')}\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _notify_alert(message: str) -> bool:
    """发送数据质量告警（尝试所有可用通知渠道）"""
    try:
        from live.gateway.notifier import send
        return send(message, channel="auto", title="QuantP 数据告警")
    except Exception:
        logger.warning(f"通知发送失败（notifier 不可用），告警内容: {message[:100]}")
        return False


# ── Baostock 内部实现 ───────────────────────────────────────

def _fetch_baostock_index(
    name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """通过 Baostock 拉取指数日线"""
    import baostock as bs

    bs_map = {
        "沪深300": "sh.000300",
        "上证50": "sh.000016",
        "中证500": "sh.000905",
        "上证指数": "sh.000001",
        "创业板指": "sz.399006",
        "深证成指": "sz.399001",
        "科创50": "sh.000688",
    }
    bs_code = bs_map.get(name)
    if not bs_code:
        raise ValueError(f"Baostock 不支持的指数: {name}")

    bs.login()
    try:
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start, end_date=end,
            frequency="d",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"Baostock 查询失败: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    finally:
        bs.logout()


def _fetch_baostock_stock(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """通过 Baostock 拉取个股日线"""
    import baostock as bs

    # Baostock 格式: sh.600519 或 sz.000001
    if symbol.startswith("6") or symbol.startswith("5"):
        bs_code = f"sh.{symbol}"
    else:
        bs_code = f"sz.{symbol}"

    bs.login()
    try:
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        fields = "date,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start, end_date=end,
            frequency="d", adjustflag="2" if adjust == "qfq" else "1",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"Baostock 查询失败: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=fields.split(","))
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    finally:
        bs.logout()


# ── 辅助函数 ─────────────────────────────────────────────────

def _index_name_to_ts_code(name: str) -> str:
    """指数中文名 → Tushare 代码"""
    mapping = {
        "沪深300": "000300.SH",
        "上证50": "000016.SH",
        "中证500": "000905.SH",
        "上证指数": "000001.SH",
        "创业板指": "399006.SZ",
        "深证成指": "399001.SZ",
        "科创50": "000688.SH",
    }
    return mapping.get(name, "000300.SH")


def _symbol_to_ts_code(symbol: str) -> str:
    """个股代码 → Tushare 代码"""
    if symbol.startswith("6") or symbol.startswith("5"):
        return f"{symbol}.SH"
    else:
        return f"{symbol}.SZ"


def _get_expected_trade_dates(
    last_data_date,
    today: datetime,
    lookback_days: int,
) -> list:
    """获取最近 N 天内的预期交易日列表（简化：排除周末）"""
    expected = []
    for i in range(lookback_days):
        d = today - timedelta(days=i)
        if d.weekday() < 5:  # 周一到周五
            expected.append(d.date())
    return expected


# ── 依赖检查（启动时调用）──────────────────────────────────

def check_dependencies() -> dict:
    """检查各数据源是否可用，返回可用状态"""
    result = {}

    # AkShare
    try:
        import akshare
        result["akshare"] = True
    except ImportError:
        result["akshare"] = False
        logger.debug("AkShare 未安装")

    # Tushare
    try:
        from data.fetchers.tushare_fetch import _get_token
        token = _get_token()
        result["tushare"] = bool(token)
    except Exception:
        result["tushare"] = False

    # Baostock
    try:
        import baostock
        result["baostock"] = True
    except ImportError:
        result["baostock"] = False
        logger.debug("Baostock 未安装: pip install baostock")

    available = sum(1 for v in result.values() if v)
    logger.info(f"数据源可用: {available}/3 (AkShare={result['akshare']}, "
                f"Tushare={result['tushare']}, Baostock={result['baostock']})")

    return result


# ── 命令行测试 ──────────────────────────────────────────────
# python data/fetchers/fallback.py

if __name__ == "__main__":
    print("=" * 60)
    print("数据降级链测试")
    print("=" * 60)

    deps = check_dependencies()
    print(f"\n数据源可用性: {deps}")

    print("\n[1] 拉取沪深300日线（自动降级）...")
    df = fetch_index_daily_safe("沪深300", "20250101", "20250601")
    if not df.empty:
        source = df.attrs.get("source", "unknown")
        print(f"  成功: {source}, {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")

        completeness = check_data_completeness(df)
        print(f"  数据完整性: {'OK' if completeness['complete'] else '告警'}")
        if completeness["warning"]:
            print(f"  {completeness['warning']}")
    else:
        print("  失败: 所有数据源均不可用")

    print(f"\n[2] 数据源健康状态:")
    for src, health in get_source_health().items():
        status = health["status"]
        icon = "OK" if status == "ok" else ("DOWN" if status == "down" else "??")
        print(f"  {src}: {icon}")

    print("\n[3] 带质量检查+通知的拉取...")
    df2 = fetch_index_daily_with_quality("沪深300", "20250501", "20250601")
    if not df2.empty:
        comp = check_data_completeness(df2)
        print(f"  数据: {len(df2)} 条, 完整性: {'OK' if comp['complete'] else '告警'}")
        print(f"  (告警已自动推送到已配置的通知渠道)" if not comp["complete"] else "")
