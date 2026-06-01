"""
实时行情轮询源 — AkShare 免费实时行情

AkShare 不提供 WebSocket，用 HTTP 轮询 stock_zh_a_spot_em() 获取全市场实时价。
交易时段内按配置间隔轮询，非交易时段休眠。
"""

import logging
import threading
import time
from datetime import datetime, time as dtime
from typing import Callable, Optional

from config.log import get_logger

logger = get_logger("realtime_feed")

# A股交易时段
_MORNING_START = dtime(9, 30)
_MORNING_END = dtime(11, 30)
_AFTERNOON_START = dtime(13, 0)
_AFTERNOON_END = dtime(15, 0)


class RealTimeFeed:
    """AkShare 实时行情轮询源。

    在后台线程中按固定间隔轮询，通过回调推送最新价格快照。
    仅交易时段轮询，非交易时段休眠到下一时段。

   用法
   --------
   >>> feed = RealTimeFeed(["000300", "000905"], poll_interval=5.0)
   >>> def on_snapshot(data):
   ...     print(data["000300"]["price"])
   >>> feed.start(on_snapshot)
   >>> # ... 运行中 ...
   >>> feed.stop()
    """

    def __init__(
        self,
        symbols: list[str],
        poll_interval: float = 5.0,
        source: str = "akshare",
    ):
        """
        参数
        ----------
        symbols : list[str]
            股票/指数代码列表，如 ["000300", "600519"]。
            支持带前缀或不带前缀的6位代码。
        poll_interval : float
            轮询间隔（秒），默认5秒
        source : str
            数据源（目前仅支持 akshare）
        """
        self.symbols = [s[-6:] if s.startswith(("sh", "sz")) else s for s in symbols]
        self.poll_interval = poll_interval
        self.source = source

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[dict], None]] = None
        self._latest: dict[str, dict] = {}
        self._error_count = 0
        self._max_errors = 10
        self._lock = threading.Lock()

    @staticmethod
    def is_trading_time() -> bool:
        """当前是否在 A 股交易时段内（周一至周五 9:30-11:30 或 13:00-15:00）。"""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return (_MORNING_START <= t <= _MORNING_END or
                _AFTERNOON_START <= t <= _AFTERNOON_END)

    @staticmethod
    def time_to_next_session() -> float:
        """距离下一个交易时段的秒数。"""
        now = datetime.now()
        today = now.date()
        am_start = datetime.combine(today, _MORNING_START)
        pm_start = datetime.combine(today, _AFTERNOON_START)
        next_day_am = datetime.combine(
            today + __import__('datetime').timedelta(days=1), _MORNING_START
        )

        t = now.time()
        if now.weekday() >= 5:
            # 周末：跳到周一上午
            days_to_monday = 7 - now.weekday()
            monday = datetime.combine(today + __import__('datetime').timedelta(days=days_to_monday), _MORNING_START)
            return (monday - now).total_seconds()
        elif t < _MORNING_START:
            return (am_start - now).total_seconds()
        elif t < _MORNING_END:
            return 0  # 正在交易中
        elif t < _AFTERNOON_START:
            return (pm_start - now).total_seconds()
        elif t < _AFTERNOON_END:
            return 0  # 正在交易中
        else:
            return (next_day_am - now).total_seconds()

    def poll_once(self) -> dict[str, dict]:
        """单次轮询全市场实时行情，返回 {symbol: {price, open, high, low, volume, timestamp}}。

        非交易时段返回空字典。
        """
        if not self.is_trading_time():
            return {}

        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()

            result = {}
            for sym in self.symbols:
                row = df[df["代码"] == sym]
                if not row.empty:
                    r = row.iloc[0]
                    result[sym] = {
                        "price": float(r["最新价"]),
                        "open": float(r["今开"]),
                        "high": float(r["最高"]),
                        "low": float(r["最低"]),
                        "volume": float(r["成交量"]),
                        "change_pct": float(r["涨跌幅"]),
                        "timestamp": datetime.now(),
                    }
            self._error_count = 0
            return result
        except Exception as e:
            self._error_count += 1
            logger.warning(f"轮询失败 ({self._error_count}/{self._max_errors}): {e}")
            if self._error_count >= self._max_errors:
                logger.error(f"连续 {self._max_errors} 次轮询失败，将停止推送")
            return {}

    def _loop(self):
        """后台轮询循环。"""
        logger.info(f"实时行情轮询启动: {self.symbols}, 间隔 {self.poll_interval}s")
        while self._running:
            if self.is_trading_time():
                snapshot = self.poll_once()
                if snapshot:
                    with self._lock:
                        self._latest = snapshot
                    if self._callback:
                        try:
                            self._callback(snapshot)
                        except Exception as e:
                            logger.error(f"回调异常: {e}")
                elif self._error_count >= self._max_errors:
                    logger.warning("轮询连续失败，停止实时行情推送")
                    self._running = False
                    break
                time.sleep(self.poll_interval)
            else:
                wait = self.time_to_next_session()
                logger.debug(f"非交易时段，休眠 {wait:.0f}s")
                # 最长休眠5分钟，避免跳过整个周末
                time.sleep(min(wait, 300))

    def start(self, callback: Callable[[dict], None]) -> None:
        """启动后台轮询线程。

        参数
        ----------
        callback : callable
            每轮轮询成功后调用，参数为 {symbol: {price, ...}} 的字典。
        """
        if self._running:
            logger.warning("RealTimeFeed 已在运行")
            return
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("RealTimeFeed 已启动")

    def stop(self) -> None:
        """停止轮询线程。"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("RealTimeFeed 已停止")

    def get_latest(self, symbol: str) -> Optional[dict]:
        """获取指定标的的最新快照。"""
        with self._lock:
            return self._latest.get(symbol[-6:])

    def get_all_latest(self) -> dict:
        """获取全部标的最新快照的副本。"""
        with self._lock:
            return dict(self._latest)

    @property
    def error_count(self) -> int:
        return self._error_count


def make_index_feed(index_name: str = "沪深300", poll_interval: float = 5.0) -> RealTimeFeed:
    """创建指数的 RealTimeFeed。

    参数
    ----------
    index_name : str
        指数名称，如 "沪深300"、"中证500"
    poll_interval : float
        轮询间隔

    返回
    -------
    RealTimeFeed
    """
    code_map = {
        "沪深300": "000300",
        "中证500": "000905",
        "上证50": "000016",
        "创业板指": "399006",
        "科创50": "000688",
    }
    code = code_map.get(index_name, "000300")
    return RealTimeFeed([code], poll_interval=poll_interval)


# ── 命令行测试 ──────────────────────────────────────────────
# python live/feed/realtime_feed.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=" * 60)
    print("RealTimeFeed 测试")
    print(f"交易时段: {RealTimeFeed.is_trading_time()}")
    print(f"距下次开市: {RealTimeFeed.time_to_next_session():.0f}s")
    print("=" * 60)

    feed = make_index_feed("沪深300", poll_interval=5.0)

    def print_snapshot(data):
        for sym, info in data.items():
            print(f"[{info['timestamp'].strftime('%H:%M:%S')}] {sym}: "
                  f"{info['price']:.2f} ({info['change_pct']:+.2f}%)")

    if RealTimeFeed.is_trading_time():
        feed.start(print_snapshot)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            pass
        finally:
            feed.stop()
    else:
        print("当前非交易时段，仅测试单次轮询...")
        result = feed.poll_once()
        if result:
            print_snapshot(result)
        else:
            print("（非交易时段或网络不可用，无行情数据）")
