"""
纸上交易守护进程 — 长期运行 + 自动恢复 + 周期报告

用法
--------
>>> daemon = PaperDaemon()
>>> daemon.start_instance("MaCrossStrategy", "000300", 100000, {"fast": 5, "slow": 20})
>>> daemon.run_forever()   # 阻塞直到手动停止
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from config.log import get_logger
from config.loader import get_config

logger = get_logger("paper_daemon")

_REPORT_DIR = "reports/paper"


class PaperDaemon:
    """纸上交易守护进程。

    负责：
    - 创建和管理纸上交易实例
    - 启动时自动恢复未完成实例
    - 定时生成日/周报
    - 与 RealTimeFeed 联动
    """

    def __init__(self):
        from live.execution.paper_store import PaperTradeStore
        self.store = PaperTradeStore()
        self.active_traders: dict[int, object] = {}  # instance_id -> PaperTrader
        self.feeds: dict = {}
        self._running = False
        self._lock = threading.Lock()
        os.makedirs(_REPORT_DIR, exist_ok=True)

        cfg = get_config().get("paper_trading", {}) if get_config else {}
        self.daemon_cfg = cfg.get("daemon", {})
        self.league_cfg = cfg.get("league", {})

    # ── 实例生命周期 ─────────────────────────────────────────

    def start_instance(
        self,
        strategy_name: str,
        symbol: str = "000300",
        initial_cash: float = 100000.0,
        params: dict = None,
    ) -> int:
        """创建并启动一个新的纸上交易实例。返回 instance_id。"""
        from live.paper_trader import PaperTrader
        from backtest.strategy_market import ALL_STRATEGIES

        params = params or {}
        cls = self._find_strategy(strategy_name)
        if not cls:
            raise ValueError(f"未找到策略: {strategy_name}")

        iid = self.store.create_instance(strategy_name, symbol, initial_cash, params)

        trader = PaperTrader(
            cls, __import__('pandas').DataFrame(),
            initial_cash=initial_cash,
            store=self.store,
            instance_id=iid,
            **params,
        )
        with self._lock:
            self.active_traders[iid] = trader

        logger.info(f"纸上交易实例 #{iid} 已创建: {strategy_name} ({symbol})")
        return iid

    def resume_all(self):
        """启动时恢复所有 running 状态的实例。"""
        from live.paper_trader import PaperTrader

        instances = self.store.load_running_instances()
        if not instances:
            logger.info("无待恢复的纸上交易实例")
            return

        logger.info(f"恢复 {len(instances)} 个纸上交易实例...")
        for inst in instances:
            cls = self._find_strategy(inst["strategy"])
            if not cls:
                logger.warning(f"策略 {inst['strategy']} 不存在，跳过实例 #{inst['id']}")
                self.store.close_instance(inst["id"], "failed")
                continue

            params = json.loads(inst.get("params", "{}"))
            trader = PaperTrader(
                cls, __import__('pandas').DataFrame(),
                initial_cash=inst["initial_cash"],
                store=self.store,
                instance_id=inst["id"],
                **params,
            )

            # 从 store 恢复持仓/订单状态
            full = self.store.load_instance(inst["id"])
            for order in full.get("orders", []):
                if order.get("status") == "filled":
                    trader.order_mgr.orders.append(order)

            for pos_data in full.get("positions", []):
                if pos_data["size"] > 0:
                    trader.position_mgr.update_position(
                        pos_data["symbol"], "buy",
                        pos_data["size"], pos_data["avg_cost"],
                        datetime.now().date()
                    )

            with self._lock:
                self.active_traders[inst["id"]] = trader
            logger.info(f"  实例 #{inst['id']} {inst['strategy']} 已恢复")

    def stop_instance(self, instance_id: int):
        """停止指定实例。"""
        self.store.close_instance(instance_id, "stopped")
        with self._lock:
            self.active_traders.pop(instance_id, None)
        logger.info(f"实例 #{instance_id} 已停止")

    def stop_all(self):
        """停止所有活跃实例。"""
        for iid in list(self.active_traders.keys()):
            self.stop_instance(iid)

    # ── 主循环 ────────────────────────────────────────────────

    def run_forever(self):
        """启动守护进程主循环（阻塞）。"""
        from live.feed.realtime_feed import RealTimeFeed

        self._running = True

        if self.daemon_cfg.get("auto_resume", True):
            self.resume_all()

        if not self.active_traders:
            logger.warning("无活跃的纸上交易实例，守护进程退出")
            return

        # 提取所有实例的标的，创建共享行情源
        symbols = set()
        for inst in self.store.load_running_instances():
            if inst.get("symbol"):
                symbols.add(inst["symbol"])

        poll_interval = self.daemon_cfg.get("poll_interval", 5)
        feed = RealTimeFeed(list(symbols), poll_interval=poll_interval)

        def on_snapshot(snap: dict):
            with self._lock:
                for iid, trader in self.active_traders.items():
                    inst = self.store.load_instance(iid)
                    sym = inst.get("symbol", "")
                    if sym and sym[-6:] in snap:
                        try:
                            trader.step_live(sym, snap[sym[-6:]])
                        except Exception as e:
                            logger.error(f"实例 #{iid} step_live 异常: {e}")

        last_report_time = datetime.now()

        try:
            feed.start(on_snapshot)
            logger.info(f"守护进程已启动，{len(self.active_traders)} 个实例")

            while self._running:
                time.sleep(10)

                # 定时报告
                now = datetime.now()
                if (now - last_report_time).total_seconds() >= 3600:  # 每小时
                    self._hourly_check(now)
                    last_report_time = now

        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            feed.stop()
            self.stop_all()
            self.store.close()
            logger.info("守护进程已停止")

    def _hourly_check(self, now: datetime):
        """每小时的检查和报告。"""
        # 日报告（收盘后）
        daily_time = self.daemon_cfg.get("daily_report_time", "15:30")
        if now.strftime("%H:%M") == daily_time[:5]:
            self.generate_daily_report()

        # 周报告（周五收盘后）
        if now.weekday() == 4 and now.strftime("%H:%M") == daily_time[:5]:
            self.generate_weekly_report()

    # ── 报告 ──────────────────────────────────────────────────

    def generate_daily_report(self):
        """生成每日汇总报告。"""
        entries = []
        for iid in self.active_traders:
            inst = self.store.load_instance(iid)
            logs = inst.get("daily_log", [])
            today_logs = [l for l in logs
                          if l.get("date") == datetime.now().strftime("%Y-%m-%d")]
            if today_logs:
                e = today_logs[-1]
                entries.append({
                    "instance_id": iid,
                    "strategy": inst.get("strategy", ""),
                    "close": e.get("close_price", 0),
                    "cash": e.get("cash", 0),
                    "signal": e.get("signal", ""),
                    "action": e.get("action", "hold"),
                })

        report = {
            "type": "daily",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "active_instances": len(self.active_traders),
            "entries": entries,
        }

        path = os.path.join(_REPORT_DIR, f"daily_{datetime.now().strftime('%Y%m%d')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"日报已生成: {path}")

        return report

    def generate_weekly_report(self):
        """生成周度汇总报告。"""
        entries = []
        for iid in self.active_traders:
            inst = self.store.load_instance(iid)
            logs = inst.get("daily_log", [])
            if not logs:
                continue
            start_val = logs[0].get("cash", inst.get("initial_cash", 100000))
            end_val = logs[-1].get("total_value", start_val)
            ret = (end_val / start_val - 1)
            guard = inst.get("guard_state", {})
            entries.append({
                "instance_id": iid,
                "strategy": inst.get("strategy", ""),
                "week_return": ret,
                "trading_days": len(logs),
                "guard_blown": guard.get("blown", 0),
                "signals": len([l for l in logs if l.get("signal")]),
            })

        report = {
            "type": "weekly",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "active_instances": len(self.active_traders),
            "entries": sorted(entries, key=lambda e: e["week_return"], reverse=True),
        }

        path = os.path.join(_REPORT_DIR, f"weekly_{datetime.now().strftime('%Y%m%d')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"周报已生成: {path}")

        return report

    def get_status(self) -> dict:
        """返回守护进程当前状态。"""
        return {
            "active_instances": len(self.active_traders),
            "instance_ids": list(self.active_traders.keys()),
            "running": self._running,
        }

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _find_strategy(name: str):
        """从 ALL_STRATEGIES 中查找策略类。"""
        from backtest.strategy_market import ALL_STRATEGIES
        try:
            info = ALL_STRATEGIES[name]
            return info["class"]
        except KeyError:
            return None


# ── 命令行 ──────────────────────────────────────────────────
# python live/paper_daemon.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    daemon = PaperDaemon()
    stats = daemon.store.stats()
    print(f"PaperTradeStore: {stats['total_instances']} 实例, {stats['running']} 运行中")

    if stats["running"] > 0:
        print("有待恢复实例。正在进行恢复...")
        daemon.resume_all()
        print(f"恢复后活跃实例: {len(daemon.active_traders)}")

    lb = daemon.store.get_leaderboard(5)
    if not lb.empty:
        print(f"\n排行榜 Top 5:")
        for _, row in lb.iterrows():
            print(f"  #{row.get('id')} {row.get('strategy'):20s} "
                  f"{row.get('total_return', 0):+.2%}")
