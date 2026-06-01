"""
纸上交易持久化存储 — SQLite

复用 BacktestStore 模式（sqlite3 + row_factory），存储所有纸上交易状态。
跨进程重启后可恢复。

数据库位置: vault_data/paper_trade.db
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

from config.log import get_logger

logger = get_logger("paper_store")

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "vault_data", "paper_trade.db")


def _get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db():
    conn = _get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS paper_instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy TEXT NOT NULL,
        symbol TEXT NOT NULL DEFAULT '',
        initial_cash REAL NOT NULL DEFAULT 100000.0,
        params TEXT DEFAULT '{}',
        status TEXT DEFAULT 'running',
        started_at TEXT NOT NULL,
        stopped_at TEXT,
        note TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS paper_daily_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        close_price REAL,
        cash REAL NOT NULL,
        position_size REAL DEFAULT 0,
        position_cost REAL DEFAULT 0,
        total_value REAL,
        signal TEXT DEFAULT '',
        action TEXT DEFAULT 'hold',
        note TEXT DEFAULT '',
        FOREIGN KEY (instance_id) REFERENCES paper_instances(id)
    );
    CREATE INDEX IF NOT EXISTS idx_daily_instance ON paper_daily_log(instance_id);

    CREATE TABLE IF NOT EXISTS paper_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL,
        local_id INTEGER NOT NULL,
        symbol TEXT NOT NULL DEFAULT 'PAPER',
        side TEXT NOT NULL,
        size INTEGER NOT NULL,
        filled_size INTEGER DEFAULT 0,
        price REAL,
        status TEXT NOT NULL DEFAULT 'created',
        created_at TEXT,
        filled_at TEXT,
        note TEXT DEFAULT '',
        FOREIGN KEY (instance_id) REFERENCES paper_instances(id)
    );
    CREATE INDEX IF NOT EXISTS idx_orders_instance ON paper_orders(instance_id);

    CREATE TABLE IF NOT EXISTS paper_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        size INTEGER NOT NULL,
        avg_cost REAL NOT NULL,
        total_cost REAL NOT NULL,
        realized_pnl REAL DEFAULT 0,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (instance_id) REFERENCES paper_instances(id)
    );

    CREATE TABLE IF NOT EXISTS paper_guard_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL UNIQUE,
        blown INTEGER DEFAULT 0,
        blown_reason TEXT DEFAULT '',
        total_signals INTEGER DEFAULT 0,
        total_trades INTEGER DEFAULT 0,
        consecutive_losses INTEGER DEFAULT 0,
        max_drawdown REAL DEFAULT 0.0,
        peak_equity REAL DEFAULT 0.0,
        total_pnl REAL DEFAULT 0.0,
        data TEXT DEFAULT '{}',
        updated_at TEXT NOT NULL,
        FOREIGN KEY (instance_id) REFERENCES paper_instances(id)
    );

    CREATE TABLE IF NOT EXISTS paper_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instance_id INTEGER NOT NULL,
        report_type TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        content TEXT NOT NULL,
        FOREIGN KEY (instance_id) REFERENCES paper_instances(id)
    );
    """)
    conn.commit()
    return conn


class PaperTradeStore:
    """纸上交易 SQLite 持久化存储。

    用法
    --------
    >>> store = PaperTradeStore()
    >>> iid = store.create_instance("MaCrossStrategy", "000300", 100000, {"fast": 5})
    >>> store.save_daily_log(iid, [{"date": "2025-01-01", ...}])
    >>> inst = store.load_instance(iid)
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        _init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _get_db()
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 实例管理 ─────────────────────────────────────────────

    def create_instance(
        self,
        strategy: str,
        symbol: str = "",
        initial_cash: float = 100000.0,
        params: dict = None,
    ) -> int:
        """创建新的纸上交易实例，返回 instance_id。"""
        c = self.conn.execute(
            """INSERT INTO paper_instances (strategy, symbol, initial_cash, params, status, started_at)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (strategy, symbol, initial_cash, json.dumps(params or {}),
             datetime.now().isoformat())
        )
        self.conn.commit()
        return c.lastrowid

    def update_instance(self, instance_id: int, **kwargs):
        """更新实例字段。"""
        if not kwargs:
            return
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [instance_id]
        self.conn.execute(f"UPDATE paper_instances SET {sets} WHERE id=?", vals)
        self.conn.commit()

    def close_instance(self, instance_id: int, final_status: str = "stopped"):
        """关闭实例。"""
        self.conn.execute(
            "UPDATE paper_instances SET status=?, stopped_at=? WHERE id=?",
            (final_status, datetime.now().isoformat(), instance_id)
        )
        self.conn.commit()

    def load_instance(self, instance_id: int) -> dict:
        """加载完整实例状态。"""
        row = self.conn.execute(
            "SELECT * FROM paper_instances WHERE id=?", (instance_id,)
        ).fetchone()
        if not row:
            return {}
        inst = dict(row)
        inst["params"] = json.loads(inst.get("params", "{}"))
        inst["daily_log"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM paper_daily_log WHERE instance_id=? ORDER BY date", (instance_id,)
        ).fetchall()]
        inst["orders"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM paper_orders WHERE instance_id=?", (instance_id,)
        ).fetchall()]
        inst["positions"] = [dict(r) for r in self.conn.execute(
            "SELECT * FROM paper_positions WHERE instance_id=?", (instance_id,)
        ).fetchall()]
        guard = self.conn.execute(
            "SELECT * FROM paper_guard_state WHERE instance_id=?", (instance_id,)
        ).fetchone()
        inst["guard_state"] = dict(guard) if guard else {}
        return inst

    def load_running_instances(self) -> list[dict]:
        """获取所有 running 状态的实例（用于崩溃恢复）。"""
        rows = self.conn.execute(
            "SELECT id, strategy, symbol, initial_cash, params, started_at "
            "FROM paper_instances WHERE status='running'"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_instances(self, limit: int = 50) -> list[dict]:
        """列出最近的实例。"""
        rows = self.conn.execute(
            "SELECT * FROM paper_instances ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 日志 ─────────────────────────────────────────────────

    def save_daily_log(self, instance_id: int, entries: list[dict]):
        """批量写入每日快照。"""
        for e in entries:
            self.conn.execute(
                """INSERT INTO paper_daily_log
                   (instance_id, date, close_price, cash, position_size,
                    position_cost, total_value, signal, action, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (instance_id,
                 str(e.get("date", "")),
                 e.get("close", e.get("close_price", 0)),
                 e.get("cash", 0),
                 e.get("position_size", 0),
                 e.get("position_cost", 0),
                 e.get("cash", 0) + e.get("position_size", 0) * e.get("close", 0),
                 e.get("signal", ""),
                 e.get("action", "hold"),
                 e.get("note", ""))
            )
        self.conn.commit()

    def get_daily_log(self, instance_id: int) -> list[dict]:
        """获取指定实例的全部每日日志。"""
        rows = self.conn.execute(
            "SELECT * FROM paper_daily_log WHERE instance_id=? ORDER BY date",
            (instance_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 订单 ─────────────────────────────────────────────────

    def save_order(self, instance_id: int, order: dict):
        """新增或更新订单。"""
        existing = self.conn.execute(
            "SELECT id FROM paper_orders WHERE instance_id=? AND local_id=?",
            (instance_id, order.get("id", 0))
        ).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE paper_orders SET status=?, filled_size=?, price=?, filled_at=?
                   WHERE instance_id=? AND local_id=?""",
                (order.get("status", ""), order.get("filled_size", order.get("size", 0)),
                 order.get("price", 0), order.get("filled_at", ""),
                 instance_id, order.get("id", 0))
            )
        else:
            self.conn.execute(
                """INSERT INTO paper_orders
                   (instance_id, local_id, symbol, side, size, filled_size, price, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (instance_id, order.get("id", 0), order.get("symbol", "PAPER"),
                 order.get("side", ""), order.get("size", 0),
                 order.get("filled_size", 0), order.get("price", 0),
                 order.get("status", "created"), datetime.now().isoformat())
            )
        self.conn.commit()

    # ── 持仓 ─────────────────────────────────────────────────

    def save_position(self, instance_id: int, positions: dict):
        """保存持仓快照。positions = {symbol: {size, avg_cost, total_cost, ...}}"""
        self.conn.execute(
            "DELETE FROM paper_positions WHERE instance_id=?", (instance_id,)
        )
        for sym, pos in positions.items():
            self.conn.execute(
                """INSERT INTO paper_positions
                   (instance_id, symbol, size, avg_cost, total_cost, realized_pnl, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (instance_id, sym,
                 pos.get("size", 0), pos.get("avg_cost", 0),
                 pos.get("total_cost", pos.get("avg_cost", 0) * pos.get("size", 0)),
                 pos.get("realized_pnl", 0), datetime.now().isoformat())
            )
        self.conn.commit()

    # ── 熔断状态 ─────────────────────────────────────────────

    def save_guard_state(self, instance_id: int, guard):
        """持久化 StrategyGuard 状态。"""
        metrics = guard.metrics
        self.conn.execute(
            """INSERT OR REPLACE INTO paper_guard_state
               (instance_id, blown, blown_reason, total_signals, total_trades,
                consecutive_losses, max_drawdown, peak_equity, total_pnl, data, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (instance_id,
             1 if guard.is_blown() else 0,
             guard.reason or "",
             metrics.total_signals, metrics.total_trades,
             metrics.consecutive_losses, metrics.max_drawdown,
             metrics.peak_equity, metrics.total_pnl,
             json.dumps({}), datetime.now().isoformat())
        )
        self.conn.commit()

    # ── 报告 ─────────────────────────────────────────────────

    def save_report(self, instance_id: int, report_type: str, content: dict):
        """保存定期报告。"""
        self.conn.execute(
            "INSERT INTO paper_reports (instance_id, report_type, generated_at, content) "
            "VALUES (?, ?, ?, ?)",
            (instance_id, report_type, datetime.now().isoformat(), json.dumps(content))
        )
        self.conn.commit()

    def get_reports(self, instance_id: int, report_type: str = "") -> list[dict]:
        """获取指定实例的报告。"""
        if report_type:
            rows = self.conn.execute(
                "SELECT * FROM paper_reports WHERE instance_id=? AND report_type=? "
                "ORDER BY generated_at DESC",
                (instance_id, report_type)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM paper_reports WHERE instance_id=? ORDER BY generated_at DESC",
                (instance_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 排行榜（多实例对比）─────────────────────────────────

    def get_leaderboard(self, limit: int = 10) -> pd.DataFrame:
        """跨实例收益排行。"""
        rows = self.conn.execute("""
            SELECT pi.id, pi.strategy, pi.symbol, pi.initial_cash, pi.status,
                   COALESCE(pdl.total_value, pi.initial_cash) as final_value,
                   (COALESCE(pdl.total_value, pi.initial_cash) / pi.initial_cash - 1) as total_return,
                   pgs.max_drawdown, pgs.consecutive_losses, pgs.blown
            FROM paper_instances pi
            LEFT JOIN (
                SELECT instance_id, total_value, MAX(date) as last_date
                FROM paper_daily_log GROUP BY instance_id
            ) pdl ON pdl.instance_id = pi.id
            LEFT JOIN paper_guard_state pgs ON pgs.instance_id = pi.id
            ORDER BY total_return DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    # ── 统计 ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """数据库概览统计。"""
        total = self.conn.execute("SELECT COUNT(*) FROM paper_instances").fetchone()[0]
        running = self.conn.execute(
            "SELECT COUNT(*) FROM paper_instances WHERE status='running'"
        ).fetchone()[0]
        logs = self.conn.execute("SELECT COUNT(*) FROM paper_daily_log").fetchone()[0]
        return {"total_instances": total, "running": running, "total_daily_logs": logs}


# ── 命令行测试 ──────────────────────────────────────────────
# python live/execution/paper_store.py

if __name__ == "__main__":
    store = PaperTradeStore()
    s = store.stats()
    print(f"PaperTradeStore: {s['total_instances']} 实例, "
          f"{s['running']} 运行中, {s['total_daily_logs']} 条日志")

    # 列出最近实例
    instances = store.list_instances(10)
    if instances:
        print("\n最近实例:")
        for inst in instances:
            print(f"  #{inst['id']} {inst['strategy']} [{inst['status']}] "
                  f"{inst['started_at'][:19]}")
    else:
        print("\n尚无实例 — 创建测试实例...")
        iid = store.create_instance("TestStrategy", "000300", 100000, {"fast": 5, "slow": 20})
        print(f"已创建实例 #{iid}")
        store.save_daily_log(iid, [{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "close": 3500.0, "cash": 100000, "position_size": 0,
            "position_cost": 0, "signal": "", "action": "hold", "note": "test"
        }])
        store.close_instance(iid, "completed")
        print(f"实例 #{iid} 已完成")

    store.close()
