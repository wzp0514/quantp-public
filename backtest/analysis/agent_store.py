"""
Agent决策日志持久化 — SQLite存储，支持历史查询和批量评估。

用法
--------
>>> from backtest.analysis.agent_store import AgentStore
>>> store = AgentStore()
>>> store.save(symbol="沪深300", action="buy", confidence=0.8,
...            position_pct=0.2, reasoning="...", mode="llm_langgraph",
...            agent_reports={...})
>>> history = store.query(days=30)
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from config.log import get_logger

logger = get_logger("agent_store")

_DEFAULT_PATH = "data/vault/vault_data/agent_decisions.db"


class AgentStore:
    """
    Agent决策日志存储（SQLite）。

    线程安全，自动建表，支持按日期/标的/模式查询。
    """

    def __init__(self, db_path: str = ""):
        self.db_path = str(Path(db_path or _DEFAULT_PATH))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT DEFAULT '',
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    position_pct REAL DEFAULT 0,
                    reasoning TEXT DEFAULT '',
                    mode TEXT DEFAULT 'rule_based',
                    risk_flags TEXT DEFAULT '[]',
                    agent_reports TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_created
                ON decisions(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol
                ON decisions(symbol, created_at)
            """)
            conn.commit()
            conn.close()

    def save(self, symbol: str, action: str, confidence: float,
             position_pct: float = 0, reasoning: str = "",
             mode: str = "rule_based", risk_flags: list = None,
             agent_reports: dict = None, timestamp: str = "") -> int:
        """
        保存一条决策记录。

        返回
        -------
        int: 插入的记录 ID
        """
        created_at = datetime.now().isoformat()
        ts = timestamp or created_at

        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT INTO decisions
                   (timestamp, symbol, action, confidence, position_pct,
                    reasoning, mode, risk_flags, agent_reports, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts,
                    symbol,
                    action,
                    confidence,
                    position_pct,
                    reasoning,
                    mode,
                    json.dumps(risk_flags or [], ensure_ascii=False),
                    json.dumps(agent_reports or {}, ensure_ascii=False),
                    created_at,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            conn.close()

        logger.debug(f"决策已保存: id={row_id}, {symbol} {action} conf={confidence:.2f}")
        return row_id

    def save_from_result(self, result: dict, symbol: str = "") -> int:
        """从 AgentDecision.decide() 返回结果中保存"""
        return self.save(
            symbol=symbol,
            action=result.get("action", "hold"),
            confidence=result.get("confidence", 0.5),
            position_pct=result.get("position", 0),
            reasoning=result.get("reasoning", ""),
            mode=result.get("mode", "rule_based"),
            risk_flags=result.get("risk_flags", []),
            agent_reports=result.get("agents", {}),
            timestamp=result.get("timestamp", ""),
        )

    def query(self, days: int = 30, symbol: str = "",
              mode: str = "", limit: int = 100) -> list[dict]:
        """
        查询历史决策。

        参数
        ----------
        days : int
            最近多少天
        symbol : str
            按标的筛选（空=全部）
        mode : str
            按模式筛选（空=全部）
        limit : int
            最大返回条数

        返回
        -------
        list[dict]: 决策记录列表
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        sql = "SELECT * FROM decisions WHERE created_at >= ?"
        params: list = [cutoff]

        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if mode:
            sql += " AND mode = ?"
            params.append(mode)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "symbol": r[2],
                "action": r[3],
                "confidence": r[4],
                "position_pct": r[5],
                "reasoning": r[6],
                "mode": r[7],
                "risk_flags": json.loads(r[8]) if r[8] else [],
                "agent_reports": json.loads(r[9]) if r[9] else {},
                "created_at": r[10],
            }
            for r in rows
        ]

    def stats(self, days: int = 30) -> dict:
        """
        决策统计。

        返回
        -------
        dict: total_decisions, action_distribution, mode_distribution, avg_confidence
        """
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        total = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE created_at >= ?", [cutoff]
        ).fetchone()[0]

        actions = {}
        for row in conn.execute(
            "SELECT action, COUNT(*) FROM decisions WHERE created_at >= ? GROUP BY action",
            [cutoff],
        ).fetchall():
            actions[row[0]] = row[1]

        modes = {}
        for row in conn.execute(
            "SELECT mode, COUNT(*) FROM decisions WHERE created_at >= ? GROUP BY mode",
            [cutoff],
        ).fetchall():
            modes[row[0]] = row[1]

        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM decisions WHERE created_at >= ?", [cutoff]
        ).fetchone()[0] or 0

        conn.close()

        return {
            "total_decisions": total,
            "action_distribution": actions,
            "mode_distribution": modes,
            "avg_confidence": round(float(avg_conf), 4),
            "days": days,
        }
