"""
主动推送+自动复盘系统（轻量版）。

盘中自动扫描→推送通知→盘后复盘进化。
轻量实现：定时检查 + notifier 推送，不引入复杂调度框架。

用法
--------
>>> from live.auto_review import AutoReview
>>> ar = AutoReview()
>>> ar.scan_and_notify()  # 扫描关键指标并推送
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config.log import get_logger

logger = get_logger("auto_review")

_REVIEW_LOG = Path(__file__).resolve().parent.parent / "data" / "vault" / "vault_data" / "auto_review_log.json"


class AutoReview:
    """
    自动复盘系统（轻量版）。

    功能：
    1. 盘后自动运行回测对比
    2. 关键指标变化检测
    3. 通知推送（通过 notifier 通道）
    4. 复盘记录持久化
    """

    def __init__(self):
        self._log = self._load_log()

    def scan_and_notify(self, symbol: str = "沪深300") -> dict:
        """
        扫描关键指标并推送。

        返回 dict: status, alerts, summary
        """
        alerts = []

        # 1. 数据源健康检查
        try:
            from data.fetchers.fallback import get_source_health
            health = get_source_health()
            for src, info in health.items():
                if info.get("status") == "down":
                    alerts.append(f"数据源 {src} 离线")
        except Exception as e:
            logger.warning(f"健康检查失败: {e}")

        # 2. 近期行情快照
        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            df = fetch_index_daily_safe(symbol,
                                        (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                                        datetime.now().strftime("%Y%m%d"))
            if not df.empty:
                latest = df.iloc[-1]
                prev5 = df.iloc[-6] if len(df) >= 6 else df.iloc[0]
                change_5d = float(latest["close"] / prev5["close"] - 1) * 100
                if abs(change_5d) > 5:
                    alerts.append(f"{symbol} 5日涨跌幅 {change_5d:+.2f}%（异常波动）")
        except Exception as e:
            logger.warning(f"行情快照失败: {e}")

        # 3. 记录结果
        record = {
            "time": datetime.now().isoformat(),
            "symbol": symbol,
            "alerts": alerts,
            "alert_count": len(alerts),
        }
        self._log.append(record)

        # 保留最近 90 天
        self._log = self._log[-90:]
        self._save_log()

        # 4. 如果有告警，尝试推送
        if alerts and self._try_notify(alerts):
            record["notified"] = True

        logger.info(f"自动复盘: {len(alerts)} 条告警")
        return {"status": "ok", "alerts": alerts, "alert_count": len(alerts), "history_days": len(self._log)}

    def review_summary(self, days: int = 7) -> dict:
        """复盘摘要（最近 N 天）"""
        recent = self._log[-days:]
        return {
            "days": len(recent),
            "total_alerts": sum(r.get("alert_count", 0) for r in recent),
            "last_scan": recent[-1]["time"] if recent else None,
        }

    def scan_with_guardian(self) -> dict:
        """
        整合 Guardian 的完整扫描：数据健康+策略漂移+持仓偏差。
        """
        alerts = []
        details = {}

        # Guardian 检查
        try:
            from live.monitor.guardian import Guardian
            g = Guardian()
            health = g.check_data_health()
            details["data_health"] = health
            if not health.get("healthy"):
                alerts.append(f"数据健康异常: {health.get('issues', [])}")
            drift = g.check_strategy_drift()
            details["strategy_drift"] = drift
            if drift.get("drift_detected"):
                alerts.append(f"策略漂移: {drift.get('details', '')}")
        except ImportError:
            logger.debug("Guardian 不可用")
        except Exception as e:
            logger.warning(f"Guardian 检查失败: {e}")

        # Notifier 推送
        if alerts:
            self._try_notify(alerts)

        record = {
            "time": datetime.now().isoformat(),
            "source": "guardian",
            "alerts": alerts,
            "alert_count": len(alerts),
        }
        self._log.append(record)
        self._log = self._log[-90:]
        self._save_log()

        return {"status": "ok", "alerts": alerts, "details": details}

    def _try_notify(self, alerts: list[str]) -> bool:
        """尝试推送通知"""
        try:
            from live.gateway.notifier import send as notify_send
            msg = "[AutoReview] " + "; ".join(alerts)
            notify_send(msg)
            return True
        except ImportError:
            logger.debug("notifier 不可用，跳过推送")
            return False
        except Exception as e:
            logger.debug(f"推送失败: {e}")
            return False

    def _load_log(self) -> list:
        if _REVIEW_LOG.exists():
            try:
                with open(_REVIEW_LOG, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_log(self):
        _REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_REVIEW_LOG, "w", encoding="utf-8") as f:
            json.dump(self._log, f, indent=2, ensure_ascii=False)
