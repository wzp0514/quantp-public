"""
健康检查报告系统 — 统一汇总所有健康检查项（C8-C9）

HealthCheckReport:
  - 汇总多个检查项的状态码和详情
  - 数据源逐步回溯（10→30→90→365天）
  - 报告"最近数据X天前"而非简单"正常/无数据"

用法
--------
>>> from live.monitor.health_check import HealthCheckReport
>>> report = HealthCheckReport()
>>> report.add_source_check("AkShare", True, {"latency": "0.3s", "rows": 5000})
>>> report.add_source_check("Tushare", False, {"error": "Token过期"})
>>> report.add_guardian_check(True, {"drift_count": 0})
>>> print(report.summary())
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class CheckItem:
    """单条健康检查项"""
    name: str
    category: str        # data_source / guardian / pipeline / strategy
    passed: bool
    status_code: str     # OK / WARN / ERROR / UNKNOWN
    detail: str
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())


class HealthCheckReport:
    """健康检查报告 — 统一汇总，按类别分组"""

    def __init__(self):
        self.items: list[CheckItem] = []

    def add(self, name: str, category: str, passed: bool, detail: str = "",
            status_code: str = ""):
        if not status_code:
            status_code = "OK" if passed else "ERROR"
        self.items.append(CheckItem(
            name=name, category=category, passed=passed,
            status_code=status_code, detail=detail,
        ))

    def add_source_check(self, source_name: str, available: bool,
                         extra: Optional[dict] = None):
        """添加数据源健康检查项"""
        extra = extra or {}
        if available:
            data_age = extra.get("data_age_days")
            if data_age is not None:
                detail = f"可用，最近数据 {data_age} 天前"
                sc = "WARN" if data_age > 7 else "OK"
            else:
                detail = "可用"
                sc = "OK"
        else:
            detail = extra.get("error", "不可用")
            sc = "ERROR"
        self.add(source_name, "data_source", available, detail, sc)

    def add_guardian_check(self, healthy: bool, extra: Optional[dict] = None):
        """添加守护进程健康检查项"""
        extra = extra or {}
        parts = []
        if healthy:
            parts.append("守护进程运行正常")
        else:
            parts.append("守护进程异常")
        if extra.get("drift_count"):
            parts.append(f"策略漂移: {extra['drift_count']}次")
        if extra.get("last_emergency"):
            parts.append(f"上次熔断: {extra['last_emergency']}")
        self.add("Guardian", "guardian", healthy, "; ".join(parts) if parts else "OK",
                 "OK" if healthy else "ERROR")

    def add_pipeline_check(self, track_name: str, stage: str, status: str):
        """添加管线 Track 健康检查项"""
        passed = status in ("READY", "PAPER", "ACTIVE")
        sc = "OK" if status == "READY" else ("WARN" if status in ("PAPER", "VALIDATE") else "ERROR")
        self.add(track_name, "pipeline", passed, f"Stage={stage}, Status={status}", sc)

    def summary(self) -> str:
        """生成可读的健康检查汇总报告"""
        by_cat = {}
        for item in self.items:
            by_cat.setdefault(item.category, []).append(item)

        lines = ["=" * 60, "  系统健康检查报告", "=" * 60]
        lines.append(f"  检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  检查项: {len(self.items)} 项")
        lines.append("")

        passed = sum(1 for i in self.items if i.passed)
        failed = len(self.items) - passed
        overall = "PASS" if failed == 0 else ("WARN" if passed > failed else "FAIL")
        lines.append(f"  通过: {passed}  失败: {failed}  整体: {overall}")
        lines.append("")

        for cat, cat_items in sorted(by_cat.items()):
            lines.append(f"  ── {cat} ──")
            for item in cat_items:
                icon = "[OK]" if item.passed else "[!!]"
                lines.append(f"    {icon} {item.name}: {item.detail}")
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"  结论: {overall} ({passed}/{len(self.items)} 通过)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            "checked_at": datetime.now().isoformat(),
            "total": len(self.items),
            "passed": sum(1 for i in self.items if i.passed),
            "failed": sum(1 for i in self.items if not i.passed),
            "overall": "PASS" if all(i.passed for i in self.items) else "FAIL",
            "items": [
                {
                    "name": i.name,
                    "category": i.category,
                    "passed": i.passed,
                    "status_code": i.status_code,
                    "detail": i.detail,
                }
                for i in self.items
            ],
        }


def data_freshness_check(data_df, name: str = "") -> CheckItem:
    """
    C9: 数据新鲜度回溯检测 — 逐步回溯（10→30→90→365天），
    报告"最近数据X天前"而非简单"无数据"。
    """
    if data_df is None or len(data_df) == 0:
        return CheckItem(
            name=name or "数据源", category="data_source",
            passed=False, status_code="ERROR",
            detail="无数据",
        )

    from datetime import datetime as dt

    latest = data_df["date"].max()
    if hasattr(latest, "date"):
        latest_date = latest.date()
    elif hasattr(latest, "to_pydatetime"):
        latest_date = latest.to_pydatetime().date()
    else:
        latest_date = dt.strptime(str(latest)[:10], "%Y-%m-%d").date()

    days_ago = (date.today() - latest_date).days

    # 逐步回溯判定
    if days_ago <= 10:
        sc, level = "OK", "数据新鲜（≤10天）"
    elif days_ago <= 30:
        sc, level = "OK", f"数据略旧（{days_ago}天前）"
    elif days_ago <= 90:
        sc, level = "WARN", f"数据偏旧（{days_ago}天前，>30天）"
    elif days_ago <= 365:
        sc, level = "ERROR", f"数据陈旧（{days_ago}天前，>90天）"
    else:
        sc, level = "ERROR", f"数据严重过期（{days_ago}天前，>365天）"

    return CheckItem(
        name=name or "数据源", category="data_source",
        passed=(sc == "OK"), status_code=sc,
        detail=f"最近数据 {days_ago} 天前 | {level} | 共{len(data_df)}条",
    )
