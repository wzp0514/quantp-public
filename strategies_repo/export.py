"""
策略导出 — 将策略导出为独立 .py 文件。

导出的文件可直接 import：
  >>> from exported.ma_cross_v1 import strategy, params, last_result

用法
--------
>>> from strategies_repo.export import export_strategy
>>> export_strategy("布林带回归")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
EXPORT_DIR = Path(__file__).resolve().parent.parent / "exported"

from config.log import get_logger
logger = get_logger("export")
def export_strategy(name: str, include_results: bool = True) -> str:
    """
    导出单个策略为独立 .py 文件。

    参数
    ----------
    name : str
        策略名称
    include_results : bool
        是否包含回测结果注释

    返回
    -------
    str : 导出文件的路径
    """
    from strategies_repo.repo import StrategyRepo
    repo = StrategyRepo()
    s = repo.get(name)
    if not s:
        logger.error(f"策略不存在: {name}")
        return ""

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace(" ", "_")

    lines = [
        f"# 导出策略: {name}",
        f"# 来源: {s.get('source', 'unknown')}",
        f"# 导出时间: {datetime.now().isoformat()}",
        f"# 类型: {s.get('type', s.get('source', ''))}",
        "",
    ]

    # 默认参数
    params = s.get("params", {})
    if params:
        lines.append(f"params = {json.dumps(params, ensure_ascii=False)}")
        lines.append("")

    # 回测结果
    result = s.get("result", {})
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    if include_results and metrics:
        lines.append(f"last_result = {json.dumps(metrics, ensure_ascii=False, default=str)}")
        lines.append("")

    # 策略代码
    code = s.get("code", "")
    if code:
        lines.append(code)

    path = EXPORT_DIR / f"{safe_name}.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"策略已导出: {path}")
    return str(path)


def export_all() -> list[str]:
    """导出全部策略"""
    from strategies_repo.repo import StrategyRepo
    repo = StrategyRepo()
    paths = []
    for s in repo.list():
        path = export_strategy(s["name"])
        if path:
            paths.append(path)
    logger.info(f"全部导出完成: {len(paths)} 个文件 → {EXPORT_DIR}")
    return paths


# ============================================================
# 命令行
# ============================================================
# python strategies_repo/export.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 50)
    print("策略导出")
    print("=" * 50)

    paths = export_all()
    print(f"\n导出完成: {len(paths)} 个文件 → {EXPORT_DIR}")
