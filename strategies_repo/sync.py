"""
策略同步器 — 跟踪源仓库更新，增量同步策略

原理：
  1. 每次导入时记录源仓库的 commit hash
  2. 定期检查源仓库是否有新提交
  3. 有更新时只重新导入变更的文件

源仓库注册表: strategies_repo/.sources.json

用法
--------
>>> from strategies_repo.sync import StrategySync
>>> sync = StrategySync()
>>> sync.check_updates()   # 检查哪些源有更新
>>> sync.sync_all()        # 同步所有有更新的源
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
REPO_ROOT = Path(__file__).resolve().parent
SOURCES_FILE = REPO_ROOT / ".sources.json"

# 已注册的外部源
from config.log import get_logger

logger = get_logger("sync")

REGISTERED_SOURCES = {
    "vnpy": {
        "name": "vnpy",
        "url": "https://github.com/vnpy/vnpy_ctastrategy.git",
        "type": "github",
        "last_commit": None,
        "last_sync": None,
        "strategies_count": 0,
    },
    "gitee-vnpy": {
        "name": "gitee-vnpy",
        "url": "https://gitee.com/vnpy/vnpy.git",
        "type": "github",
        "last_commit": None,
        "last_sync": None,
        "strategies_count": 0,
    },
    "freqtrade-strategies": {
        "name": "freqtrade-strategies",
        "url": "https://github.com/freqtrade/freqtrade-strategies.git",
        "type": "github",
        "last_commit": None,
        "last_sync": None,
        "strategies_count": 0,
    },
    "tradingview": {
        "name": "tradingview",
        "url": "https://www.tradingview.com/scripts/",
        "type": "web",
        "last_sync": None,
        "strategies_count": 0,
    },
}


class StrategySync:
    """策略同步管理器"""

    def __init__(self):
        self.sources = self._load_sources()

    # ============================================================
    # 检查更新
    # ============================================================

    def check_updates(self) -> list[dict]:
        """
        检查所有已注册源是否有更新。

        返回有更新的源列表。
        """
        updated = []
        for name, source in self.sources.items():
            if source["type"] == "github":
                new_commit = self._get_remote_commit(source["url"])
                old_commit = source.get("last_commit")

                if new_commit and new_commit != old_commit:
                    source["_new_commit"] = new_commit
                    source["_has_update"] = True
                    updated.append(source)
                    logger.info(f"[{name}] 有更新: {old_commit[:8] if old_commit else '首次'} → {new_commit[:8]}")
                else:
                    source["_has_update"] = False
                    logger.info(f"[{name}] 已是最新")

            elif source["type"] == "web":
                # Web 源：按时间判断（上次同步超过 7 天就检查）
                last = source.get("last_sync")
                if not last or (datetime.now() - datetime.fromisoformat(last)).days >= 7:
                    source["_has_update"] = True
                    updated.append(source)
                    logger.info(f"[{name}] 距上次同步超过 7 天，建议检查")

        self._save_sources()
        return updated

    def check_single(self, name: str) -> bool:
        """检查单个源是否有更新"""
        updates = self.check_updates()
        return any(s["name"] == name for s in updates)

    # ============================================================
    # 同步
    # ============================================================

    def sync_all(self) -> dict:
        """同步所有有更新的源"""
        from strategies_repo.importer import import_from_vnpy, import_from_gitee_vnpy, import_from_freqtrade
        from strategies_repo.repo import StrategyRepo

        updates = self.check_updates()
        if not updates:
            logger.info("所有源都是最新的")
            return {}

        repo = StrategyRepo()
        results = {}
        for source in updates:
            name = source["name"]
            logger.info(f"同步: {name}")

            if name == "vnpy":
                n = import_from_vnpy(repo)
                results[name] = n
                self._mark_synced(name, source.get("_new_commit", ""), n)
            elif name == "gitee-vnpy":
                n = import_from_gitee_vnpy(repo)
                results[name] = n
                self._mark_synced(name, source.get("_new_commit", ""), n)
            elif name == "freqtrade-strategies":
                n = import_from_freqtrade(repo)
                results[name] = n
                self._mark_synced(name, source.get("_new_commit", ""), n)
                self._mark_synced(name, "", n)
            elif name == "tradingview":
                from strategies_repo.crawler import crawl_tradingview_top
                n = crawl_tradingview_top(top_n=30, repo=repo)
                results[name] = n
                self._mark_synced(name, "", n)

        return results

    def sync_source(self, name: str) -> int:
        """同步单个源"""
        updates = self.check_updates()
        for s in updates:
            if s["name"] == name:
                results = self.sync_all()
                return results.get(name, 0)
        logger.info(f"[{name}] 无更新")
        return 0

    # ============================================================
    # 状态查询
    # ============================================================

    def status(self) -> str:
        """返回可读的状态文本"""
        lines = ["=" * 60, "策略同步状态", "=" * 60]
        for name, source in self.sources.items():
            last_sync = source.get("last_sync") or "从未"
            commit = (source.get("last_commit") or "未知")[:8]
            count = source.get("strategies_count", 0)
            lines.append(f"  {name:<25} 策略:{count:>4}  同步:{last_sync}  commit:{commit}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ============================================================
    # 内部
    # ============================================================

    def _load_sources(self) -> dict:
        if SOURCES_FILE.exists():
            with open(SOURCES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        # 首次：用默认值
        sources = REGISTERED_SOURCES.copy()
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)
        return sources

    def _save_sources(self):
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.sources, f, indent=2, ensure_ascii=False)

    def _get_remote_commit(self, url: str) -> Optional[str]:
        """获取远程仓库的最新 commit hash（不 clone）"""
        try:
            r = subprocess.run(
                ["git", "ls-remote", url, "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout.split()[0]
        except Exception as e:
            logger.warning(f"获取远程 commit 失败: {url} — {e}")
        return None

    def _mark_synced(self, name: str, commit: str, count: int):
        """标记源已同步"""
        if name in self.sources:
            self.sources[name]["last_commit"] = commit
            self.sources[name]["last_sync"] = datetime.now().isoformat()
            self.sources[name]["strategies_count"] = count
            self._save_sources()


# ============================================================
# 命令行
# ============================================================
# python strategies_repo/sync.py

if __name__ == "__main__":
    sync = StrategySync()
    print(sync.status())
    print("\n检查更新...")
    updates = sync.check_updates()
    if updates:
        print(f"有 {len(updates)} 个源需要更新")
        for s in updates:
            print(f"  - {s['name']}")
        ans = input("立即同步？[y/N] ").strip().lower()
        if ans == "y":
            results = sync.sync_all()
            print(f"同步完成: {results}")
    else:
        print("所有源都是最新的")
