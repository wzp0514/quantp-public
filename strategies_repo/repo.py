"""
策略仓库 — 结构化存储、管理所有策略

每个策略 = 一个文件夹，包含：
  meta.yaml      → 元数据（名称/来源/类型/标签/参数/回测指标）
  strategy.py    → Backtrader 策略类（可直接 import 运行）
  result.json    → 最近一次回测结果（缓存，方便快速对比）

用法
--------
>>> from strategies_repo.repo import StrategyRepo
>>> repo = StrategyRepo()
>>> repo.list()           # 列出所有策略
>>> repo.search("均线")   # 搜索
>>> repo.run(name)        # 加载并跑回测
>>> repo.compare(["策略A", "策略B"])  # 对比
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yaml
# 仓库根目录
REPO_ROOT = Path(__file__).resolve().parent

# 分类目录
from config.log import get_logger

logger = get_logger("repo")

CATEGORIES = {
    "market": "market",
    "mined": "mined",
    "custom": "custom",
}

# C14: 策略归档子目录（market 下三级分类）
ARCHIVE_TIERS = {
    "active": "active",        # 达标，可进纸上联赛
    "candidate": "candidate",  # 待调参/待适配
    "archived": "archived",    # 淘汰，保留代码+回测结果+失败标签
}

# C15: 策略标签体系
REJECTION_TAGS = {
    "rejected:weak_sharpe": "夏普比率过低",
    "rejected:high_dd": "最大回撤过高",
    "rejected:low_trades": "交易次数不足",
    "rejected:regime_mismatch": "市态不匹配",
}
ARCHIVED_TAGS = {
    "archived:passed": "已通过验证，归档保留",
}


class StrategyRepo:
    """策略仓库管理器"""

    def __init__(self):
        self.root = REPO_ROOT
        for cat in CATEGORIES:
            (self.root / cat).mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 查询
    # ============================================================

    def list(self, category: str = "") -> list[dict]:
        """
        列出所有策略。

        参数
        ----------
        category : str
            分类筛选。空 = 全部, "market"/"mined"/"custom"

        返回
        -------
        list[dict] : 每个策略的元数据 + 路径
        """
        results = []
        cats = [category] if category in CATEGORIES else CATEGORIES.values()

        for cat in cats:
            cat_dir = self.root / cat
            if not cat_dir.exists():
                continue
            for item in sorted(cat_dir.iterdir()):
                if item.is_dir() and (item / "meta.yaml").exists():
                    meta = self._read_meta(item)
                    if meta:
                        meta["_path"] = str(item)
                        meta["_category"] = cat
                        results.append(meta)

        return results

    def search(self, keyword: str) -> list[dict]:
        """
        搜索策略（名称/来源/类型/标签任意匹配）

        示例
        --------
        >>> repo.search("均线")  → 找所有和均线相关的
        >>> repo.search("vnpy")  → 找所有来自vnpy的
        """
        keyword_lower = keyword.lower()
        results = []
        for s in self.list():
            text = (
                s.get("name", "") + " "
                + s.get("source", "") + " "
                + s.get("type", "") + " "
                + " ".join(s.get("tags", []))
            ).lower()
            if keyword_lower in text:
                results.append(s)
        return results

    def get(self, name: str) -> Optional[dict]:
        """获取单个策略的完整信息"""
        for s in self.list():
            if s["name"] == name:
                return s
        return None

    def stats(self) -> dict:
        """仓库统计"""
        all_s = self.list()
        by_category = {}
        by_type = {}
        for s in all_s:
            cat = s.get("_category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            typ = s.get("type", "unknown")
            by_type[typ] = by_type.get(typ, 0) + 1

        return {
            "total": len(all_s),
            "by_category": by_category,
            "by_type": by_type,
        }

    # ============================================================
    # 导入
    # ============================================================

    def import_from_market(
        self,
        name: str,
        source: str,
        source_url: str,
        stype: str,
        desc: str,
        params: dict,
        strategy_code: str,
        category: str = "market",
        tags: list = None,
    ) -> bool:
        """
        导入一个策略到仓库。

        参数
        ----------
        name : str            策略名（会作为文件夹名）
        source : str          来源（如 "vnpy社区"）
        source_url : str      来源URL
        stype : str           策略类型（趋势跟踪/均值回归/震荡/突破）
        desc : str            一句话描述
        params : dict         默认参数
        strategy_code : str   Python 策略类源码
        category : str        分类目录
        tags : list           标签
        """
        safe_name = name.replace("/", "_").replace("\\", "_")
        strategy_dir = self.root / category / safe_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # meta.yaml
        meta = {
            "name": name,
            "source": source,
            "source_url": source_url,
            "type": stype,
            "tags": tags or [],
            "description": desc,
            "params": params,
            "created": date.today().isoformat(),
            "last_backtest": None,
            "metrics": {
                "annual_return": None,
                "drawdown": None,
                "sharpe": None,
                "total_trades": None,
                "win_rate": None,
            },
        }
        with open(strategy_dir / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)

        # strategy.py
        with open(strategy_dir / "strategy.py", "w", encoding="utf-8") as f:
            f.write(strategy_code)

        # result.json (空模板)
        with open(strategy_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump({}, f)

        logger.info(f"策略已导入: {name} → {category}/{safe_name}")
        return True

    def import_from_miner(self, combo: dict, category: str = "mined") -> bool:
        """
        从策略挖掘结果导入一个策略。

        combo 格式（来自 strategy_miner.py）:
          {"entry": ..., "exit": ..., "params": ..., "score": ..., ...}
        """
        name = f"mined_{combo['entry']}_{combo['exit']}_{datetime.now().strftime('%H%M%S')}"
        desc = combo.get("desc", f"{combo['entry']}+{combo['exit']}")

        # 生成策略类代码
        entry_name = combo["entry"]
        exit_name = combo["exit"]
        params = combo.get("params", {})

        code_lines = [
            "import backtrader as bt",
            "from backtest.engine.bt_runner import BaseStrategy",
            "",
            f"# 挖掘策略: {desc}",
            f"# 入场: {entry_name}, 出场: {exit_name}",
            f"# 参数: {params}",
            "",
            "class MinedStrategy(BaseStrategy):",
            f"    params = {tuple(params.items())}",
            "",
            "    def __init__(self):",
            "        super().__init__()",
            "        # TODO: 填入具体的指标逻辑",
            "        pass",
            "",
            "    def next(self):",
            "        pass",
        ]
        strategy_code = "\n".join(code_lines)

        return self.import_from_market(
            name=name,
            source="策略挖掘",
            source_url="",
            stype="auto",
            desc=desc,
            params=params,
            strategy_code=strategy_code,
            category=category,
            tags=["mined", entry_name, exit_name],
        )

    # ============================================================
    # 运行
    # ============================================================

    def run(
        self,
        name: str,
        df,
        cash: float = 100000.0,
        save_result: bool = True,
    ) -> Optional[dict]:
        """
        加载策略并运行回测。

        参数
        ----------
        name : str    策略名
        df : DataFrame 行情数据
        cash : float  初始资金
        save_result : bool  是否保存结果到仓库

        返回
        -------
        dict : run_backtest 的结果，或 None
        """
        meta = self.get(name)
        if not meta:
            logger.error(f"策略不存在: {name}")
            return None

        strategy_path = Path(meta["_path"]) / "strategy.py"

        # 动态加载策略类
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, strategy_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 找到策略类（模块中第一个 BaseStrategy 子类）
        strategy_class = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and attr.__name__.endswith("Strategy"):
                strategy_class = attr
                break

        if strategy_class is None:
            logger.error(f"策略文件中未找到 Strategy 类: {strategy_path}")
            return None

        # 运行回测
        from backtest.engine.bt_runner import run_backtest
        params = meta.get("params", {})
        result = run_backtest(strategy_class, df, initial_cash=cash, **params)

        # 保存结果
        if save_result:
            self._save_result(name, result)

        return result

    def compare(
        self,
        names: list[str],
        df,
        cash: float = 100000.0,
    ) -> list[dict]:
        """
        对比多个策略（逐个跑，输出排名）。

        返回
        -------
        list[dict] : 按年化收益降序排列
        """
        results = []
        for name in names:
            logger.info(f"运行: {name}")
            r = self.run(name, df, cash, save_result=True)
            if r:
                meta = self.get(name)
                results.append({
                    "name": name,
                    "source": meta.get("source", ""),
                    "type": meta.get("type", ""),
                    "annual_return": r["annual_return"],
                    "drawdown": r["drawdown"],
                    "sharpe": r["sharpe"],
                    "trades": r["total_trades"],
                    "win_rate": r["win_rate"],
                })

        results.sort(key=lambda x: x["annual_return"], reverse=True)
        return results

    # ============================================================
    # 管理
    # ============================================================

    def remove(self, name: str) -> bool:
        """删除策略"""
        meta = self.get(name)
        if not meta:
            logger.error(f"策略不存在: {name}")
            return False
        strategy_dir = Path(meta["_path"])
        shutil.rmtree(strategy_dir)
        logger.info(f"已删除: {name}")
        return True

    # ============================================================
    # 社区工作流: Fork & Publish
    # ============================================================

    def fork(self, name: str, new_name: str = None, category: str = "custom") -> Optional[dict]:
        """
        从仓库中 Fork 一个策略到本地工作区。

        参数
        ----------
        name : str       源策略名
        new_name : str   新策略名（默认原名_fork）
        category : str   目标分类目录（默认 custom）
        """
        meta = self.get(name)
        if not meta:
            logger.error(f"源策略不存在: {name}")
            return None

        src_dir = Path(meta["_path"])
        if new_name is None:
            new_name = f"{name}_fork"

        safe_name = new_name.replace("/", "_").replace("\\", "_")
        dst_dir = self.root / category / safe_name

        if dst_dir.exists():
            base = new_name
            i = 1
            while dst_dir.exists():
                new_name = f"{base}_{i}"
                safe_name = new_name.replace("/", "_").replace("\\", "_")
                dst_dir = self.root / category / safe_name
                i += 1

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / "strategy.py", dst_dir / "strategy.py")
        if (src_dir / "result.json").exists():
            shutil.copy2(src_dir / "result.json", dst_dir / "result.json")
        else:
            (dst_dir / "result.json").write_text("{}", encoding="utf-8")

        meta["name"] = new_name
        meta["forked_from"] = name
        meta["forked_at"] = datetime.now().isoformat()
        meta["last_backtest"] = None
        meta["metrics"] = {k: None for k in meta.get("metrics", {})}
        meta["_category"] = category
        meta["_path"] = str(dst_dir)

        with open(dst_dir / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"Fork 完成: {name} → {category}/{new_name}")
        return meta

    def publish(self, name: str, tier: str = "candidate") -> dict:
        """
        发布策略到 market 分级目录。

        参数
        ----------
        name : str   策略名
        tier : str   目标级别: active / candidate
        """
        meta = self.get(name)
        if not meta:
            return {"success": False, "reason": f"策略不存在: {name}"}

        if tier not in ("active", "candidate"):
            return {"success": False, "reason": f"无效级别: {tier}，可选 active/candidate"}

        metrics = meta.get("metrics", {})
        if not metrics.get("sharpe") or metrics["sharpe"] is None:
            return {"success": False, "reason": "策略未回测，请先运行回测获得 Sharpe"}

        current_cat = meta.get("_category", "")
        if current_cat == "market":
            current_tier = meta.get("archived_tier", "")
            if current_tier == tier:
                return {"success": False, "reason": f"策略已在 market/{tier}，无需重复发布"}

        self._ensure_archive_tiers()
        current_path = Path(meta["_path"])
        safe_name = name.replace("/", "_").replace("\\", "_")
        target_path = self.root / "market" / tier / safe_name

        if target_path.exists() and str(current_path) != str(target_path):
            return {"success": False, "reason": f"目标路径已存在: {target_path}"}

        if str(current_path) != str(target_path):
            shutil.move(str(current_path), str(target_path))

        meta["published"] = True
        meta["published_at"] = datetime.now().isoformat()
        meta["published_tier"] = tier
        meta["archived_tier"] = tier
        meta["_path"] = str(target_path)

        with open(target_path / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"发布完成: {name} → market/{tier}")
        return {"success": True, "tier": tier, "path": str(target_path)}

    # ============================================================
    # C14-C15: 策略归档与分级
    # ============================================================

    def _ensure_archive_tiers(self):
        """确保 market/ 下 active/candidate/archived 子目录存在"""
        for tier in ARCHIVE_TIERS.values():
            (self.root / "market" / tier).mkdir(parents=True, exist_ok=True)

    def classify(self, name: str, tier: str, tags: list[str] = None,
                 reason: str = "") -> bool:
        """
        C14: 将策略移入三级分类目录。

        参数
        ----------
        name : str      策略名
        tier : str      目标级别: active / candidate / archived
        tags : list     标签列表，如 ["rejected:weak_sharpe", "rejected:high_dd"]
        reason : str    分级原因说明
        """
        meta = self.get(name)
        if not meta:
            logger.error(f"策略不存在: {name}")
            return False

        if tier not in ARCHIVE_TIERS:
            logger.error(f"无效级别: {tier}，可选 {list(ARCHIVE_TIERS.keys())}")
            return False

        self._ensure_archive_tiers()
        current_path = Path(meta["_path"])
        safe_name = name.replace("/", "_").replace("\\", "_")
        target_path = self.root / "market" / tier / safe_name

        if target_path.exists():
            logger.warning(f"目标已存在: {target_path}，跳过移动")
            return False

        shutil.move(str(current_path), str(target_path))

        # 更新 meta.yaml 中的归档信息
        meta_file = target_path / "meta.yaml"
        if meta_file.exists():
            meta["archived_tier"] = tier
            meta["archived_tags"] = tags or []
            meta["archived_reason"] = reason
            meta["archived_date"] = date.today().isoformat()
            with open(meta_file, "w", encoding="utf-8") as f:
                yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"策略已分级: {name} → {tier}（{reason or '无说明'}）")
        return True

    def list_by_tier(self, tier: str = "") -> list[dict]:
        """C14: 按归档级别列出策略"""
        self._ensure_archive_tiers()
        results = []
        tiers = [tier] if tier in ARCHIVE_TIERS else ARCHIVE_TIERS.values()
        for t in tiers:
            tier_dir = self.root / "market" / t
            if not tier_dir.exists():
                continue
            for item in sorted(tier_dir.iterdir()):
                if item.is_dir() and (item / "meta.yaml").exists():
                    meta = self._read_meta(item)
                    if meta:
                        meta["_path"] = str(item)
                        meta["_tier"] = t
                        results.append(meta)
        return results

    def tag_strategy(self, name: str, tags: list[str]) -> bool:
        """C15: 为策略添加标签"""
        meta = self.get(name)
        if not meta:
            return False
        existing = set(meta.get("tags", []))
        existing.update(tags)
        meta["tags"] = sorted(existing)
        strategy_dir = Path(meta["_path"])
        with open(strategy_dir / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"标签已更新: {name} → {meta['tags']}")
        return True

    def refresh_results(self, name: str, df, cash: float = 100000.0) -> bool:
        """刷新某个策略的回测结果缓存"""
        r = self.run(name, df, cash, save_result=True)
        return r is not None

    # ============================================================
    # 内部
    # ============================================================

    def _read_meta(self, strategy_dir: Path) -> Optional[dict]:
        try:
            with open(strategy_dir / "meta.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def _save_result(self, name: str, result: dict):
        meta = self.get(name)
        if not meta:
            return
        strategy_dir = Path(meta["_path"])

        # 更新 meta 中的回测指标
        meta["last_backtest"] = date.today().isoformat()
        meta["metrics"] = {
            "annual_return": result.get("annual_return"),
            "drawdown": result.get("drawdown"),
            "sharpe": result.get("sharpe"),
            "total_trades": result.get("total_trades"),
            "win_rate": result.get("win_rate"),
        }
        with open(strategy_dir / "meta.yaml", "w", encoding="utf-8") as f:
            yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)

        # 保存原始结果 JSON
        clean_result = {
            k: v for k, v in result.items()
            if k not in ("cerebro", "strategy", "trades_df", "equity_df")
        }
        with open(strategy_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(clean_result, f, indent=2, ensure_ascii=False, default=str)

    # ============================================================
    # 批量导入 — 把现有策略全部入库
    # ============================================================

    def import_all_builtin(self):
        """一键导入所有内置策略到仓库（首次使用调用一次即可）"""
        from backtest.strategy_market import MARKET_STRATEGIES, BUILTIN_STRATEGIES

        all_s = {**MARKET_STRATEGIES, **BUILTIN_STRATEGIES}
        imported = 0

        for name, info in all_s.items():
            # 检查是否已存在
            safe_name = name.replace("/", "_")
            if (self.root / "market" / safe_name / "meta.yaml").exists():
                continue

            # 获取策略类源码
            cls = info["class"]
            try:
                import inspect
                code = inspect.getsource(cls)
            except Exception:
                code = f"# 无法获取源码: {name}\n# 来源: {info['source']}"

            stype = "unknown"
            if "均线" in name or "交叉" in name:
                stype = "趋势跟踪"
            elif "布林" in name or "回归" in name or "RSI" in name:
                stype = "均值回归"
            elif "动量" in name:
                stype = "趋势跟踪"
            elif "网格" in name:
                stype = "震荡"
            elif "海龟" in name or "通道" in name or "Donchian" in name:
                stype = "突破"
            elif "ATR" in name or "止损" in name:
                stype = "趋势跟踪"

            self.import_from_market(
                name=name,
                source=info.get("source", "quantp"),
                source_url=info.get("url", ""),
                stype=stype,
                desc=info.get("desc", ""),
                params=info.get("params", {}),
                strategy_code=code,
                category="market",
            )
            imported += 1

        logger.info(f"批量导入完成: {imported} 个策略")
        return imported


# ============================================================
# 便捷函数
# ============================================================

def init_repo() -> StrategyRepo:
    """初始化仓库（首次使用）"""
    repo = StrategyRepo()
    existing = len(repo.list())
    if existing == 0:
        logger.info("首次使用，导入所有内置策略...")
        repo.import_all_builtin()
    else:
        logger.info(f"仓库已有 {existing} 个策略")
    return repo


# ============================================================
# 命令行操作
# ============================================================
# python strategies_repo/repo.py

if __name__ == "__main__":
    repo = init_repo()

    print("=" * 60)
    print("策略仓库")
    print("=" * 60)

    stats = repo.stats()
    print(f"总数: {stats['total']}")
    print(f"按分类: {stats['by_category']}")
    print(f"按类型: {stats['by_type']}")

    print("\n全部策略:")
    for s in repo.list():
        m = s.get("metrics", {})
        ret = m.get("annual_return")
        ret_str = f"{ret:.2%}" if ret else "未回测"
        print(f"  [{s['_category']}] {s['name']:<16} | {s['type']:<10} | {s['source']:<20} | {ret_str}")

    print("\n搜索示例: repo.search('均线')")
    results = repo.search("均线")
    for r in results:
        print(f"  - {r['name']}: {r['description']}")
