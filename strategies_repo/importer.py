"""
策略自动导入器 — 从开源仓库批量拉取策略

支持来源：
  - vnpy_ctastrategy: vnpy 官方 CTA 策略仓库（20+ 策略）
  - Freqtrade: freqtrade-strategies 社区仓库（已归档，约 300+ 策略）
  - 更多可扩展

原理：
  git clone --depth=1（只拉最新版本，不拉整个历史），
  扫描指定目录下的 .py 文件，
  解析策略类名和参数，自动入库。

用法
--------
>>> from strategies_repo.importer import import_from_vnpy, import_from_freqtrade
>>> import_from_vnpy()       # 从 vnpy 导入所有策略
>>> import_from_freqtrade()  # 从 Freqtrade 社区导入所有策略
"""

import ast
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from strategies_repo.repo import StrategyRepo
TEMP_DIR = Path(tempfile.gettempdir()) / "quantp_imports"

from config.log import get_logger
logger = get_logger("importer")
# ============================================================
# Python 源码解析（提取类名、参数、描述）
# ============================================================

def parse_strategy_file(filepath: Path) -> Optional[dict]:
    """
    解析一个策略 .py 文件，提取元数据。

    返回
    -------
    dict 或 None: {name, class_name, params, desc, code}
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception:
        return None

    if len(code.strip()) < 50:  # 太短，跳过
        return None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    strategy_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 检查是否继承了 Strategy（或包含 Strategy 的基类）
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)

            is_strategy = any("Strategy" in b or "IStrategy" in b for b in bases)

            if is_strategy or node.name.endswith("Strategy"):
                # 提取 params
                params = {}
                for item in node.body:
                    if isinstance(item, ast.Assign) and len(item.targets) == 1:
                        target = item.targets[0]
                        if isinstance(target, ast.Name) and target.id == "params":
                            # 尝试解析 params 字典
                            if isinstance(item.value, ast.Dict):
                                for key, val in zip(item.value.keys, item.value.values):
                                    key_name = getattr(key, "value", getattr(key, "s", None))
                                    val_value = getattr(val, "value", None)
                                    if key_name and val_value is not None:
                                        params[str(key_name)] = val_value

                # 提取 docstring
                desc = ast.get_docstring(node) or ""

                strategy_classes.append({
                    "class_name": node.name,
                    "params": params,
                    "desc": desc,
                })

    if not strategy_classes:
        return None

    # 取第一个策略类（一个文件通常只有一个主策略类）
    main = strategy_classes[0]
    name = main["class_name"].replace("Strategy", "").replace("_", "")

    return {
        "name": name,
        "class_name": main["class_name"],
        "params": main["params"],
        "desc": main["desc"][:200] if main["desc"] else "",
        "code": code,
    }


# ============================================================
# GitHub 仓库操作
# ============================================================

def _git_clone_shallow(url: str, target_dir: Path) -> bool:
    """浅克隆一个仓库（只拉最新版本）"""
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)

    try:
        subprocess.run(
            ["git", "clone", "--depth=1", url, str(target_dir)],
            capture_output=True, text=True, timeout=120, check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"克隆失败: {e.stderr[:200]}")
        return False
    except Exception as e:
        logger.error(f"克隆异常: {e}")
        return False


def _git_get_commit(repo_dir: Path) -> str:
    """获取仓库当前 commit hash"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_dir), timeout=10,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _verify_repo_url(url: str) -> bool:
    """检查远程仓库 URL 是否可访问（不 clone）"""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "--heads", url],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


# ============================================================
# 具体来源导入
# ============================================================

def import_from_vnpy(repo: StrategyRepo = None) -> int:
    """
    从 vnpy 官方仓库导入策略。

    vnpy_ctastrategy/strategies/ 目录下约有 20+ 策略。
    （历史路径 vnpy/app/cta_strategy/strategies/，v2.4.0 后移至 vnpy_ctastrategy/）
    """
    if repo is None:
        repo = StrategyRepo()

    logger.info("从 vnpy 导入策略...")
    url = "https://github.com/vnpy/vnpy_ctastrategy.git"
    target = TEMP_DIR / "vnpy_ctastrategy"

    if not _git_clone_shallow(url, target):
        return 0

    commit = _git_get_commit(target)

    # vnpy_ctastrategy 仓库的策略在根目录或 strategies/ 子目录下
    strategy_dirs = [target] + list(target.glob("**/strategies/"))
    imported = 0

    for strat_dir in strategy_dirs:
        for py_file in strat_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            info = parse_strategy_file(py_file)
            if info is None:
                continue

            # 生成策略名
            name = f"vnpy_{info['name']}"
            safe_name = name.replace("/", "_")
            if (repo.root / "market" / safe_name).exists():
                continue

            repo.import_from_market(
                name=name,
                source="vnpy社区",
                source_url=f"https://github.com/vnpy/vnpy_ctastrategy/tree/{commit[:7]}/{py_file.relative_to(target)}",
                stype=_guess_type(info),
                desc=info["desc"] or f"vnpy策略: {info['class_name']}",
                params=info["params"],
                strategy_code=info["code"],
                category="market",
                tags=["vnpy"] + _guess_tags(info),
            )
            imported += 1
            logger.info(f"  导入: {name}")

    logger.info(f"vnpy 导入完成: {imported} 个策略")
    return imported


def import_from_gitee_vnpy(repo: StrategyRepo = None) -> int:
    """
    从 Gitee vnpy 镜像仓库导入策略（国内可访问）。

    vnpy/alpha/strategy/strategies/ 目录下约有 1+ 策略。
    另有 examples/ 目录下有更多示例策略。
    """
    if repo is None:
        repo = StrategyRepo()

    logger.info("从 Gitee vnpy 导入策略...")
    url = "https://gitee.com/vnpy/vnpy.git"
    target = TEMP_DIR / "vnpy_gitee"

    if not _git_clone_shallow(url, target):
        return 0

    commit = _git_get_commit(target)

    # vnpy Gitee 镜像的策略路径
    strategy_dirs = (
        list(target.glob("**/strategies/"))
        + list(target.glob("**/strategy/"))
        + [target / "examples"]
    )
    imported = 0

    for strat_dir in strategy_dirs:
        if not strat_dir.exists():
            continue
        for py_file in strat_dir.rglob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue

            info = parse_strategy_file(py_file)
            if info is None:
                continue

            name = f"gitee_vnpy_{info['name']}"
            safe_name = name.replace("/", "_")
            if (repo.root / "market" / safe_name).exists():
                continue

            repo.import_from_market(
                name=name,
                source="Gitee-vnpy社区",
                source_url=f"https://gitee.com/vnpy/vnpy/tree/{commit[:7]}/{py_file.relative_to(target)}",
                stype=_guess_type(info),
                desc=info["desc"] or f"Gitee vnpy策略: {info['class_name']}",
                params=info["params"],
                strategy_code=info["code"],
                category="market",
                tags=["vnpy", "gitee"] + _guess_tags(info),
            )
            imported += 1
            logger.info(f"  导入: {name}")

    logger.info(f"Gitee vnpy 导入完成: {imported} 个策略")
    return imported


def import_from_local(
    local_path: str,
    source_label: str = "",
    repo: StrategyRepo = None,
) -> int:
    """
    从本地目录导入策略（已 clone 好的仓库）。

    参数
    ----------
    local_path : str   本地目录路径
    source_label : str 来源标签（如 "vnpy本地"）
    repo : StrategyRepo

    返回
    -------
    int : 导入数量
    """
    if repo is None:
        repo = StrategyRepo()

    target = Path(local_path)
    if not target.exists():
        logger.error(f"路径不存在: {local_path}")
        return 0

    label = source_label or target.name
    logger.info(f"从本地路径导入: {target}")

    # 找策略目录：先按标准路径，找不到再扫描根级子目录
    strategy_dirs = (list(target.glob("**/strategies/"))
                     + list(target.glob("**/strategy/"))
                     + [target / "examples"])
    # 如果标准路径都没找到策略文件，加根级子目录兜底
    has_strategies = False
    for d in strategy_dirs:
        if d.exists():
            has_strategies = True
            break
    if not has_strategies:
        # 每个根级子目录都可能是一个策略
        strategy_dirs.extend([d for d in target.iterdir() if d.is_dir() and not d.name.startswith(".")])
    imported = 0

    for strat_dir in strategy_dirs:
        if not strat_dir.exists():
            continue
        for py_file in strat_dir.rglob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue

            info = parse_strategy_file(py_file)
            if info is None:
                continue

            name = f"local_{label}_{info['name']}"
            safe_name = name.replace("/", "_")
            if (repo.root / "market" / safe_name).exists():
                continue

            repo.import_from_market(
                name=name,
                source=f"本地-{label}",
                source_url=str(py_file),
                stype=_guess_type(info),
                desc=info["desc"] or f"本地策略: {info['class_name']}",
                params=info["params"],
                strategy_code=info["code"],
                category="market",
                tags=["local", label] + _guess_tags(info),
            )
            imported += 1
            logger.info(f"  导入: {name}")

    logger.info(f"本地导入完成: {imported} 个策略")
    return imported


def import_from_freqtrade(repo: StrategyRepo = None) -> int:
    """
    从 Freqtrade 社区策略仓库导入策略。

    freqtrade-strategies 仓库包含 300+ 用户贡献的策略。
    每个策略是一个 .py 文件。
    """
    if repo is None:
        repo = StrategyRepo()

    logger.info("从 Freqtrade 社区导入策略（可能需要几分钟）...")
    url = "https://github.com/freqtrade/freqtrade-strategies.git"
    target = TEMP_DIR / "freqtrade_strategies"

    if not _git_clone_shallow(url, target):
        return 0

    commit = _git_get_commit(target)

    imported = 0
    skipped = 0
    # 只取 user_data/strategies/ 目录
    strategies_dir = target / "user_data" / "strategies"
    if not strategies_dir.exists():
        strategies_dir = target  # fallback to root

    for py_file in strategies_dir.rglob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        info = parse_strategy_file(py_file)
        if info is None:
            skipped += 1
            continue

        name = f"ft_{info['name']}"
        safe_name = name.replace("/", "_")
        if (repo.root / "market" / safe_name).exists():
            continue

        repo.import_from_market(
            name=name,
            source="Freqtrade社区",
            source_url=f"https://github.com/freqtrade/freqtrade-strategies/blob/{commit[:7]}/{py_file.relative_to(target)}",
            stype=_guess_type(info),
            desc=info["desc"] or f"Freqtrade策略: {info['class_name']}",
            params=info["params"],
            strategy_code=info["code"],
            category="market",
            tags=["freqtrade", "crypto"] + _guess_tags(info),
        )
        imported += 1
        if imported % 50 == 0:
            logger.info(f"  进度: {imported} 个...")

    logger.info(f"Freqtrade 导入完成: {imported} 个策略, 跳过 {skipped} 个")
    return imported


def import_all(repo: StrategyRepo = None) -> dict:
    """一键从所有来源导入"""
    if repo is None:
        repo = StrategyRepo()
    return {
        "vnpy": import_from_vnpy(repo),
        "gitee_vnpy": import_from_gitee_vnpy(repo),
        "freqtrade": import_from_freqtrade(repo),
    }


# ============================================================
# 辅助
# ============================================================

def _guess_type(info: dict) -> str:
    """根据策略代码内容猜测类型"""
    text = (info.get("desc", "") + " " + info.get("name", "") + " " +
            " ".join(info.get("params", {}).keys())).lower()

    if any(w in text for w in ["boll", "rsi", "mean", "revert", "回归", "回归", "布林"]):
        return "均值回归"
    if any(w in text for w in ["momentum", "trend", "roc", "动量", "趋势"]):
        return "趋势跟踪"
    if any(w in text for w in ["grid", "网格", "震荡"]):
        return "震荡"
    if any(w in text for w in ["breakout", "donchian", "channel", "turtle", "突破", "通道", "海龟"]):
        return "突破"
    if any(w in text for w in ["arbitrage", "stat", "套利"]):
        return "套利"
    return "unknown"


def _guess_tags(info: dict) -> list:
    """根据策略内容猜测标签"""
    tags = []
    text = (info.get("desc", "") + " " + info.get("name", "")).lower()
    mapping = [
        (["ma", "sma", "ema", "均线"], "均线"),
        (["rsi"], "RSI"),
        (["macd"], "MACD"),
        (["boll", "布林"], "布林带"),
        (["volume", "量"], "成交量"),
        (["atr", "止损", "stop"], "止损"),
        (["多因子", "factor"], "多因子"),
        (["ml", "machine", "ai", "learn", "predict"], "ML"),
    ]
    for keywords, tag in mapping:
        if any(kw in text for kw in keywords):
            tags.append(tag)
    return tags


# ============================================================
# 命令行
# ============================================================
# python strategies_repo/importer.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 60)
    print("策略自动导入器")
    print("=" * 60)
    print("[1] vnpy GitHub (20+ 策略)")
    print("[2] vnpy Gitee   (国内可访问，1+ 策略+示例)")
    print("[3] Freqtrade 社区 (300+ 策略)")
    print("[4] 全部")
    print("[V] 验证 — 检查远程仓库是否可访问（不导入）")
    print("[0] 退出")
    choice = input("选择: ").strip().upper()

    if choice == "1":
        n = import_from_vnpy()
        print(f"导入完成: {n} 个")
    elif choice == "2":
        n = import_from_gitee_vnpy()
        print(f"导入完成: {n} 个")
    elif choice == "3":
        n = import_from_freqtrade()
        print(f"导入完成: {n} 个")
    elif choice == "4":
        result = import_all()
        print(f"导入完成: vnpy={result['vnpy']}, gitee_vnpy={result['gitee_vnpy']}, freqtrade={result['freqtrade']}")
    elif choice == "V":
        print("\n验证远程仓库...")
        repos = [
            ("vnpy_ctastrategy", "https://github.com/vnpy/vnpy_ctastrategy.git"),
            ("vnpy_gitee", "https://gitee.com/vnpy/vnpy.git"),
            ("freqtrade-strategies", "https://github.com/freqtrade/freqtrade-strategies.git"),
        ]
        for name, url in repos:
            ok = _verify_repo_url(url)
            print(f"  {name:<25} {'OK' if ok else 'FAIL'}")
            if not ok:
                print(f"    检查: git ls-remote {url}")
        print("验证完成")
