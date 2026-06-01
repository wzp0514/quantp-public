"""
策略评分排名系统 — 跨来源统一评分，按分数取 Top N

评分公式（可配置权重）:
  total = external_rating × 0.3 + sharpe_score × 0.3 + return_score × 0.2 - drawdown_score × 0.2

external_rating: 来源平台的评分归一化（TradingView stars, GitHub stars, 聚宽热度...）
sharpe_score:     夏普比率归一化（负值惩罚）
return_score:     年化收益归一化
drawdown_score:   最大回撤惩罚

用法
--------
>>> from strategies_repo.scoring import rank_strategies, get_top_n
>>> ranked = rank_strategies(repo.list())  # 全部策略排名
>>> top10 = get_top_n(repo, 10)            # 取 Top 10
"""

import logging
from typing import Optional

import numpy as np

# 默认权重（可改）
WEIGHTS = {
    "external_rating": 0.30,   # 社区评分（TradingView星级/GitHub stars/聚宽热度）
    "sharpe": 0.30,             # 夏普比率（风险调整后收益）
    "annual_return": 0.20,      # 年化收益
    "drawdown": -0.20,          # 最大回撤（惩罚项，负权重）
}


def score_strategy(strategy: dict) -> float:
    """
    计算单个策略的综合评分。

    输入 strategy 来自 repo.list()。

    评分分三步：
      1. 提取各维度原始值
      2. 归一化到 0~1 区间
      3. 加权求和
    """
    score = 0.0

    # 1. 外部评分
    ext_rating = _extract_external_rating(strategy)
    score += ext_rating * WEIGHTS["external_rating"]

    # 2. 回测指标（从 metrics 字段）
    metrics = strategy.get("metrics", {}) or {}
    if isinstance(metrics, str):
        import json
        try:
            metrics = json.loads(metrics)
        except Exception:
            metrics = {}

    sharpe = metrics.get("sharpe") or strategy.get("sharpe")
    ann_ret = metrics.get("annual_return") or strategy.get("annual_return")
    dd = metrics.get("drawdown") or strategy.get("drawdown")

    # 夏普：0 以下全惩罚，0~3 线性，3 以上满分
    if sharpe is not None:
        sharpe_norm = min(max((sharpe + 1) / 4, 0), 1.0)  # -1→0, 0→0.25, 1→0.5, 3→1.0
        score += sharpe_norm * WEIGHTS["sharpe"]

    # 年化收益：-20%→0, 0→0.3, 20%→1.0
    if ann_ret is not None:
        ret_norm = min(max((ann_ret + 0.2) / 0.4, 0), 1.0)
        score += ret_norm * WEIGHTS["annual_return"]

    # 回撤：0%→1.0（满分）, 30%→0（严重惩罚）
    if dd is not None:
        dd_pct = abs(dd)
        dd_norm = 1.0 - min(dd_pct / 0.3, 1.0)  # 回撤越大分数越低
        score += dd_norm * WEIGHTS["drawdown"]

    return round(score, 4)


def rank_strategies(strategies: list[dict]) -> list[dict]:
    """
    对策略列表评分并排名。

    返回
    -------
    list[dict] : 原策略 dict + score 字段，按分数降序
    """
    ranked = []
    for s in strategies:
        s = dict(s)  # 不修改原数据
        s["_score"] = score_strategy(s)
        ranked.append(s)

    ranked.sort(key=lambda x: x["_score"], reverse=True)
    return ranked


def get_top_n(repo, n: int = 10, min_score: float = 0.0) -> list[dict]:
    """
    取评分最高的 N 个策略。

    参数
    ----------
    repo : StrategyRepo 或 DbStrategyRepo
    n : int
        取多少个
    min_score : float
        最低分数门槛（低于此分数的不考虑）

    返回
    -------
    list[dict]
    """
    strategies = repo.list()
    ranked = rank_strategies(strategies)
    filtered = [s for s in ranked if s["_score"] >= min_score]
    return filtered[:n]


def score_from_source(source_rating: float, sharpe: float = 0,
                      annual_return: float = 0, drawdown: float = 0) -> float:
    """
    给定原始值计算分数（不依赖策略 dict）。

    用于在导入策略时预计算评分。
    """
    s = score_strategy({
        "external_rating": source_rating,
        "metrics": {
            "sharpe": sharpe,
            "annual_return": annual_return,
            "drawdown": drawdown,
        }
    })
    return s


def _extract_external_rating(strategy: dict) -> float:
    """
    从策略元数据中提取来源评分，归一化到 0~1。

    不同来源的评分标准不同，这里做粗略归一化：
      TradingView: 0-5 stars → /5
      GitHub: 0-50K stars → log scale
      聚宽:  热度值 → /max_heat
    """
    source = strategy.get("source", "").lower()
    tags = strategy.get("tags", []) or []

    # TradingView 评分（0-5 stars）
    if "tradingview" in source or "tv" in tags:
        rating = strategy.get("external_rating", 0)
        if isinstance(rating, (int, float)):
            return min(rating / 5.0, 1.0)

    # GitHub stars（用 log 压缩）
    if "github" in source:
        stars = strategy.get("external_rating", 0)
        if isinstance(stars, (int, float)) and stars > 0:
            return min(np.log10(stars + 1) / np.log10(50001), 1.0)  # 1→0, 50K→1.0

    # 聚宽热度
    if "聚宽" in source or "joinquant" in source.lower():
        heat = strategy.get("external_rating", 0)
        if isinstance(heat, (int, float)) and heat > 0:
            return min(heat / 1000.0, 1.0)

    # 自己的策略（已回测过的给中等分，没回测的给 0 外部评分）
    if source.strip().lower() == "quantp":
        has_backtest = strategy.get("last_backtest") or strategy.get("metrics", {}).get("annual_return")
        return 0.3 if has_backtest else 0.1

    return 0.0


def summary(ranked: list[dict]) -> str:
    """生成排名摘要文本"""
    lines = [
        "=" * 85,
        f"{'排名':<4} {'策略':<20} {'来源':<16} {'评分':>6} {'年化':>6} {'回撤':>6} {'夏普':>6}",
        "-" * 85,
    ]
    for i, s in enumerate(ranked):
        m = s.get("metrics", {}) or {}
        ret = m.get("annual_return")
        dd = m.get("drawdown")
        sp = m.get("sharpe")
        lines.append(
            f"{i+1:<4} {s['name'][:19]:<20} {s.get('source','')[:15]:<16} "
            f"{s['_score']:>6.3f} "
            f"{ret if ret else 0:>5.1%} "
            f"{dd if dd else 0:>5.1%} "
            f"{f'{sp:.2f}' if sp else 'N/A':>6}"
        )
    lines.append("=" * 85)
    return "\n".join(lines)


# ============================================================
# 命令行测试
# ============================================================
# python strategies_repo/scoring.py

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from strategies_repo.repo import StrategyRepo
    repo = StrategyRepo()

    print("评分排名...")
    top10 = get_top_n(repo, n=10)
    print(summary(top10))
