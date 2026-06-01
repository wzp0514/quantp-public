"""
策略横向对比（Strategy Shootout） — 参考 Sattoro Hub 并行分析引擎

同一份数据、同一时间区间、同一初始资金，全部 11 个策略一起跑，
输出排名表和多维度对比报告。

用法
--------
>>> from backtest.analysis.shootout import run_shootout
>>> result = run_shootout(df, cash=100000)
>>> print(result["ranking"])
"""

import logging
from typing import Optional

import pandas as pd
from backtest.strategy_market import ALL_STRATEGIES

from config.log import get_logger
logger = get_logger("shootout")
def run_shootout(
    df: pd.DataFrame,
    cash: float = 100000.0,
    strategies: Optional[list[str]] = None,
    custom_params: Optional[dict] = None,
    apply_risk_controls: bool = True,
) -> dict:
    """
    策略大比武：全部策略在相同条件下回测，输出排名。

    参数
    ----------
    df : DataFrame
        行情数据
    cash : float
        初始资金
    strategies : list[str]
        要测试的策略名列表。不填 = 全部
    custom_params : dict
        自定义参数，如 {"双均线交叉": {"fast": 10, "slow": 30}}
    apply_risk_controls : bool
        True=大比武共用同一套风控参数（C12: 统一风控标准）

    返回
    -------
    dict，包含 ranking/raw_results/comparison_table
    """
    from backtest.engine.bt_runner import run_backtest

    # 加载共用风控参数（C12）
    risk_params = {}
    if apply_risk_controls:
        try:
            from config.loader import get_config
            cfg = get_config()
            risk = cfg.get("risk", {})
            risk_params = {
                "max_position_pct": risk.get("max_position_pct", 0.3),
                "max_drawdown_pct": risk.get("max_drawdown_pct", 0.15),
            }
        except Exception:
            pass

    names = strategies or list(ALL_STRATEGIES.keys())
    logger.info(f"策略大比武开始: {len(names)} 个策略, 资金 {cash:,.0f}, 数据 {len(df)} 条")
    if risk_params:
        logger.info(f"  共用风控: 仓位上限{risk_params['max_position_pct']:.0%}, "
                    f"回撤熔断{risk_params['max_drawdown_pct']:.0%}")

    results = []
    for name in names:
        info = ALL_STRATEGIES[name]
        params = dict(info.get("params", {}))
        if custom_params and name in custom_params:
            params.update(custom_params[name])

        strategy_class = info["class"]

        strategy_type = info.get("type", info.get("source", ""))
        logger.info(f"  运行: {name} ({strategy_type}) {params}")
        r = run_backtest(strategy_class, df, initial_cash=cash, **params)

        # 风控审查标注（C12: 回撤超限标风险）
        risk_flag = ""
        if r["drawdown"] > risk_params.get("max_drawdown_pct", 0.15):
            risk_flag = " [回撤超限]"
        elif r["total_trades"] < 5:
            risk_flag = " [交易过少]"

        results.append({
            "name": name,
            "type": strategy_type,
            "source": info.get("source", "unknown"),
            "desc": info.get("desc", ""),
            "annual_return": r["annual_return"],
            "total_return": r["total_return"],
            "drawdown": r["drawdown"],
            "sharpe": r["sharpe"],
            "total_trades": r["total_trades"],
            "win_rate": r["win_rate"],
            "final_value": r["final_value"],
            "risk_flag": risk_flag,
        })

    # 按年化收益排名
    ranking = sorted(results, key=lambda x: x["annual_return"], reverse=True)
    comparison = pd.DataFrame(ranking)

    # 排名表
    lines = [
        "=" * 95,
        "                     策略大比武 — 排名表",
        "=" * 95,
        f"{'排名':<4} {'策略':<12} {'类型':<10} {'来源':<6} {'年化收益':>8} {'最大回撤':>8} {'夏普':>6} {'交易':>5} {'胜率':>6} {'风控':<12}",
        "-" * 95,
    ]
    for i, r in enumerate(ranking):
        sharpe_str = f"{r['sharpe']:.2f}" if r['sharpe'] else "N/A"
        lines.append(
            f"{i+1:<4} {r['name']:<12} {r['type']:<10} {r.get('source','?'):<6} "
            f"{r['annual_return']:>7.2%} {r['drawdown']:>7.2%} "
            f"{sharpe_str:>6} {r['total_trades']:>5} {r['win_rate']:>5.1%} "
            f"{r.get('risk_flag',''):<12}"
        )
    lines.append("=" * 95)

    # 风控汇总
    risk_flagged = [r for r in ranking if r.get("risk_flag")]
    if risk_flagged:
        lines.append(f"\n风控提醒: {len(risk_flagged)} 个策略需关注")
        for r in risk_flagged:
            lines.append(f"  - {r['name']}:{r['risk_flag']}")

    # 最佳策略推荐
    best = ranking[0]
    lines.append(f"\n推荐: '{best['name']}' — {best['desc']}")
    lines.append(f"年化 {best['annual_return']:.2%}, 回撤 {best['drawdown']:.2%}, "
                 f"夏普 {best.get('sharpe') or 'N/A'}")
    lines.append("\n建议将推荐策略进入纸上交易验证 4 周。")

    summary = "\n".join(lines)
    print(summary)

    return {
        "ranking": ranking,
        "comparison": comparison,
        "raw_results": results,
        "summary": summary,
        "risk_flagged": risk_flagged,
    }


def quick_compare(df: pd.DataFrame, cash: float = 100000.0, **param_overrides) -> pd.DataFrame:
    """
    快速对比（返回 DataFrame，方便在 Jupyter 里看）。

    示例
    --------
    >>> df = fetch_index_daily("沪深300")
    >>> table = quick_compare(df)
    >>> table[["name", "annual_return", "drawdown"]]
    """
    result = run_shootout(df, cash, custom_params=param_overrides or None)
    return result["comparison"]


def multi_dimension_rank(df: pd.DataFrame, cash: float = 100000.0) -> dict:
    """
    多维度 Leaderboard：复合/收益/风险/人气 四种排名。

    返回
    -------
    dict: composite/return_rank/risk_rank/popularity_rank + history
    """
    result = run_shootout(df, cash)
    ranking = result["ranking"]

    composite = []
    for i, r in enumerate(ranking):
        return_score = max(0, r["annual_return"]) * 100
        risk_score = max(0, 1 - r["drawdown"]) * 100
        trade_score = min(r["total_trades"], 50) * 2
        composite_score = round(return_score * 0.4 + risk_score * 0.3 + trade_score * 0.2 + (11 - i) * 0.1, 1)
        composite.append({
            "name": r["name"], "source": r.get("source", ""),
            "annual_return": r["annual_return"], "drawdown": r["drawdown"],
            "sharpe": r.get("sharpe"), "composite_score": composite_score,
        })

    composite.sort(key=lambda x: x["composite_score"], reverse=True)

    return_rank = sorted(ranking, key=lambda x: x["annual_return"], reverse=True)[:5]
    risk_rank = sorted(ranking, key=lambda x: x["drawdown"])[:5]
    popularity = sorted(ranking, key=lambda x: x.get("total_trades", 0), reverse=True)[:5]

    # 保存历史
    import json, os
    history_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vault", "vault_data", "leaderboard_history.json")
    history = []
    try:
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
    except Exception:
        pass

    record = {
        "date": str(pd.Timestamp.now().date()),
        "top_composite": composite[0]["name"] if composite else "",
        "top_return": return_rank[0]["name"] if return_rank else "",
        "strategy_count": len(ranking),
    }
    history.append(record)
    history = history[-30:]  # 保留最近 30 次

    try:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return {
        "composite": composite,
        "return_rank": [{"name": r["name"], "annual_return": r["annual_return"]} for r in return_rank],
        "risk_rank": [{"name": r["name"], "drawdown": r["drawdown"]} for r in risk_rank],
        "popularity_rank": [{"name": r["name"], "total_trades": r["total_trades"]} for r in popularity],
        "history": history,
        "history_count": len(history),
    }


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/shootout.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily

    df = fetch_index_daily("沪深300", "20230101", "20250601")
    run_shootout(df, cash=100000)
