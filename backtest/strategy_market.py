"""
策略市场扫描器 — 从策略注册中心获取策略，自动回测，对比排名。

对应两种运作模式：
  模式 A（市场扫描）← 这个模块：从现有策略库中拿策略 → 逐个回测 → 记录 → 对比
  模式 B（策略挖掘）← strategy_miner.py：自由组合元素 → 生成候选 → 回测 → 筛选

用法
--------
>>> from backtest.strategy_market import scan_market
>>> results = scan_market(df, cash=100000)
>>> print(results["ranking"])
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger
from backtest.engine.bt_runner import run_backtest
from backtest.engine.strategy_registry import (
    ALL_STRATEGIES, BUILTIN_STRATEGIES, MARKET_STRATEGIES,
    FEP_STRATEGIES, STRATEGY_TYPES, SOURCE_LABELS,
)

logger = get_logger("strategy_market")


# ============================================================
# 市场扫描器
# ============================================================

def scan_market(
    df: pd.DataFrame,
    cash: float = 100000.0,
    strategies: Optional[list[str]] = None,
    split_date: str = "",
) -> dict:
    """
    扫描策略市场：逐个回测，记录结果，对比排名。

    这就是"模式 A"——从现有策略库里拿策略，每个跑一遍，看哪个好。

    参数
    ----------
    df : DataFrame
        行情数据
    cash : float
        初始资金
    strategies : list[str]
        要测试的策略名。不填=全部
    split_date : str
        样本外分割日期。不填则只做全量回测，不做样本外

    返回
    -------
    dict : ranking/records/summary
    """
    names = strategies or list(ALL_STRATEGIES.keys())
    total = len(names)
    logger.info(f"策略市场扫描: {total} 个策略, 资金 {cash:,.0f}, 数据 {len(df)} 条")

    records = []
    start_time = time.time()

    for i, name in enumerate(names):
        info = ALL_STRATEGIES[name]
        logger.info(f"  [{i+1}/{total}] {name} ({info['source']})")

        try:
            params = dict(info.get("params", {}))
            result = run_backtest(info["class"], df, initial_cash=cash, **params)

            records.append({
                "name": name,
                "source": info["source"],
                "desc": info["desc"],
                "annual_return": result["annual_return"],
                "total_return": result["total_return"],
                "drawdown": result["drawdown"],
                "sharpe": result["sharpe"],
                "total_trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "final_value": result["final_value"],
            })
        except Exception as e:
            logger.error(f"  {name} 失败: {e}")
            records.append({
                "name": name,
                "source": info["source"],
                "desc": info["desc"],
                "annual_return": 0,
                "total_return": 0,
                "drawdown": 1.0,
                "sharpe": None,
                "total_trades": 0,
                "win_rate": 0,
                "final_value": cash,
                "error": str(e),
            })

    # 排名（按年化收益）
    records.sort(key=lambda x: x["annual_return"], reverse=True)
    elapsed = time.time() - start_time

    # 构建总结
    lines = [
        "=" * 95,
        "                   策略市场扫描 — 排名",
        "=" * 95,
        f"{'排名':<4} {'策略':<14} {'来源':<16} {'年化':>6} {'回撤':>6} {'夏普':>6} {'交易':>4} {'胜率':>6}",
        "-" * 95,
    ]
    for i, r in enumerate(records):
        s = r.get("sharpe")
        sharpe_str = f"{s:.2f}" if s else "N/A"
        lines.append(
            f"{i+1:<4} {r['name']:<14} {r['source']:<16} "
            f"{r['annual_return']:>5.1%} {r['drawdown']:>5.1%} "
            f"{sharpe_str:>6} {r['total_trades']:>4} {r['win_rate']:>5.1%}"
        )
    lines.append("=" * 95)

    best = records[0]
    lines.append(f"\n冠军: '{best['name']}' ({best['source']})")
    lines.append(f"年化 {best['annual_return']:.2%} | 回撤 {best['drawdown']:.2%} | {best['desc']}")
    lines.append(f"\n共扫描 {len(records)} 个策略, 耗时 {elapsed:.0f}s")

    summary = "\n".join(lines)
    print(summary)

    return {
        "ranking": records,
        "records": records,
        "summary": summary,
        "total_scanned": len(records),
        "elapsed_seconds": elapsed,
    }


def scan_and_save(
    df: pd.DataFrame,
    output_path: str = "notebooks/market_scan.csv",
    cash: float = 100000.0,
) -> pd.DataFrame:
    """
    扫描并保存结果到 CSV（方便在 Excel 里看）。

    示例
    --------
    >>> table = scan_and_save(df)
    >>> print(table[["name", "annual_return", "drawdown"]])
    """
    result = scan_market(df, cash)
    records = pd.DataFrame(result["records"])
    records.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"扫描结果已保存: {output_path}")
    return records


# ============================================================
# 命令行测试
# ============================================================
# python backtest/strategy_market.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    print("=" * 60)
    print("策略市场扫描测试")
    print("=" * 60)

    df = fetch_index_daily("沪深300", "20230101", "20250601")
    scan_market(df, cash=100000)
