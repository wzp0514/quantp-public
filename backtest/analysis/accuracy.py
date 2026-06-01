"""
回测准确性增强模块 — 参考 Qlib 的统计验证体系

C1. 统计显著性检验 — t-test + bootstrap → "收益是能力还是运气？"
C2. 多基准对比 — 同时对比沪深300/中证500/国债 → 不只是单一基准
C4. 过拟合量化指标 — 参数敏感度指数 → "这策略是调出来的吗？"

用法
--------
>>> from backtest.analysis.accuracy import statistical_test, benchmark_compare, overfitting_score
>>> st = statistical_test(returns)
>>> print(f"p值: {st['p_value']:.4f} — {'显著' if st['significant'] else '不显著'}")
"""

import logging
import numpy as np
import pandas as pd# ============================================================

from config.log import get_logger
logger = get_logger("accuracy")
# C1. 统计显著性检验
# ============================================================

def statistical_test(daily_returns: pd.Series, n_bootstrap: int = 1000) -> dict:
    """
    t-test + bootstrap → 策略收益是否统计显著。

    参数
    ----------
    daily_returns : pd.Series
        策略的每日收益率
    n_bootstrap : int
        bootstrap 重采样次数（默认1000）

    返回
    -------
    dict: {
        mean_return, t_stat, p_value, significant,
        bootstrap_ci: (lower, upper),
        prob_negative: float (bootstrap中亏钱的概率),
        interpretation: str,
    }
    """
    if len(daily_returns) < 30:
        return {"error": "数据不足（<30个交易日）"}

    returns = daily_returns.dropna().values
    n = len(returns)

    # 1. t-test
    mean_ret = np.mean(returns)
    std_err = np.std(returns, ddof=1) / np.sqrt(n)
    t_stat = mean_ret / std_err if std_err > 0 else 0

    # 自由度 = n-1, 近似 p 值（双尾）
    # 用正态近似（样本够大时）
    from math import erf, sqrt
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

    # 2. Bootstrap
    bs_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        bs_means.append(np.mean(sample))

    bs_means = np.array(bs_means)
    ci_lower = np.percentile(bs_means, 2.5)
    ci_upper = np.percentile(bs_means, 97.5)
    prob_negative = np.mean(bs_means < 0)

    # 3. 判断
    significant = p_value < 0.05 and prob_negative < 0.10

    if significant:
        interp = (f"统计显著 (p={p_value:.4f})——策略收益不太可能是运气。"
                  f"Bootstrap显示仅 {prob_negative:.0%} 的概率亏钱。")
    elif p_value < 0.10:
        interp = (f"边际显著 (p={p_value:.4f})——有一定可信度，但不够稳健。"
                  f"建议扩大样本或增加样本外验证。")
    else:
        interp = (f"不显著 (p={p_value:.4f})——策略收益可能是随机波动。"
                  f"Bootstrap显示 {prob_negative:.0%} 的概率亏钱。不建议实盘。")

    result = {
        "mean_return": mean_ret,
        "annual_return": mean_ret * 252,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": significant,
        "bootstrap_ci": (ci_lower, ci_upper),
        "prob_negative": prob_negative,
        "n_bootstrap": n_bootstrap,
        "interpretation": interp,
    }

    logger.info(f"统计检验: p={p_value:.4f}, bootstrap亏钱概率={prob_negative:.0%}, {'显著' if significant else '不显著'}")
    return result


# ============================================================
# C2. 多基准对比
# ============================================================

def benchmark_compare(
    strategy_annual_return: float,
    strategy_drawdown: float,
    strategy_sharpe: float = None,
    benchmarks: dict = None,
) -> dict:
    """
    策略 vs 多个基准的一键对比。

    默认基准：沪深300总收益、中证500总收益、年化3%（约等于国债/余额宝）

    用法
    --------
    >>> bm = benchmark_compare(0.08, -0.15, 0.5)
    >>> print(bm['verdict'])
    """
    if benchmarks is None:
        benchmarks = {
            "沪深300(买入持有)": {"annual": 0.0229, "dd": -0.47},
            "中证500(买入持有)": {"annual": 0.0150, "dd": -0.52},
            "无风险(余额宝)": {"annual": 0.03, "dd": 0},
        }

    results = {}
    for name, bm in benchmarks.items():
        excess = strategy_annual_return - bm["annual"]
        risk_better = abs(strategy_drawdown) < abs(bm["dd"]) if bm["dd"] != 0 else False
        results[name] = {
            "excess": excess,
            "risk_better": risk_better,
        }

    # 综合判定
    beat_count = sum(1 for r in results.values() if r["excess"] > 0)
    total = len(results)
    beat_ratio = beat_count / total if total > 0 else 0

    if beat_ratio >= 0.8:
        verdict = "优秀——跑赢 80% 以上基准"
    elif beat_ratio >= 0.5:
        verdict = "及格——至少跑赢一半基准"
    else:
        verdict = "不及格——跑输大部分基准，不如买指数"

    lines = ["多基准对比:", f"  策略年化: {strategy_annual_return:.2%}"]
    for name, r in results.items():
        emoji = "+" if r["excess"] > 0 else "-"
        lines.append(f"  {emoji} vs {name}: 超额 {r['excess']:+.2%}")
    lines.append(f"  判定: {verdict} ({beat_count}/{total})")

    output = {
        "results": results,
        "beat_ratio": beat_ratio,
        "verdict": verdict,
        "summary": "\n".join(lines),
    }

    print(output["summary"])
    return output


# ============================================================
# C4. 过拟合量化指标
# ============================================================

def overfitting_score(
    train_returns: list[float],
    param_variations: list[dict],
) -> dict:
    """
    计算参数敏感度指数（Parameter Sensitivity Index, PSI）。

    原理：把策略参数微调（如均线 20→19/21），看收益变化多少。
    变化 > 20% → 过拟合警告。

    参数
    ----------
    train_returns : list[float]
        不同参数组合下的训练期收益
    param_variations : list[dict]
        每组参数的具体值

    返回
    -------
    dict: {psi, max_change, overfit_risk, interpretation}
    """
    if len(train_returns) < 3:
        return {"error": "至少需要3组参数变体"}

    returns = np.array(train_returns)
    base = returns[0]

    # 最大偏离
    max_change = max(abs(r - base) for r in returns[1:]) / max(abs(base), 0.001)

    # PSI = 标准差 / 均值绝对值（变异系数）
    psi = np.std(returns) / (abs(np.mean(returns)) + 0.001)

    # 判定
    if max_change < 0.15:
        risk = "low"
        interp = f"参数稳健——最大收益变化 {max_change:.0%} < 15%，不过拟合"
    elif max_change < 0.30:
        risk = "medium"
        interp = f"中度敏感——最大收益变化 {max_change:.0%}，可能过拟合，建议更多样本外验证"
    else:
        risk = "high"
        interp = f"[!] 高度过拟合 -- 最大收益变化 {max_change:.0%} >= 30%，策略不可靠"

    result = {
        "psi": round(psi, 4),
        "max_change": round(max_change, 4),
        "overfit_risk": risk,
        "interpretation": interp,
    }

    logger.info(f"过拟合检查: PSI={psi:.3f}, 最大变化={max_change:.1%}, 风险={risk}")
    return result


# ============================================================
# 命令行
# ============================================================
# python backtest/analysis/accuracy.py

if __name__ == "__main__":
    # 模拟数据演示
    np.random.seed(42)
    fake_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))

    st = statistical_test(fake_returns)
    print("统计显著性测试（模拟数据）:")
    print(f"  p={st['p_value']:.4f}, 显著={st['significant']}, {st['interpretation'][:60]}")

    print()
    benchmark_compare(0.08, -0.15, 0.5)

    print()
    os_result = overfitting_score(
        [0.08, 0.06, 0.09, 0.04, 0.11],
        [{"slow": 20}, {"slow": 18}, {"slow": 22}, {"slow": 15}, {"slow": 25}],
    )
    print(f"过拟合检查: {os_result['interpretation']}")
