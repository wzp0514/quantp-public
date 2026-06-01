"""
绩效归因 — 参考 BsStrategy 绩效评估

把策略收益拆解为几个来源，知道"赚的是什么钱"：

1. 市场 Beta 收益 — 大盘涨了你也涨，这个钱不是你的本事
2. 超额 Alpha — 扣除市场涨跌后，你的策略真正创造的收益
3. 行业/风格贡献 — 你的策略偏重什么行业/风格

如果策略收益主要来自 Beta（市场好所以赚），那换个熊市就亏回去了。
真正有价值的策略是 Alpha 稳定为正。

参考：
  Brinson 归因模型
  Fama-French 因子模型
"""

import logging
import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("attribution")
def simple_attribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    简单归因：把策略收益拆成 市场Beta + 超额Alpha

    原理：
      策略收益 = α + β × 基准收益 + ε（随机噪声）

      跑线性回归，β 是策略对市场的敏感度，α 是截距（超额收益）。
      β = 1.0 → 市场涨1%你也涨1%，就是个指数基金
      β > 1.0 → 比市场更激进（涨得多跌得也多）
      β < 1.0 → 比市场更保守
      α > 0 → 真的有超额收益！

    参数
    ----------
    strategy_returns : pd.Series
        策略的每日收益率（小数）
    benchmark_returns : pd.Series
        基准的每日收益率（例如沪深300自身）

    返回
    -------
    dict : {alpha, beta, r_squared, annual_alpha, detail}

    示例
    --------
    >>> from backtest.engine.bt_runner import run_backtest
    >>> from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    >>> result = run_backtest(MaCrossStrategy, df)
    >>> attr = simple_attribution(strategy_returns, benchmark_returns)
    >>> print(f"Alpha: {attr['annual_alpha']:.2%}, Beta: {attr['beta']:.2f}")
    """
    # 对齐两个序列的日期
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 30:
        return {"error": f"共同数据点不足 ({len(common_idx)} < 30)，无法可靠归因"}

    y = strategy_returns[common_idx].values
    x = benchmark_returns[common_idx].values

    # 简单线性回归: y = alpha + beta * x
    # 用 numpy 的 polyfit（1阶多项式 = 直线）
    beta, alpha = np.polyfit(x, y, 1)

    # R²（拟合优度）：策略收益有多大比例能被市场解释
    y_pred = alpha + beta * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # 年化 Alpha（假设 252 个交易日）
    annual_alpha = alpha * 252

    # 解读
    if annual_alpha > 0.05:
        interpretation = "显著正Alpha，策略有真正的超额收益能力"
    elif annual_alpha > 0:
        interpretation = "轻微正Alpha，勉强跑赢市场"
    elif annual_alpha > -0.05:
        interpretation = "轻微负Alpha，和买指数差不多甚至略差"
    else:
        interpretation = "显著负Alpha，策略跑输市场，需要改进"

    if r_squared > 0.8:
        interpretation += "；高度依赖市场走势（不是坏事，但需要知道）"
    elif r_squared < 0.3:
        interpretation += "；和市场关联度低，策略是独立于大盘的逻辑"

    logger.info(
        f"归因: α(daily)={alpha:.4%}, β={beta:.2f}, R²={r_squared:.2f} | "
        f"年化Alpha={annual_alpha:.2%} | {interpretation}"
    )

    return {
        "alpha_daily": alpha,
        "annual_alpha": annual_alpha,
        "beta": beta,
        "r_squared": r_squared,
        "interpretation": interpretation,
        "detail": (
            f"每日Alpha: {alpha:.4%}\n"
            f"年化Alpha: {annual_alpha:.2%}（策略真正创造的超额收益）\n"
            f"Beta: {beta:.2f}（市场涨1% → 策略涨{beta:.2%}）\n"
            f"R²: {r_squared:.2f}（{r_squared*100:.0f}%的收益波动可由市场解释）\n"
            f"结论: {interpretation}"
        ),
    }


def compare_to_benchmark(
    strategy_total_return: float,
    strategy_annual: float,
    strategy_drawdown: float,
    benchmark_total_return: float,
    benchmark_annual: float,
    benchmark_drawdown: float,
) -> str:
    """
    简单的基准对比报告（看策略有没有跑赢大盘）

    返回纯文本报告。
    """
    excess_return = strategy_annual - benchmark_annual
    risk_advantage = abs(benchmark_drawdown) - abs(strategy_drawdown)

    lines = [
        "=" * 50,
        "              基准对比",
        "=" * 50,
        f"{'指标':<15} {'策略':>12} {'基准':>12} {'差值':>12}",
        "-" * 50,
        f"{'总收益':<15} {strategy_total_return:>11.2%} {benchmark_total_return:>11.2%}",
        f"{'年化收益':<15} {strategy_annual:>11.2%} {benchmark_annual:>11.2%} {excess_return:>+11.2%}",
        f"{'最大回撤':<15} {strategy_drawdown:>11.2%} {benchmark_drawdown:>11.2%} {risk_advantage:>+11.2%}",
        "=" * 50,
    ]

    if excess_return > 0 and risk_advantage > 0:
        lines.append("结论: 策略收益高于基准且回撤更低 — 理想状态")
    elif excess_return > 0:
        lines.append("结论: 收益高于基准但回撤也更大 — 用额外风险换了额外收益")
    elif excess_return < 0 and risk_advantage > 0:
        lines.append("结论: 回撤更低但收益不如基准 — 太保守了")
    else:
        lines.append("结论: 收益和风险都不如直接买基准 — 策略需要改进")

    return "\n".join(lines)


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/attribution.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.engine.bt_runner import run_backtest
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    df = fetch_index_daily("沪深300", "20230101", "20250601")

    # 策略回测
    result = run_backtest(MaCrossStrategy, df, fast=5, slow=20)

    # 计算每日收益率
    df["daily_ret"] = df["close"].pct_change()
    bench_returns = df.set_index("date")["daily_ret"].dropna()

    # 模拟策略每日收益（从 equity_df 推算）
    if not result["equity_df"].empty:
        eq = result["equity_df"]
        eq["daily_ret"] = eq["equity"].pct_change()
        strategy_rets = eq.set_index("date")["daily_ret"].dropna()

        attr = simple_attribution(strategy_rets, bench_returns)
        if "error" not in attr:
            print(attr["detail"])

        # 和基准（买入持有沪深300）对比
        buy_hold_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        buy_hold_annual = (1 + buy_hold_return) ** (252 / len(df)) - 1
        # 简单算基准回撤
        peak = df["close"].expanding().max()
        dd = (df["close"] - peak) / peak
        buy_hold_dd = dd.min()

        report = compare_to_benchmark(
            result["total_return"], result["annual_return"], result["drawdown"],
            buy_hold_return, buy_hold_annual, buy_hold_dd,
        )
        print(report)
