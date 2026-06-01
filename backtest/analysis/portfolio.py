"""
多策略组合模拟 — 一个账户跑多个策略

回答核心问题：
  Q: 一个实盘账户用多种策略是否可行？
  A: 可行，但有前提条件。

前提：
  1. 总资金够切分。10万资金分 3-5 个策略 = 每个 2-3万，ETF 1手=几百元，够用。
  2. 策略之间不能高度相关。5个策略如果都是"均线金叉买"，跌一起跌 = 白分散。
  3. 必须独立风控。每个策略各自有止损，不能互相影响。

资金切分参考：
  ┌──────────┬──────────┬────────────────┐
  │ 总资金    │ 建议策略数 │ 每个策略资金    │
  ├──────────┼──────────┼────────────────┤
  │ 3,000     │ 1        │ 3,000          │
  │ 10,000    │ 2-3      │ 3,000-5,000    │
  │ 50,000    │ 3-5      │ 10,000-15,000  │
  │ 100,000   │ 4-6      │ 15,000-25,000  │
  └──────────┴──────────┴────────────────┘

用法
--------
>>> from backtest.analysis.portfolio import PortfolioSimulator
>>> ps = PortfolioSimulator(cash=50000, strategy_names=["布林带回归", "海龟交易"])
>>> result = ps.run(df)
>>> print(f"组合收益: {result['total_return']:.2%}")
>>> print(f"相关性: {result['correlation']:.2f}")
"""

import logging
import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("portfolio")
class PortfolioSimulator:
    """多策略组合模拟器"""

    def __init__(
        self,
        cash: float = 100000.0,
        strategy_names: list[str] = None,
        equal_weight: bool = True,
    ):
        self.total_cash = cash
        self.strategy_names = strategy_names or []
        self.equal_weight = equal_weight

        if len(strategy_names) < 1:
            raise ValueError("至少需要 1 个策略")

    def run(self, df: pd.DataFrame) -> dict:
        """
        运行多策略组合回测。

        每个策略独立运行，资金等分/自定义分配，
        最后汇总组合表现，计算策略间相关性。
        """
        from backtest.engine.bt_runner import run_backtest
        from backtest.strategy_market import ALL_STRATEGIES

        n = len(self.strategy_names)
        if self.equal_weight:
            per_cash = self.total_cash / n
        else:
            per_cash = self.total_cash / n

        logger.info(f"组合回测: {n} 策略, 总资金 {self.total_cash:,.0f}, 每策略 {per_cash:,.0f}")

        # 逐策略回测
        individual_results = []
        equity_curves = {}

        for name in self.strategy_names:
            if name not in ALL_STRATEGIES:
                logger.warning(f"策略不存在: {name}, 跳过")
                continue

            info = ALL_STRATEGIES[name]
            result = run_backtest(
                info["class"], df,
                initial_cash=per_cash,
                a_share_mode=True,
                **info.get("params", {}),
            )

            individual_results.append({
                "name": name,
                "cash": per_cash,
                "annual_return": result["annual_return"],
                "drawdown": result["drawdown"],
                "sharpe": result["sharpe"],
                "final_value": result["final_value"],
                "trades": result["total_trades"],
            })

            # 提取权益曲线（如果有）
            if not result["equity_df"].empty:
                eq = result["equity_df"]
                equity_curves[name] = eq.set_index("date")["equity"]

        if not individual_results:
            return {"error": "无有效策略"}

        # 汇总
        combined_final = sum(r["final_value"] for r in individual_results)
        combined_return = combined_final / self.total_cash - 1

        # 日均收益（年化）
        days = len(df)
        combined_annual = (1 + combined_return) ** (252 / days) - 1 if days > 0 else 0

        # 策略相关性矩阵
        corr_matrix = None
        avg_corr = 0
        if len(equity_curves) >= 2:
            eq_df = pd.DataFrame(equity_curves)
            returns = eq_df.pct_change().dropna()
            if len(returns) > 10:
                corr = returns.corr()
                corr_matrix = corr
                # 平均两两相关性（对角线下三角）
                n_cols = len(corr.columns)
                if n_cols >= 2:
                    vals = []
                    for i in range(n_cols):
                        for j in range(i + 1, n_cols):
                            vals.append(corr.iloc[i, j])
                    avg_corr = np.mean(vals) if vals else 0

        # 判定相关性
        if avg_corr > 0.7:
            corr_warning = "[!] 策略间高度相关(>0.7) -- 分散效果差，涨跌同步"
        elif avg_corr > 0.4:
            corr_warning = "策略间中度相关(0.4-0.7) -- 有部分分散效果"
        elif len(equity_curves) >= 2:
            corr_warning = "[OK] 策略间低相关(<0.4) -- 分散效果好"
        else:
            corr_warning = "策略不足2个，无法计算相关性"

        # 资金切分合理性
        min_lot_value = 400  # ETF一手约400元(4元×100)
        min_cash_per = min(r["cash"] for r in individual_results)
        max_lots = min_cash_per / min_lot_value

        if max_lots < 5:
            sizing_warning = f"[!] 每策略 {min_cash_per:.0f} 元约买 {max_lots:.0f} 手 -- 太少，盈亏不明显"
        else:
            sizing_warning = f"[OK] 每策略 {min_cash_per:.0f} 元约买 {max_lots:.0f} 手 -- 合理"

        # 排序
        ranked = sorted(individual_results, key=lambda x: x["annual_return"], reverse=True)

        summary_lines = [
            "=" * 60,
            "  多策略组合报告",
            "=" * 60,
            f"  策略数: {n}",
            f"  总资金: {self.total_cash:,.0f} 元",
            f"  最终资金: {combined_final:,.0f} 元",
            f"  总收益: {combined_return:.2%}",
            f"  年化: {combined_annual:.2%}",
            f"  平均相关性: {avg_corr:.2f}",
            f"  {corr_warning}",
            f"  {sizing_warning}",
            "=" * 60,
        ]

        for i, r in enumerate(ranked):
            summary_lines.append(
                f"  {i+1}. {r['name']:<16} 年化{r['annual_return']:>6.1%} "
                f"回撤{r['drawdown']:>5.1%} 交易{r['trades']:>3}笔"
            )

        output = {
            "total_cash": self.total_cash,
            "final_value": combined_final,
            "total_return": combined_return,
            "annual_return": combined_annual,
            "correlation": avg_corr,
            "correlation_matrix": corr_matrix,
            "corr_warning": corr_warning,
            "sizing_warning": sizing_warning,
            "strategies": ranked,
            "summary": "\n".join(summary_lines),
        }

        print(output["summary"])
        return output

    @staticmethod
    def recommend_strategy_count(cash: float) -> tuple[int, float]:
        """
        根据资金量推荐策略数量。

        返回 (建议数量, 每策略资金)
        """
        if cash < 5000:
            return 1, cash
        elif cash < 20000:
            return 2, cash / 2
        elif cash < 50000:
            return 3, cash / 3
        elif cash < 100000:
            return 5, cash / 5
        else:
            return 6, cash / 6


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/portfolio.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    df = fetch_index_daily("沪深300", "20230101", "20250601")

    n, per = PortfolioSimulator.recommend_strategy_count(50000)
    print(f"5万资金建议: {n}个策略, 每个{per:,.0f}元")

    ps = PortfolioSimulator(cash=50000, strategy_names=["布林带回归", "海龟交易", "均值回归"])
    ps.run(df)
