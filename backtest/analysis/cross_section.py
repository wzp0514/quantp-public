"""
截面因子分析 — 多品种横比 + 行业中性化 + 先进数学工具

与 factor_miner.py 的时序因子互补：那里判断"这个指数明天涨还是跌"，
这里判断"这 50 只股票中哪只明天涨最多"。

专业截面分析流程：
  1. 计算每只股票的时序特征（动量/波动/量/质量）
  2. 每日截面上将所有股票排名（Rank 化）
  3. 行业/市值中性化（剥离系统性偏差）
  4. IC 分析（截面因子值与次日收益的秩相关）
  5. 输出股票排序（从好到差）

数学工具：
  - Ledoit-Wolf 协方差收缩（组合优化用）
  - PCA 因子正交化（去冗余）
  - IC/IR 加权（因子质量评分）

用法
--------
>>> from backtest.analysis.cross_section import CrossSectionAnalyzer
>>> csa = CrossSectionAnalyzer(panel_df)
>>> csa.compute_factors()
>>> csa.analyze()
"""

import logging
import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("cross_section")


# ═══════════════════════════════════════════
# Ledoit-Wolf 协方差收缩（2004）
# 比样本协方差更稳健，专业组合优化的标配
# ═══════════════════════════════════════════

def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf 协方差收缩估计。

    将样本协方差向"常数相关"目标收缩，减少估计误差。
    适用于股票数量 > 样本量时的稳健协方差估计。

    返回: 收缩后的协方差矩阵 (n_assets × n_assets)
    """
    n, p = returns.shape  # n=时间, p=资产

    # 样本协方差
    S = returns.cov().values

    # 目标矩阵：常数相关模型
    # 每个元素 = mean_var * sqrt(var_i * var_j) * mean_corr
    stds = np.sqrt(np.diag(S))
    corr = S / np.outer(stds, stds + 1e-10)
    # 平均相关系数（非对角线均值）
    mask = ~np.eye(p, dtype=bool)
    mean_corr = corr[mask].mean() if mask.sum() > 0 else 0
    F = np.outer(stds, stds) * mean_corr
    np.fill_diagonal(F, np.diag(S))  # 对角线用样本方差

    # 收缩强度（简化版：OLS 估计）
    # pi = sum of asymptotic variances
    diff = S - F
    pi = (diff ** 2).sum().sum() / p

    # rho = sum of covariances
    rho = 0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            rho += (corr[i, j] - mean_corr) ** 2
    rho = rho / (p * (p - 1)) if p > 1 else 0

    # 收缩系数
    shrinkage = min(pi / (rho + 1e-10), 1.0)

    # 收缩后的协方差
    result = (1 - shrinkage) * S + shrinkage * F
    return result


# ═══════════════════════════════════════════
# PCA 因子正交化
# ═══════════════════════════════════════════

def pca_orthogonalize(factor_df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    """
    PCA 提取正交主成分，去除因子间冗余。

    34 个因子中有 32 对高度相关 → PCA 提取 5 个正交主成分。
    每个主成分是原始因子的线性组合，彼此独立。
    """
    from numpy.linalg import eigh

    # 标准化
    X = (factor_df - factor_df.mean()) / (factor_df.std() + 1e-10)
    X = X.fillna(0).values

    # 特征分解
    cov = X.T @ X / (len(X) - 1)
    eigenvalues, eigenvectors = eigh(cov)

    # 取最大的 n_components 个
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, idx]

    # 投影
    scores = X @ components
    result = pd.DataFrame(
        scores,
        index=factor_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    # 报告解释方差
    explained = eigenvalues[idx] / eigenvalues.sum()
    logger.info(f"PCA: {n_components} components, explained variance: "
                f"{', '.join(f'PC{i+1}={explained[i]:.1%}' for i in range(n_components))}")

    return result


# ═══════════════════════════════════════════
# 截面分析器
# ═══════════════════════════════════════════

class CrossSectionAnalyzer:
    """
    截面因子分析器。

    输入: 多股票面板数据（每行=日期×股票，含 close/volume 等）
    输出: 每日股票排名 + 截面 IC 分析 + 最优组合构建
    """

    def __init__(self, panel: pd.DataFrame):
        """
        panel: DataFrame with date, symbol, close, volume, open, high, low
        """
        self.panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.features: pd.DataFrame = None
        self.ic_results: list = []
        self.rankings: pd.DataFrame = None

    def compute_factors(self, winsorize: bool = True) -> pd.DataFrame:
        """
        计算截面因子。

        对每个 date × symbol，计算：
          - 动量: ret_5d, ret_20d, ret_60d
          - 波动: vol_20d
          - 流动性: vol_chg, turnover_5d
          - 价格位置: ma20_dev

        winsorize=True → 对每个因子做 1%/99% 截尾
        """
        df = self.panel.copy()
        df = df.sort_values(["symbol", "date"])

        df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
        df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5)
        df["ret_20d"] = df.groupby("symbol")["close"].pct_change(20)
        df["ret_60d"] = df.groupby("symbol")["close"].pct_change(60)
        df["vol_20d"] = df.groupby("symbol")["ret_1d"].transform(lambda x: x.rolling(20).std())
        df["vol_chg"] = df.groupby("symbol")["volume"].transform(
            lambda x: x / x.rolling(20).mean() - 1)
        df["ma20_dev"] = df["close"] / df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(20).mean()) - 1
        if "turnover" in df.columns:
            df["turnover_5d"] = df.groupby("symbol")["turnover"].transform(
                lambda x: x.rolling(5).mean())
        if "amount" in df.columns:
            df["dollar_vol"] = df["close"] * df["amount"] / df["volume"].replace(0, 1)

        df = df.dropna()
        factor_cols = [c for c in df.columns if c not in
                       ("date", "symbol", "close", "open", "high", "low", "volume", "amount",
                        "turnover", "ret_1d")]

        if winsorize:
            for col in factor_cols:
                lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
                df[col] = df[col].clip(lo, hi)

        self.features = df
        logger.info(f"Computed {len(factor_cols)} cross-sectional factors on {df['symbol'].nunique()} stocks")
        return df

    def rank_factors(self) -> pd.DataFrame:
        """
        横截面上每日排名（Rank 化）。

        排名比原始值更稳健，不受异常值影响。
        rank ∈ [0, 1]，1=最好，0=最差。
        """
        if self.features is None:
            self.compute_factors()

        factor_cols = [c for c in self.features.columns if c not in
                       ("date", "symbol", "close", "open", "high", "low", "volume", "amount",
                        "turnover", "ret_1d")]

        df = self.features.copy()
        for col in factor_cols:
            df[col + "_rank"] = df.groupby("date")[col].rank(pct=True)

        self.rankings = df
        return df

    def neutralize(self, factor_col: str) -> pd.Series:
        """
        行业中性化（简化版：用市值/价格代理）。

        对因子值做截面回归：factor = alpha + beta * market_cap_proxy + residual。
        残差即为纯 alpha（已剥离规模效应）。
        """
        if self.rankings is None:
            self.rank_factors()

        df = self.rankings.copy()
        results = pd.Series(index=df.index, dtype=float)

        for date, group in df.groupby("date"):
            y = group[factor_col].values
            # 用 ma20_dev 做市值代理（大盘股偏离均线小，小盘股波动大）
            X = np.column_stack([np.ones(len(y)), group["ma20_dev"].fillna(0).values])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residual = y - X @ beta
            except np.linalg.LinAlgError:
                residual = y - y.mean()
            results[group.index] = residual

        return results

    def calc_ic(self, forward_days: int = 1) -> list[dict]:
        """
        计算截面 IC（Spearman 秩相关）。

        对每个交易日，计算"因子排名"与"次日收益"的截面相关性。
        IC > 0.03 = 弱，> 0.05 = 中等，> 0.10 = 强。
        """
        if self.rankings is None:
            self.rank_factors()

        df = self.rankings.copy()
        rank_cols = [c for c in df.columns if c.endswith("_rank")]
        results = []

        # Pre-compute forward return per stock
        df = df.sort_values(["symbol", "date"])
        df["fwd_ret"] = df.groupby("symbol")["ret_1d"].shift(-forward_days)

        for col in rank_cols:
            ics = []
            for _, group in df.groupby("date"):
                valid = group[col].notna() & group["fwd_ret"].notna()
                if valid.sum() < 10:
                    continue
                # Spearman = Pearson on ranks (no scipy needed)
                x = group.loc[valid, col].rank()
                y = group.loc[valid, "fwd_ret"].rank()
                ic = x.corr(y)
                if not np.isnan(ic):
                    ics.append(ic)

            if ics:
                mean_ic = np.mean(ics)
                ic_ir = mean_ic / (np.std(ics) + 1e-10)
                factor_name = col.replace("_rank", "")
                results.append({
                    "factor": factor_name, "mean_ic": round(mean_ic, 4),
                    "ic_ir": round(ic_ir, 4), "n_days": len(ics),
                    "significant": abs(ic_ir) > 1.0,
                })

        results.sort(key=lambda x: abs(x["mean_ic"]), reverse=True)
        self.ic_results = results
        return results

    def composite_score(self, ic_weighted: bool = True, neutralized: bool = True) -> pd.DataFrame:
        """
        综合评分：每日输出股票从好到差的排名。

        ic_weighted=True → 按 IC 质量加权各因子
        neutralized=True → 做截面中性化
        """
        if self.rankings is None:
            self.rank_factors()

        if not self.ic_results:
            self.calc_ic()

        rank_cols = [c for c in self.rankings.columns if c.endswith("_rank")]
        df = self.rankings.copy()

        if ic_weighted and self.ic_results:
            # IC/IR 加权：IC 高的因子权重高
            weights = {}
            for r in self.ic_results:
                name = r["factor"] + "_rank"
                w = abs(r["mean_ic"]) * max(0, r["ic_ir"])
                weights[name] = w
            total_w = sum(weights.values()) + 1e-10
            df["score"] = sum(df[c].fillna(0.5) * weights.get(c, 0) / total_w
                             for c in rank_cols)
        else:
            # 等权
            df["score"] = df[rank_cols].mean(axis=1)

        if neutralized:
            df["score"] = self.neutralize("score")

        # 每日排名：score 越高 = 越好
        df["rank"] = df.groupby("date")["score"].rank(pct=True)

        return df[["date", "symbol", "score", "rank"] + rank_cols]

    def report(self) -> str:
        """文本报告"""
        if not self.ic_results:
            self.calc_ic()

        lines = [
            "=" * 60,
            "  Cross-Sectional Factor Report",
            "=" * 60,
            f"  Stocks: {self.panel['symbol'].nunique()} | Days: {self.panel['date'].nunique()}",
            f"  Factors: {len(self.ic_results)}",
            "─" * 60,
            f"  {'Factor':<18} {'IC':>7} {'IR':>6} {'Sig':>5}",
            "  " + "-" * 40,
        ]
        for r in self.ic_results[:8]:
            lines.append(f"  {r['factor']:<18} {r['mean_ic']:>+7.4f} {r['ic_ir']:>+6.2f} "
                        f"{'YES' if r['significant'] else 'no':>5}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/cross_section.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.multi_stock import fetch_multi_stock_daily, build_panel

    print("Fetching 10 stocks for quick test...")
    data = fetch_multi_stock_daily(n_stocks=10, start="20240101")
    if data:
        panel = build_panel(data)

        csa = CrossSectionAnalyzer(panel)
        csa.compute_factors()
        csa.rank_factors()
        csa.calc_ic()
        print(csa.report())

        rankings = csa.composite_score()
        print(f"\nTop 3 stocks on {rankings['date'].max().date()}:")
        latest = rankings[rankings["date"] == rankings["date"].max()]
        for _, row in latest.nlargest(3, "score").iterrows():
            print(f"  {row['symbol']}: score={row['score']:.3f} rank={row['rank']:.1%}")
