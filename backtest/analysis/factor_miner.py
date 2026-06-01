"""
因子挖掘 — CogAlpha 的轻量替代方案

CogAlpha 是一个基于多智能体 LLM 的 Alpha 因子挖掘框架（学术研究级别），
直接集成复杂且需要 GPU/LLM API。

替代方案：系统化计算技术因子 + 评估预测能力（IC分析）+ 组合 Top 因子。

因子清单（按学术分类组织，参考 Fama-French/WorldQuant/AQR 公开文献）：

  动量类 (Momentum, Jegadeesh & Titman 1993):
    - ret_1m / ret_3m / ret_6m / ret_12m: 多周期动量
    - ret_12m_1m: 12月动量减1月(去短期反转, Carhart 1997)
    - momentum_smooth: 平滑动量(EMA of 3m ret)

  波动类 (Volatility, Ang et al. 2006 低波动异象):
    - volatility_1m / volatility_3m: 波动率
    - vol_ratio: 短/长波动比
    - low_vol_premium: -volatility_1m(低波动溢价代理)

  成交量/流动性类 (Liquidity, Amihud 2002):
    - vol_change / vol_trend: 放量/量趋势
    - amihud_illiq: |ret|/volume Amihud非流动性(简化)
    - dollar_volume: 成交额(流动性代理)

  价值/质量类 (Value/Quality, Fama-French 1993/Asness 2019):
    - price_to_ma200: close/MA200-1(趋势偏离,替代估值)
    - earnings_yield_proxy: 1/PE-like基于波动调整收益
    - stability_63d: 低波动=高质量代理(Novy-Marx 2013)

  技术形态类 (Technical, Lo et al. 2000):
    - rsi_14 / ma_dev / bb_position: 传统技术指标
    - zscore_20: (close-MA20)/std 标准化偏离
    - range_ratio: (high-low)/close 日内振幅比
    - ma_ribbon: 多均线发散度(MA5-10-20-60 spread)

  统计/另类类 (Statistical):
    - skewness_63d: 收益率偏度(Harvey & Siddique 2000)
    - price_volume_corr: 量价相关性(技术分析)
    - hurst_approx: Hurst指数近似(趋势vs回归判断)
    - ret_1m_reversal: 短期反转(Jegadeesh 1990)

评估方法：
  IC (Information Coefficient) = 因子值与未来收益的相关系数
  IC > 0.02 → 弱预测力, IC > 0.05 → 中等, IC > 0.10 → 强预测力

用法
--------
>>> from backtest.analysis.factor_miner import FactorMiner
>>> fm = FactorMiner(df)
>>> result = fm.mine()                  # 全因子评估
>>> print(result["top_factors"])        # Top 5
"""

import logging
import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("factor_miner")
def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 OHLCV 数据计算全部 ~34 个因子（12基础 + 22扩展）。

    返回
    -------
    DataFrame: 原始列 + 所有因子列
    """
    df = df.copy()
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1000000, index=df.index))

    # 收益率 winsorization（专业标配：1%/99% 分位数截尾）
    # 防止财报日 15% 涨跌等极端值污染后续数月的信号计算
    # 使用 expanding window 分位数避免前视偏差：每时刻只用[0..t]的数据
    ret = close.pct_change()
    min_window = 252  # 至少1年数据才可靠
    lo = ret.expanding(min_periods=min_window).quantile(0.01)
    hi = ret.expanding(min_periods=min_window).quantile(0.99)
    ret = ret.clip(lo, hi)

    # ═══════════════════════════════════════════
    # 动量类 (Momentum)
    # ═══════════════════════════════════════════
    df["ret_1m"] = close.pct_change(21)
    df["ret_3m"] = close.pct_change(63)
    df["ret_6m"] = close.pct_change(126)
    df["ret_12m"] = close.pct_change(252)                       # 12月动量 (Carhart)
    df["ret_12m_1m"] = close.pct_change(252) - close.pct_change(21)  # 去短期反转
    df["ret_1m_reversal"] = -close.pct_change(21)               # 短期反转 (Jegadeesh)
    df["momentum_smooth"] = df["ret_3m"].ewm(span=5).mean()     # 平滑动量

    # ═══════════════════════════════════════════
    # 波动类 (Volatility)
    # ═══════════════════════════════════════════
    df["volatility_1m"] = ret.rolling(21).std()
    df["volatility_3m"] = ret.rolling(63).std()
    df["vol_ratio"] = df["volatility_1m"] / (df["volatility_3m"] + 1e-10)
    df["low_vol_premium"] = -df["volatility_1m"]                # 低波动溢价

    # ═══════════════════════════════════════════
    # 成交量/流动性类 (Liquidity)
    # ═══════════════════════════════════════════
    vol_ma20 = volume.rolling(20).mean()
    df["vol_change"] = volume / (vol_ma20 + 1) - 1
    df["vol_trend"] = vol_ma20.pct_change(20)
    df["amihud_illiq"] = -(ret.abs() / (volume * close + 1e-10)).rolling(21).mean()  # 非流动性(- = 流动性好)
    df["dollar_volume"] = (close * volume).rolling(21).mean()  # 成交额

    # ═══════════════════════════════════════════
    # 价值/质量类 (Value/Quality)
    # ═══════════════════════════════════════════
    ma200 = close.rolling(200).mean()
    df["price_to_ma200"] = close / (ma200 + 1e-10) - 1          # 偏离长期均线(估值代理)
    df["earnings_yield_proxy"] = (close.pct_change(252) /
                                   (df["volatility_3m"] + 1e-10))  # 收益/波动=质量代理
    df["stability_63d"] = -ret.rolling(63).std()                # 收益稳定性(低波=高质量)

    # ═══════════════════════════════════════════
    # 技术形态类 (Technical Patterns)
    # ═══════════════════════════════════════════
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ma20 = close.rolling(20).mean()
    df["ma_dev"] = close / ma20 - 1
    std20 = close.rolling(20).std()
    df["zscore_20"] = (close - ma20) / (std20 + 1e-10)          # 标准化偏离
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_position"] = (close - lower) / (upper - lower + 1e-10)
    df["range_ratio"] = (high - low) / (close + 1e-10)          # 日内振幅比
    # 均线发散度
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma60 = close.rolling(60).mean()
    df["ma_ribbon"] = ((ma5 - ma10) + (ma10 - ma20) + (ma20 - ma60)) / (close + 1e-10)

    # ═══════════════════════════════════════════
    # 统计/另类类 (Statistical / Alternative)
    # ═══════════════════════════════════════════
    df["skewness_63d"] = ret.rolling(63).skew()                 # 收益偏度
    # 量价相关性: corr = cov(ret, vol_chg) / (std(ret) * std(vol_chg))
    vol_chg = volume.pct_change()
    cov_pv = (ret * vol_chg).rolling(20).mean() - ret.rolling(20).mean() * vol_chg.rolling(20).mean()
    std_prod = ret.rolling(20).std() * vol_chg.rolling(20).std()
    df["price_volume_corr"] = cov_pv / (std_prod + 1e-10)
    # Hurst 指数近似 (H > 0.5=趋势, H < 0.5=均值回归)
    high_n = high.rolling(100).max()
    low_n = low.rolling(100).min()
    rng = np.log((high_n - low_n) / (close + 1e-10) + 1e-10)
    df["hurst_approx"] = (rng / np.log(100)).clip(-2, 2)       # Hurst 近似

    # ── 因子值 winsorization（1%/99% 分位数截尾，避免极端值污染后续信号）──
    factor_cols = [c for c in df.columns if c not in
                   ("date", "open", "high", "low", "close", "volume")]
    for col in factor_cols:
        vals = df[col]
        lo = vals.quantile(0.01)
        hi = vals.quantile(0.99)
        if hi > lo:
            df[col] = vals.clip(lo, hi)

    return df


def compute_ic(df: pd.DataFrame, factor_col: str, forward_period: int = 21) -> dict:
    """
    计算单个因子的 IC (Information Coefficient)。

    IC = 因子值(t) 与 未来收益(t+forward_period) 的相关系数。

    参数
    ----------
    df : DataFrame
        含因子列的数据
    factor_col : str
        因子列名
    forward_period : int
        预测未来多少天的收益（默认21=1个月）

    返回
    -------
    dict: {ic, abs_ic, ic_ir, interpretation}
    """
    factor = df[factor_col].dropna()
    forward_ret = df["close"].pct_change(forward_period).shift(-forward_period)

    # 对齐索引
    common_idx = factor.index.intersection(forward_ret.dropna().index)
    if len(common_idx) < 30:
        return {"ic": 0, "abs_ic": 0, "ic_ir": 0, "interpretation": "数据不足"}

    f = factor[common_idx]
    r = forward_ret[common_idx]

    # Pearson 相关系数
    ic = f.corr(r)
    abs_ic = abs(ic)

    # IC_IR = IC均值 / IC标准差（衡量稳定性）
    # 滚动IC
    rolling_ic = []
    for i in range(0, len(f) - 21, 21):
        if i + 21 < len(f):
            rolling_ic.append(f.iloc[i:i+21].corr(r.iloc[i:i+21]))

    if rolling_ic:
        ic_ir = np.mean(rolling_ic) / (np.std(rolling_ic) + 1e-10)
    else:
        ic_ir = 0

    if abs_ic > 0.10:
        interp = "强预测力 — 推荐纳入多因子模型"
    elif abs_ic > 0.05:
        interp = "中等预测力 — 可纳入"
    elif abs_ic > 0.02:
        interp = "弱预测力 — 辅助参考"
    else:
        interp = "无预测力 — 不推荐"

    return {
        "factor": factor_col,
        "ic": round(ic, 4),
        "abs_ic": round(abs_ic, 4),
        "ic_ir": round(ic_ir, 4),
        "interpretation": interp,
    }


# ============================================================
# 专业级因子分析（参考 WorldQuant/AQR 方法）
# ============================================================

def compute_rank_ic(df: pd.DataFrame, factor_col: str, forward_period: int = 21) -> dict:
    """
    Rank IC — Spearman 秩相关（用 Pandas 原生实现，不依赖 scipy）。
    比 Pearson IC 更稳健，不受异常值影响，是业界标准。
    """
    factor = df[factor_col].dropna()
    forward_ret = df["close"].pct_change(forward_period).shift(-forward_period)

    common_idx = factor.index.intersection(forward_ret.dropna().index)
    if len(common_idx) < 30:
        return {"rank_ic": 0, "p_value": 1.0}

    # Spearman: Pearson on ranks
    f = factor[common_idx].rank()
    r = forward_ret[common_idx].rank()
    ic = f.corr(r)
    return {"rank_ic": round(ic, 4), "p_value": round(1.0, 4)}


def ic_decay_analysis(df: pd.DataFrame, factor_col: str) -> dict:
    """
    IC 衰减分析 — 预测力能持续多久？
    计算因子对 1/5/10/21/63 日后的收益率预测能力。
    衰减越快 = 因子越不稳定 = 需要更高频的交易。
    """
    periods = [1, 5, 10, 21, 63]
    decay = {}
    for p in periods:
        ic = compute_ic(df, factor_col, forward_period=p)
        ric = compute_rank_ic(df, factor_col, forward_period=p)
        decay[p] = {"ic": ic["abs_ic"], "rank_ic": ric["rank_ic"]}
    return {"factor": factor_col, "decay": decay}


def factor_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    因子相关性矩阵 — 识别冗余因子。
    两个高度相关的因子（|corr|>0.7）只保留 IC 更高的那个。
    """
    factor_cols = [c for c in df.columns if c not in
                   ("date", "open", "high", "low", "close", "volume")]
    if len(factor_cols) < 2:
        return pd.DataFrame()
    corr = df[factor_cols].corr()
    return corr


def neutralize_market_beta(df: pd.DataFrame, factor_col: str) -> pd.Series:
    """
    市场 beta 中性化 — 剔除因子中的市场收益部分。
    对单指数标的，用收益率做线性回归取残差，保留纯 alpha。
    残差 = 因子值 - beta * 市场收益。
    """
    factor = df[factor_col].dropna()
    mkt_ret = df["close"].pct_change(21).reindex(factor.index)

    common_idx = factor.index.intersection(mkt_ret.dropna().index)
    if len(common_idx) < 30:
        return pd.Series(index=df.index, dtype=float)

    f = factor[common_idx]
    m = mkt_ret[common_idx]

    # OLS: factor = alpha + beta * market_return + epsilon
    X = np.vstack([np.ones(len(m)), m.values]).T
    try:
        beta = np.linalg.lstsq(X, f.values, rcond=None)[0]
        residual = f.values - (beta[0] + beta[1] * m.values)
    except np.linalg.LinAlgError:
        residual = f.values

    result = pd.Series(index=df.index, dtype=float)
    result[common_idx] = residual
    return result


# ============================================================
def neutralize_industry(df: pd.DataFrame, factor_col: str,
                        industry_map: dict = None) -> pd.Series:
    """
    行业中性化 — 剔除因子中的行业暴露部分。

    对单股票数据（有 industry 列）或面板数据（有 industry_map），
    用行业虚拟变量回归取残差，保留纯行业内 alpha。

    参数
    ----------
    df : DataFrame
        含因子列和可选的 industry 列
    factor_col : str
        因子列名
    industry_map : dict, optional
        {symbol: industry_name} 映射。若未提供且 df 有 industry 列，则使用该列。

    返回
    -------
    Series: 行业中性化后的因子值（残差）
    """
    factor = df[factor_col].dropna()
    if len(factor) < 30:
        return pd.Series(index=df.index, dtype=float)

    # 获取行业标签
    if industry_map is not None and "symbol" in df.columns:
        industries = df["symbol"].map(industry_map)
    elif "industry" in df.columns:
        industries = df["industry"]
    else:
        # 无行业数据，返回原值
        logger.debug(f"无行业分类数据，跳过行业中性化 for {factor_col}")
        return factor

    # 行业虚拟变量
    industry_dummies = pd.get_dummies(industries, drop_first=True)
    common_idx = factor.index.intersection(industry_dummies.dropna().index)
    if len(common_idx) < 30:
        return pd.Series(index=df.index, dtype=float)

    f = factor[common_idx]
    X = industry_dummies.loc[common_idx].values
    X = np.column_stack([np.ones(len(X)), X])  # add intercept

    try:
        beta = np.linalg.lstsq(X, f.values, rcond=None)[0]
        residual = f.values - X @ beta
    except np.linalg.LinAlgError:
        residual = f.values

    result = pd.Series(index=df.index, dtype=float)
    result[common_idx] = residual
    return result


def neutralize_industry_and_market(df: pd.DataFrame, factor_col: str,
                                   industry_map: dict = None) -> pd.Series:
    """
    行业+市值双重中性化：先行业虚拟变量回归，再对残差做市场 beta 中性化。
    因子纯净度最高的处理方式。
    """
    industry_neutral = neutralize_industry(df, factor_col, industry_map)
    if industry_neutral.dropna().empty:
        return industry_neutral
    temp_df = df.copy()
    temp_df[factor_col] = industry_neutral
    return neutralize_market_beta(temp_df, factor_col)


# ============================================================
# 因子挖掘器（增强版）
# ============================================================

class FactorMiner:
    """因子挖掘器 — 含专业级分析（Rank IC/IC衰减/相关性/中性化）"""

    def __init__(self, df: pd.DataFrame, max_turnover: float = 0.7):
        self.df = compute_factors(df)
        self.results = []
        self._ic_decay_results = {}
        self.max_turnover = max_turnover
        self._prev_score = None

    def mine(self, neutralized: bool = False) -> dict:
        """
        评估全部因子。

        参数
        ----------
        neutralized : bool
            True = 使用市场中性化后的因子值计算 IC（纯 alpha 评估）
        """
        factor_cols = [c for c in self.df.columns if c not in
                       ("date", "open", "high", "low", "close", "volume")]

        logger.info(f"评估 {len(factor_cols)} 个因子 (中性化={neutralized})...")

        results = []
        for col in factor_cols:
            if neutralized:
                # 用中性化后的因子值
                neutralized_values = neutralize_market_beta(self.df, col)
                temp_df = self.df.copy()
                temp_df[col] = neutralized_values
                ic_result = compute_ic(temp_df, col)
            else:
                ic_result = compute_ic(self.df, col)

            if ic_result.get("ic") is not None:
                rank_ic = compute_rank_ic(self.df, col)
                ic_result["rank_ic"] = rank_ic["rank_ic"]
                ic_result["rank_ic_pvalue"] = rank_ic["p_value"]
                results.append(ic_result)

        results.sort(key=lambda x: abs(x["abs_ic"]), reverse=True)
        self.results = results

        top5 = results[:5]
        strong = [r for r in results if r["abs_ic"] > 0.05]
        medium = [r for r in results if 0.02 < r["abs_ic"] <= 0.05]

        # 相关性矩阵
        corr = factor_correlation_matrix(self.df)

        # 冗余检测
        redundant_pairs = []
        if not corr.empty:
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        redundant_pairs.append(
                            (corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3))
                        )

        # ICIR≥0.3 追踪（C19: 行业要求 ICIR≥0.3 为有效因子）
        icir_valid = [r for r in results if abs(r.get("ic_ir", 0)) >= 0.3]

        summary_lines = [
            "=" * 70,
            "  因子挖掘报告 (增强版: Rank IC + 相关性 + ICIR + 中性化)",
            "=" * 70,
            f"  评估因子: {len(results)} 个",
            f"  强预测力(|IC|>0.05): {len(strong)} 个",
            f"  中等(0.02<|IC|≤0.05): {len(medium)} 个",
            f"  弱/无效: {len(results) - len(strong) - len(medium)} 个",
            f"  ICIR≥0.3（有效因子）: {len(icir_valid)} 个",
            f"  中性化: {'是' if neutralized else '否'}",
            f"  冗余因子对(|corr|>0.7): {len(redundant_pairs)} 对",
        ]

        if self.max_turnover is not None and self.max_turnover < 1.0:
            summary_lines.append(f"  [!] 换手率约束: max_turnover={self.max_turnover}（EMA平滑，限制单期变化≤{self.max_turnover*100:.0f}%）")

        if redundant_pairs:
            summary_lines.append("  ── 冗余因子警告 ──")
            for f1, f2, c in redundant_pairs:
                summary_lines.append(f"    {f1} <-> {f2}: r={c}")

        summary_lines.append("─" * 70)
        summary_lines.append(f"  {'':<4} {'因子':<18} {'IC':>7} {'Rank IC':>7} {'IR':>6} {'评价'}")
        summary_lines.append("  " + "-" * 58)
        for i, r in enumerate(top5):
            summary_lines.append(
                f"  {i+1}.  {r['factor']:<18} {r['ic']:>+7.4f} {r.get('rank_ic', 0):>+7.4f} "
                f"{r['ic_ir']:>6.2f}  {r['interpretation'][:12]}"
            )
        summary_lines.append("=" * 70)

        # 建议
        if strong:
            # 从强因子中排除冗余
            selected = _deduplicate_factors(strong, corr)
            summary_lines.append(
                f"\n建议: {len(selected)} 个独立强因子 → 构建多因子模型"
                f"（已排除 {len(strong) - len(selected)} 个冗余因子）"
            )

        summary = "\n".join(summary_lines)
        print(summary)

        return {
            "results": results,
            "top_factors": top5,
            "strong_count": len(strong),
            "medium_count": len(medium),
            "icir_valid_count": len(icir_valid),
            "icir_valid": icir_valid,
            "redundant_pairs": redundant_pairs,
            "correlation_matrix": corr,
            "neutralized": neutralized,
            "summary": summary,
        }

    def ic_decay_report(self, top_n: int = 5) -> dict:
        """对 Top N 因子做 IC 衰减分析"""
        if not self.results:
            self.mine()

        logger.info(f"IC 衰减分析 (Top {top_n})...")
        results = {}
        for r in self.results[:top_n]:
            col = r["factor"]
            decay = ic_decay_analysis(self.df, col)
            results[col] = decay
            self._ic_decay_results[col] = decay

        # 打印衰减表
        periods = [1, 5, 10, 21, 63]
        print("\n" + "=" * 70)
        print("  IC 衰减分析 — 预测力随时间的变化")
        print("=" * 70)
        header = f"  {'因子':<18}"
        for p in periods:
            header += f" {'T+'+str(p):>8}"
        print(header)
        print("  " + "-" * 60)
        for r in self.results[:top_n]:
            col = r["factor"]
            if col in results:
                line = f"  {col:<18}"
                for p in periods:
                    ic_val = results[col]["decay"].get(p, {}).get("rank_ic", 0)
                    line += f" {ic_val:>+8.4f}"
                print(line)
        print("=" * 70)
        print("  解读: |IC| 越大 = 预测力越强; 衰减越快 = 信号越短命")
        print("  衰减慢的因子适合低频策略，衰减快的因子需要高频交易")

        return results

    def multi_factor_score(self, deduplicated: bool = True) -> pd.Series:
        """
        综合因子评分（剔除冗余后）。
        deduplicated=True → 只保留独立强因子，去除高度相关的冗余因子。

        max_turnover 约束 → 用 EMA 平滑降低换手率，限制持仓变化幅度。
        """
        if not self.results:
            self.mine()

        corr = factor_correlation_matrix(self.df)
        strong = [r for r in self.results if r["abs_ic"] > 0.05]

        if deduplicated and strong:
            selected = _deduplicate_factors(strong, corr)
        else:
            selected = strong[:5] if strong else self.results[:5]

        if not selected:
            selected = self.results[:3]

        score = pd.Series(0.5, index=self.df.index)
        weight = 1.0 / len(selected)
        for r in selected:
            col = r["factor"]
            ic_sign = 1 if r["ic"] > 0 else -1
            vals = self.df[col].dropna()
            if vals.std() > 0:
                normalized = (vals - vals.mean()) / vals.std()
                sigmoid = 1 / (1 + np.exp(-ic_sign * normalized))
                score = score * (1 - weight) + sigmoid * weight

        score = score.clip(0, 1)

        # 换手率约束：EMA 平滑限制单期变化幅度
        if self.max_turnover is not None and self.max_turnover < 1.0:
            if self._prev_score is not None and not self._prev_score.empty:
                # 对齐索引
                common_idx = score.index.intersection(self._prev_score.index)
                if len(common_idx) > 0:
                    alpha = self.max_turnover
                    score.loc[common_idx] = (
                        alpha * score.loc[common_idx] + (1 - alpha) * self._prev_score.loc[common_idx]
                    )
            self._prev_score = score.copy()

        return score.clip(0, 1)

    def check_uniqueness(self, name: str, factor_df: pd.DataFrame, threshold: float = 0.7) -> dict:
        """
        新因子入库前独特性检验：与 FactorStore 已有因子做 Pearson 相关。
        任一现有因子相关性 > threshold → 拒绝或降级。

        返回 {"unique": bool, "conflicts": [因子名], "max_corr": float}
        """
        from data.vault.factor_store import FactorStore
        fs = FactorStore()
        existing = fs.list_factors()
        if not existing:
            return {"unique": True, "conflicts": [], "max_corr": 0}

        if "factor_value" not in factor_df.columns:
            return {"unique": True, "conflicts": [], "max_corr": 0}

        new_vals = factor_df["factor_value"].dropna()
        conflicts = []
        max_corr = 0

        for f in existing:
            df_existing = fs.load_factor(f["name"])
            if df_existing is None or "factor_value" not in df_existing.columns:
                continue
            existing_vals = df_existing["factor_value"].dropna()
            # 按 date 对齐
            common = pd.DataFrame({"new": new_vals, "old": existing_vals}).dropna()
            if len(common) < 30:
                continue
            corr = common["new"].corr(common["old"])
            max_corr = max(max_corr, abs(corr))
            if abs(corr) > threshold:
                conflicts.append(f["name"])

        return {"unique": len(conflicts) == 0, "conflicts": conflicts, "max_corr": max_corr}

    def save(self, name: str, factor_df: pd.DataFrame, ic: float = 0, ic_ir: float = 0,
             category: str = "", description: str = "", reject_if_duplicate: bool = True) -> bool:
        """
        保存因子到 FactorStore，入库前自动独特性检验。

        reject_if_duplicate=True → 有冲突时拒绝保存
        reject_if_duplicate=False → 有冲突时降级标注但仍保存
        """
        from data.vault.factor_store import FactorStore

        check = self.check_uniqueness(name, factor_df)
        if check["conflicts"]:
            logger.warning(
                f"因子 '{name}' 与已有因子 {check['conflicts']} 高度相关"
                f"（max_corr={check['max_corr']:.3f} > 0.7）"
            )
            if reject_if_duplicate:
                logger.info(f"因子 '{name}' 拒绝保存（独特性检验失败）")
                return False
            logger.info(f"因子 '{name}' 降级标注后保存")
            description = f"[降级: 与{', '.join(check['conflicts'])}相关] " + description

        fs = FactorStore()
        fs.save_factor(name, factor_df, ic=ic, ic_ir=ic_ir, category=category, description=description)
        return True


def winsorize_mad(series: pd.Series, n_deviations: float = 5.0) -> pd.Series:
    """
    MAD (Median Absolute Deviation) 截断 — 对极端值的鲁棒替代方案。

    MAD = median(|x_i - median(x)|)
    比百分位截断更不受异常值影响，适用于重尾分布。

    参数
    ----------
    series : pd.Series
        待截断的因子值或收益率序列
    n_deviations : float
        允许偏离中位数的 MAD 倍数（默认 5，约等价于 3σ 在正态分布下）

    返回
    -------
    pd.Series: MAD 截断后的序列
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + n_deviations * mad
    lower = median - n_deviations * mad
    return series.clip(lower, upper)


def _deduplicate_factors(factors: list[dict], corr: pd.DataFrame,
                         threshold: float = 0.7) -> list[dict]:
    """
    剔除冗余因子：两个高度相关的因子只保留 |IC| 更大的。
    """
    if len(factors) <= 1:
        return factors

    selected = [factors[0]]  # IC 最高的自动入选
    for f in factors[1:]:
        name = f["factor"]
        is_redundant = False
        for s in selected:
            s_name = s["factor"]
            if name in corr.index and s_name in corr.columns:
                r = abs(corr.loc[name, s_name])
                if r > threshold:
                    is_redundant = True
                    break
        if not is_redundant:
            selected.append(f)

    return selected


# ============================================================
# C11: 分批多进程 + Parquet 缓存 + 增量更新
# ============================================================

import os
from pathlib import Path

_FACTOR_CACHE_DIR = Path("data/vault/factors")
_FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_factors_batched(
    df: pd.DataFrame,
    batch_size: int = 500,
    n_workers: int = 0,
    cache_key: str = "",
) -> pd.DataFrame:
    """
    C11: 分批计算因子（大 DataFrame 时避免内存溢出）+ 可选多进程。

    如果数据量 < batch_size，直接单进程计算；
    超过则分批计算后 concat。
    n_workers > 0 时使用 multiprocessing.Pool（需 pickle 兼容）。
    """
    n = len(df)
    if n <= batch_size or n_workers <= 1:
        return compute_factors(df)

    import multiprocessing as mp
    from functools import partial

    batches = [df.iloc[i:i + batch_size] for i in range(0, n, batch_size)]
    n_procs = min(n_workers, len(batches), mp.cpu_count())

    logger.info(f"分批多进程因子计算: {len(batches)} 批 × {batch_size}条, {n_procs} 进程")
    with mp.Pool(n_procs) as pool:
        results = pool.map(compute_factors, batches)

    result = pd.concat(results, ignore_index=True)
    # 去重（按date列）
    if "date" in result.columns:
        result = result.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return result


def factor_cache_path(symbol: str, period: str = "daily") -> Path:
    """因子缓存文件路径"""
    return _FACTOR_CACHE_DIR / f"{symbol}_{period}_factors.parquet"


def compute_factors_cached(
    df: pd.DataFrame,
    symbol: str,
    use_cache: bool = True,
    incremental: bool = True,
) -> pd.DataFrame:
    """
    C11: 带 Parquet 缓存和增量更新的因子计算。

    增量模式：
      1. 读取已有因子缓存
      2. 找出缓存中不存在的日期（新数据）
      3. 只对新数据计算因子（节省时间）
      4. 合并新旧结果 → 写回缓存

    缓存模式：
      如果缓存存在且 use_cache=True → 直接返回缓存；
      否则全量计算 → 写缓存。
    """
    cache_path = factor_cache_path(symbol)

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if incremental and len(df) > len(cached):
            # 找出新日期
            if "date" in df.columns and "date" in cached.columns:
                new_dates = set(df["date"].values) - set(cached["date"].values)
                if new_dates:
                    new_df = df[df["date"].isin(new_dates)].copy()
                    logger.info(f"增量更新因子: {len(new_df)} 新行 "
                                f"({len(cached)} 缓存 + {len(new_df)} 新)")
                    new_factors = compute_factors(new_df)
                    result = pd.concat([cached, new_factors], ignore_index=True)
                    result = result.drop_duplicates(subset=["date"]).sort_values("date")
                    result.to_parquet(cache_path, index=False)
                    return result.reset_index(drop=True)
            return cached
        # 全量覆盖
        logger.info(f"缓存命中但全量重算: {symbol}, {len(df)} 条")
    else:
        logger.info(f"全量因子计算: {symbol}, {len(df)} 条")

    result = compute_factors(df)
    result.to_parquet(cache_path, index=False)
    logger.info(f"因子缓存已保存: {cache_path} ({len(result)} 行 × {len(result.columns)} 列)")
    return result


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/factor_miner.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    df = fetch_index_daily("沪深300", "20200101", "20250601")
    fm = FactorMiner(df)
    fm.mine()
