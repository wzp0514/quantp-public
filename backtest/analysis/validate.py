"""
回测验证 — 样本外/跨品种/参数稳健性

不做验证的回测 = 看着后视镜开车。核心验证三项：

1. 样本外测试：训练期开发和调参，然后用完全不同时期的验证期数据跑。
   验证期收益衰减 ≤ 30% 才算通过。

2. 跨品种验证：在沪深 300 上有效的策略，换到中证 500 上再跑一遍。
   只在单一品种上有效 = 过拟合。

3. 参数稳健性：把你调好的参数稍微改动一下（比如均线 20→19/21），
   结果变化不应太大。

每次验证返回 dict，包含通过/未通过标记和详细说明。
"""

import logging
import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("validate")


# ============================================================
# 0. 衰退率硬过滤（P2-2 改造4）
# ============================================================

def check_decay_rate(train_sharpe: float, val_sharpe: float = None) -> dict:
    """
    验证集 vs 训练集夏普衰退率检测。

    验证集夏普/训练集夏普 < 0.5 → overfit，直接拒绝。
    """
    if train_sharpe <= 0:
        return {"overfit": True, "decay": 1.0, "reason": "训练集夏普≤0，策略无效"}
    if val_sharpe is None:
        return {"overfit": False, "decay": 0.0, "reason": "无验证集数据"}

    decay = max(0, (train_sharpe - val_sharpe) / train_sharpe)
    overfit = decay > 0.5

    return {
        "overfit": overfit,
        "decay": round(decay, 4),
        "train_sharpe": round(train_sharpe, 4),
        "val_sharpe": round(val_sharpe, 4),
        "reason": f"验证集夏普衰退{decay:.1%}{'>50%疑似过拟合' if overfit else '，通过'}",
    }


# ============================================================
# 0.5. FWER/FDR 多重比较校正（P2-2 改造5）
# ============================================================

def fwer_control(results: list[dict], n_total: int, method: str = "fdr") -> dict:
    """
    多重比较校正。

    默认使用 FDR (Benjamini-Hochberg) 控制错误发现率。
    可选 'bonferroni' 使用更保守的 FWER 控制。

    200 个组合 → bonferroni_threshold = 0.05/200 = 0.00025。
    只有 p 值足够小的策略才标 fwer_pass=True。
    """
    if n_total < 1:
        n_total = 1

    if method == "bonferroni":
        p_threshold = 0.05 / n_total
        passed = 0
        for r in results:
            if isinstance(r, dict):
                p = r.get("p_value", 1.0)
                r["fwer_pass"] = p < p_threshold
                r["fwer_threshold"] = p_threshold
            else:
                p = getattr(r, "p_value", 1.0)
            if p < p_threshold:
                passed += 1
        return {"threshold": p_threshold, "passed": passed, "total": n_total, "method": "bonferroni"}

    # 默认 FDR: Benjamini-Hochberg
    p_values = []
    for r in results:
        p = r.get("p_value", 1.0) if isinstance(r, dict) else getattr(r, "p_value", 1.0)
        p_values.append(p)
    fdr_result = fdr_control(p_values, alpha=0.05)
    for i, r in enumerate(results):
        if isinstance(r, dict):
            r["fwer_pass"] = i in fdr_result["significant_ids"]
            r["fwer_threshold"] = fdr_result["threshold"]
    return {"threshold": fdr_result["threshold"], "passed": fdr_result["n_rejected"],
            "total": n_total, "method": "fdr"}


# ============================================================
# 0.6. FDR 多重比较校正（Benjamini-Hochberg, C20）
# ============================================================

def fdr_control(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    FDR (False Discovery Rate) 多重比较校正 — Benjamini-Hochberg 方法。

    N 较大时（数百次试验），Bonferroni 过于保守（阈值 = 0.05/N）。
    BH 方法控制期望错误发现率而非犯任意一个错误的概率，检验效力更高。

    步骤：
      1. p值升序排列: p₁ ≤ p₂ ≤ ... ≤ pₙ
      2. 计算 BH 临界值: (i/N) * alpha
      3. 找到最大 k 使得 p_k ≤ (k/N) * alpha
      4. 拒绝 p₁ 到 p_k 对应的所有原假设

    返回
    -------
    dict: {significant_ids, threshold, n_rejected, method}
    """
    n = len(p_values)
    if n == 0:
        return {"significant_ids": [], "threshold": 0, "n_rejected": 0, "n_total": 0}

    # 排序 p 值，记录原始索引
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_p = [p for _, p in sorted_pairs]

    # 找最大 k
    k = 0
    for i, p in enumerate(sorted_p, 1):
        threshold_i = (i / n) * alpha
        if p <= threshold_i:
            k = i
        else:
            break

    # k 之前的所有项 = 拒绝原假设
    significant_ids = [idx for idx, _ in sorted_pairs[:k]]
    threshold = (k / n) * alpha if k > 0 else 0

    return {
        "significant_ids": significant_ids,
        "threshold": round(threshold, 6),
        "n_rejected": k,
        "n_total": n,
        "method": "Benjamini-Hochberg",
        "alpha": alpha,
    }


# ============================================================
# 1. 样本外验证
# ============================================================

def out_of_sample_test(
    strategy_class,
    df: pd.DataFrame,
    split_date: str,
    **strategy_params,
) -> dict:
    """
    样本外验证：用训练期参数跑验证期数据，看收益衰减多少。

    参数
    ----------
    strategy_class : bt.Strategy 子类
    df : DataFrame
        完整数据（必须包含 date 列）
    split_date : str
        分割日期，"YYYY-MM-DD"，此前为训练期，此后为验证期
    **strategy_params :
        策略参数

    返回
    -------
    dict，包含：
        - passed: bool（是否通过）
        - decay: float（验证期 vs 训练期收益衰减比）
        - train_return: float
        - test_return: float
        - detail: str
    """
    from backtest.engine.bt_runner import run_backtest

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        return {"passed": False, "detail": f"数据不足：训练期{len(train_df)}条, 验证期{len(test_df)}条"}

    logger.info(f"样本外验证: 训练期 {train_df['date'].min()} ~ {train_df['date'].max()} "
                f"({len(train_df)}条) → 验证期 {test_df['date'].min()} ~ {test_df['date'].max()} "
                f"({len(test_df)}条)")

    # 训练期回测
    train_result = run_backtest(strategy_class, train_df, **strategy_params)
    train_return = train_result["annual_return"]
    train_sharpe = train_result.get("sharpe")

    # 验证期回测（用同样的参数）
    test_result = run_backtest(strategy_class, test_df, **strategy_params)
    test_return = test_result["annual_return"]

    # 收益衰减 = (训练期年化 - 验证期年化) / 训练期年化
    # 衰减 ≤ 30% 才及格
    if train_return > 0:
        decay = (train_return - test_return) / train_return
    else:
        decay = 1.0  # 训练期都亏钱，不用看衰减了

    passed = decay <= 0.30

    detail = (
        f"训练期年化: {train_return:.2%}, "
        f"验证期年化: {test_return:.2%}, "
        f"衰减: {decay:.1%} "
        f"({'通过' if passed else '未通过——衰减超30%，可能过拟合'})"
    )

    logger.info(detail)

    return {
        "passed": passed,
        "decay": decay,
        "train_return": train_return,
        "test_return": test_return,
        "train_sharpe": train_sharpe,
        "detail": detail,
    }


def cross_symbol_test(
    strategy_class,
    symbol_results: dict,
    **strategy_params,
) -> dict:
    """
    跨品种验证：同一个策略在不同品种上分别跑，比较结果差异。

    参数
    ----------
    strategy_class : bt.Strategy 子类
    symbol_results : dict
        预先跑好的结果，格式: {"沪深300": result1, "中证500": result2}
    **strategy_params :
        策略参数

    返回
    -------
    dict
    """
    if len(symbol_results) < 2:
        return {"passed": False, "detail": "需要至少 2 个品种才能做跨品种验证"}

    returns = []
    details = []
    for name, result in symbol_results.items():
        ret = result.get("annual_return", 0)
        returns.append(ret)
        details.append(f"{name}: {ret:.2%}")

    # 标准：所有品种收益都应该为正，且差异不太大
    all_positive = all(r > 0 for r in returns)
    max_diff = max(returns) - min(returns)
    passed = all_positive and max_diff < 0.20  # 品种间收益差异 < 20%

    detail = " | ".join(details)
    detail += f" | 差异: {max_diff:.2%}"
    if not passed:
        if not all_positive:
            detail += " | 未通过——有品种亏损"
        else:
            detail += " | 未通过——品种间差异过大，可能过拟合"

    logger.info(f"跨品种验证: {detail}")

    return {
        "passed": passed,
        "max_diff": max_diff,
        "returns": returns,
        "detail": detail,
    }


def param_robustness_test(
    strategy_class,
    df: pd.DataFrame,
    param_name: str,
    base_value: float,
    variations: list = None,
    **fixed_params,
) -> dict:
    """
    参数稳健性测试：微小改动参数值，看结果是否稳定。

    例如：均线周期 = 20，分别用 18、19、21、22 跑一遍，
    如果收益变化超过 20% → 参数过拟合。

    增强规则: 微扰后夏普下降超过 30% → stability_warning=True（P2-2 改造6）

    返回新增字段 stability: 0-1 的稳健性评分（1=最稳定）
    """
    from backtest.engine.bt_runner import run_backtest

    if variations is None:
        if isinstance(base_value, int) and base_value > 2:
            variations = [base_value - 2, base_value - 1, base_value + 1, base_value + 2]
        else:
            variations = [
                base_value * 0.9,
                base_value * 0.95,
                base_value * 1.05,
                base_value * 1.1,
            ]

    logger.info(f"参数稳健性测试: {param_name}={base_value}, 测试值={variations}")

    base_params = {param_name: base_value, **fixed_params}
    base_result = run_backtest(strategy_class, df, **base_params)
    base_return = base_result["annual_return"]
    base_sharpe = base_result.get("sharpe") or 0

    var_returns = []
    var_sharpes = []
    for v in variations:
        params = {param_name: v, **fixed_params}
        r = run_backtest(strategy_class, df, **params)
        var_returns.append(r["annual_return"])
        var_sharpes.append(r.get("sharpe") or 0)
        logger.info(f"  {param_name}={v}: 年化收益 {r['annual_return']:.2%}, 夏普={r.get('sharpe', 'N/A')}")

    all_returns = [base_return] + var_returns
    max_deviation = (max(all_returns) - min(all_returns)) / abs(base_return) if base_return != 0 else float("inf")

    passed = max_deviation < 0.20

    # 增强: 微扰后夏普下降 > 30% 标 unstable
    stability_warning = False
    min_sharpe = min(var_sharpes) if var_sharpes else 0
    if base_sharpe > 0 and min_sharpe > 0:
        sharpe_drop = (base_sharpe - min_sharpe) / base_sharpe
        if sharpe_drop > 0.3:
            stability_warning = True

    # 稳定性评分: 1 / (1 + deviation) → 偏离越小分数越高
    stability = round(1.0 / (1.0 + min(max_deviation, 2.0)), 4)

    # 综合失败判断
    overall_passed = passed and not stability_warning

    detail = (
        f"基准 ({param_name}={base_value}): {base_return:.2%}, "
        f"变动范围: {min(all_returns):.2%} ~ {max(all_returns):.2%}, "
        f"最大偏离: {max_deviation:.1%} "
        f"({'通过' if passed else '未通过——参数过拟合'})"
    )
    if stability_warning:
        detail += " | 夏普波动>30%——参数不稳定"

    logger.info(detail)

    return {
        "passed": passed,
        "stability_warning": stability_warning,
        "overall_passed": overall_passed,
        "base_return": base_return,
        "base_sharpe": base_sharpe,
        "var_returns": var_returns,
        "var_sharpes": var_sharpes,
        "max_deviation": max_deviation,
        "stability": stability,
        "detail": detail,
    }


# ============================================================
# 综合验证（一键跑完三项）
# ============================================================

def full_validation(
    strategy_class,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy_params: dict,
    param_name: str = "",
    param_value: float = 0,
) -> dict:
    """
    一站式验证：样本外 + 参数稳健性（如果有指定参数名）

    返回
    -------
    dict，包含三项验证的结果和 overall（是否全部通过）
    """
    results = {}

    # 1. 样本外
    # 合并 train + test 为完整数据
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    split_date = str(test_df["date"].min())[:10]

    oos_result = out_of_sample_test(
        strategy_class, full_df, split_date, **strategy_params
    )
    results["out_of_sample"] = oos_result

    # 2. 参数稳健性（只有提供了参数时）
    if param_name and param_value:
        param_result = param_robustness_test(
            strategy_class, train_df,
            param_name=param_name,
            base_value=param_value,
            **{k: v for k, v in strategy_params.items() if k != param_name},
        )
        results["param_robustness"] = param_result

    # 3. 综合判断
    all_passed = all(
        r.get("passed", False) for r in results.values()
    )
    results["overall"] = "ALL PASSED" if all_passed else "FAILED"

    return results


# ============================================================
# 4. Walk-Forward 分析（专业级滚动窗口验证，参考 QuantConnect/Quantopian）
# ============================================================

def walk_forward_analysis(
    strategy_class,
    df: pd.DataFrame,
    train_window: int = 504,     # 训练窗口（交易日，默认2年=504天）
    test_window: int = 126,      # 测试窗口（交易日，默认6个月=126天）
    step: int = 63,              # 步长（每季度滚动一次）
    **strategy_params,
) -> dict:
    """
    Walk-Forward 滚动窗口验证。

    不是一次 train/test split，而是多次滚动训练→测试：
      窗口1: train[2015-2017] → test[2018H1]
      窗口2: train[2015.25-2017.25] → test[2018H2]
      窗口3: train[2015.5-2017.5] → test[2019H1]
      ...
    汇总所有测试窗口的表现。

    真正稳健的策略在所有窗口都表现一致。随机过拟合的策略窗口间方差极大。

    返回
    -------
    dict: {windows, metrics, passed}
    """
    from backtest.engine.bt_runner import run_backtest

    n = len(df)
    if n < train_window + test_window:
        return {"passed": False, "detail": f"数据不足（{n}条 < {train_window + test_window}）",
                "windows": [], "metrics": {}}

    windows = []
    start = 0
    while start + train_window + test_window <= n:
        train_df = df.iloc[start:start + train_window].copy()
        test_df = df.iloc[start + train_window:start + train_window + test_window].copy()

        try:
            train_r = run_backtest(strategy_class, train_df, **strategy_params)
            test_r = run_backtest(strategy_class, test_df, **strategy_params)
        except Exception:
            start += step
            continue

        windows.append({
            "train_start": str(train_df["date"].min().date()) if "date" in train_df.columns else "",
            "train_end": str(train_df["date"].max().date()) if "date" in train_df.columns else "",
            "test_start": str(test_df["date"].min().date()) if "date" in test_df.columns else "",
            "test_end": str(test_df["date"].max().date()) if "date" in test_df.columns else "",
            "train_ar": train_r.get("annual_return", 0),
            "train_sharpe": train_r.get("sharpe") or 0,
            "test_ar": test_r.get("annual_return", 0),
            "test_sharpe": test_r.get("sharpe") or 0,
        })
        start += step

    if len(windows) < 3:
        return {"passed": False, "detail": f"窗口不足（{len(windows)}个 < 3）",
                "windows": windows, "metrics": {}}

    test_ars = [w["test_ar"] for w in windows]
    test_sharpes = [w["test_sharpe"] for w in windows]
    train_ars = [w["train_ar"] for w in windows]

    # 核心指标
    n_wins = len(windows)
    win_rate = sum(1 for a in test_ars if a > 0) / n_wins          # 正收益窗口比例
    mean_test_ar = np.mean(test_ars)
    std_test_ar = np.std(test_ars, ddof=1)
    consistency = mean_test_ar / (std_test_ar + 1e-10)            # 一致性 = 均值/标准差（越高越好）

    # OOS 衰减
    mean_train_ar = np.mean(train_ars)
    if mean_train_ar > 0:
        wf_decay = (mean_train_ar - mean_test_ar) / mean_train_ar
    else:
        wf_decay = 0

    # 通过标准: 正收益窗口 > 50% AND 一致性 > 0.3 AND 衰减 < 30%
    passed = win_rate >= 0.5 and consistency > 0.3 and wf_decay < 0.30

    metrics = {
        "windows": n_wins,
        "win_rate": round(win_rate, 3),
        "mean_test_ar": round(mean_test_ar, 4),
        "mean_test_sharpe": round(np.mean(test_sharpes), 4),
        "std_test_ar": round(std_test_ar, 4),
        "consistency": round(consistency, 4),
        "wf_decay": round(wf_decay, 4),
        "passed": passed,
    }

    detail = (
        f"Walk-Forward ({n_wins}窗口): "
        f"正收益={win_rate:.0%}, 一致性={consistency:.2f}, "
        f"OOS衰减={wf_decay:.1%}, "
        f"平均测试AR={mean_test_ar:.2%} "
        f"({'通过' if passed else '未通过'})"
    )
    logger.info(detail)

    return {"passed": passed, "detail": detail, "windows": windows, "metrics": metrics}


# ============================================================
# 5. Deflated Sharpe Ratio（Marcos Lopez de Prado, 2014）
# ============================================================

def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_periods: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """
    Deflated Sharpe Ratio — 多重比较下的夏普衰减校正。

    试了 N 个策略后，最高夏普的标准差会增大。
    DSR = P(真实夏普 > 0) after deflating for multiple comparisons.

    参数
    ----------
    sharpe : float
        观测到的夏普比率（年化）
    n_trials : int
        总共测试了多少个策略/参数组合
    n_periods : int
        年化用的周期数（日线=252）
    skewness : float
        收益偏度
    kurtosis : float
        收益峰度

    返回
    -------
    dict: {dsr, deflated_sharpe, significant, ...}

    参考文献
    --------
    Lopez de Prado, M., & Bailey, D. (2014). "The Deflated Sharpe Ratio."
    """
    import math

    # Expected maximum Sharpe from N independent trials
    # E[max(SR)] ≈ sqrt(2 * log(N)) / sqrt(1 + (2*log(N)-1)/(2*n))
    if n_trials <= 1:
        e_max = 0.0
    else:
        e_max = math.sqrt(2 * math.log(n_trials)) * (1 - 0.25 / math.log(n_trials))
        # Scale to annual
        e_max = e_max * math.sqrt(n_periods) / math.sqrt(n_periods)

    # Deflate: SR_def = SR - E[max(SR|H0)]
    deflated_sr = sharpe - e_max * (1.0 / math.sqrt(n_periods / 252))

    # PSR: Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)
    # PSR = Prob(SR > SR_benchmark)
    sr_benchmark = 0.0  # null hypothesis: SR = 0
    numerator = (sharpe - sr_benchmark) * math.sqrt(n_periods - 1)
    denominator = math.sqrt(
        1 - skewness * sharpe + (kurtosis - 1) / 4.0 * sharpe ** 2
    )
    if denominator > 0:
        psr = float(scipy_stats_norm_cdf(numerator / denominator)) if 'scipy' not in str(type(None)) else 0.5
    else:
        psr = 0.5

    # DSR: Deflated version of PSR
    # Z_deflated = (SR - E[max]) * sqrt(T-1) / sqrt(1 - gamma3*SR + ...)
    dsr_numerator = (deflated_sr - sr_benchmark) * math.sqrt(n_periods - 1)
    dsr_denominator = math.sqrt(
        1 - skewness * deflated_sr + (kurtosis - 1) / 4.0 * deflated_sr ** 2
    )
    if dsr_denominator > 0:
        dsr = dsr_numerator / dsr_denominator
    else:
        dsr = 0

    significant = deflated_sr > 0

    return {
        "sharpe": round(sharpe, 4),
        "n_trials": n_trials,
        "e_max_sharpe": round(e_max, 4),
        "deflated_sharpe": round(deflated_sr, 4),
        "dsr_z_score": round(dsr, 4),
        "significant": significant,
        "interpretation": (
            f"DSR={dsr:.3f}: {'显著' if significant else '不显著'} "
            f"(deflated SR={deflated_sr:.3f} > 0 = {'通过' if significant else '未通过'})"
        ),
    }


# ============================================================
# 6. 专业级滚动窗口验证（参考 QuantInsti/Edgeful 2025 最佳实践）
# ============================================================

def rolling_window_validate(
    strategy_class,
    df: pd.DataFrame,
    is_years: float = 2.0,          # IS窗口（年），专业推荐 1-3 年
    oos_months: int = 6,            # OOS窗口（月），专业推荐 3-6 个月
    step_months: int = 1,           # 步长（月），月度重训
    min_oos_windows: int = 20,      # 最少 OOS 窗口数（专业建议 ≥30）
    **strategy_params,
) -> dict:
    """
    专业级滚动窗口验证 —— 模拟真实交易中的定期重训流程。

    不是一次 train/test split，而是模拟真实世界：
    每月用最近 2 年数据重训 → 锁参数 → 在接下来 6 个月 OOS 验证 → 推进。

    关键指标:
      - WFE (Walk-Forward Efficiency) = OOS收益 / IS收益，≥50% 才通过
      - OOS正收益窗口比例 ≥ 50%
      - IS/OOS Sharpe 比 < 2.0（否则严重过拟合）

    参考文献
    ----------
    QuantInsti (2025): Walk-Forward Optimization Introduction
    Edgeful (2025): Are You Backtesting Wrong?
    """
    from backtest.engine.bt_runner import run_backtest

    trading_days_per_year = 252
    trading_days_per_month = 21

    is_bars = int(is_years * trading_days_per_year)
    oos_bars = oos_months * trading_days_per_month
    step_bars = step_months * trading_days_per_month

    n = len(df)
    if n < is_bars + oos_bars:
        return {
            "passed": False,
            "detail": f"数据不足: {n}条 < {is_bars + oos_bars}",
            "windows": [],
            "metrics": {"n_windows": 0, "wfe": 0, "win_rate": 0, "consistency": 0, "sr_decay": 1.0, "mean_oos_ar": 0, "mean_oos_sr": 0},
        }

    windows = []
    start = 0
    while start + is_bars + oos_bars <= n:
        is_df = df.iloc[start:start + is_bars].copy()
        oos_df = df.iloc[start + is_bars:start + is_bars + oos_bars].copy()

        try:
            is_r = run_backtest(strategy_class, is_df, **strategy_params)
            oos_r = run_backtest(strategy_class, oos_df, **strategy_params)
        except Exception:
            start += step_bars
            continue

        is_sr = is_r.get("sharpe") or 0
        oos_sr = oos_r.get("sharpe") or 0
        is_ar = is_r.get("annual_return", 0)
        oos_ar = oos_r.get("annual_return", 0)

        windows.append({
            "is_start": str(is_df["date"].min().date()) if "date" in is_df.columns else "",
            "oos_start": str(oos_df["date"].min().date()) if "date" in oos_df.columns else "",
            "is_sharpe": round(is_sr, 4),
            "oos_sharpe": round(oos_sr, 4),
            "is_ar": round(is_ar, 4),
            "oos_ar": round(oos_ar, 4),
            "sr_ratio": round(oos_sr / is_sr, 4) if is_sr > 0 else 0,
        })
        start += step_bars

    n_win = len(windows)
    if n_win < 5:
        return {
            "passed": False,
            "detail": f"窗口不足: {n_win} < 5",
            "windows": windows,
            "metrics": {"n_windows": n_win, "wfe": 0, "win_rate": 0, "consistency": 0, "sr_decay": 1.0, "mean_oos_ar": 0, "mean_oos_sr": 0},
        }

    oos_ars = [w["oos_ar"] for w in windows]
    oos_srs = [w["oos_sharpe"] for w in windows]
    is_ars = [w["is_ar"] for w in windows]

    mean_oos_ar = np.mean(oos_ars)
    mean_is_ar = np.mean(is_ars)
    mean_oos_sr = np.mean(oos_srs)

    wfe = mean_oos_ar / mean_is_ar if mean_is_ar > 0 else 0  # Walk-Forward Efficiency
    win_rate = sum(1 for a in oos_ars if a > 0) / n_win       # OOS正收益比例
    consistency = mean_oos_ar / (np.std(oos_ars, ddof=1) + 1e-10)  # OOS一致性

    # SR 衰减比（>2.0 = 严重过拟合）
    is_srs = [w["is_sharpe"] for w in windows if w["is_sharpe"] > 0]
    oos_srs_filtered = [w["oos_sharpe"] for w in windows if w["is_sharpe"] > 0]
    sr_decay = (np.mean(is_srs) - np.mean(oos_srs_filtered)) / (np.mean(is_srs) + 1e-10) if is_srs else 1.0

    # 五维通过标准（来自搜索验证的专业阈值）
    passed = (
        wfe >= 0.50 and                    # WFE ≥ 50%
        win_rate >= 0.50 and               # 过半OOS窗口正收益
        sr_decay < 0.50 and                # SR衰减 < 50%
        mean_oos_sr > -0.20 and            # 平均OOS Sharpe > -0.2
        n_win >= min(min_oos_windows, 20)  # 至少足够窗口
    )

    detail = (
        f"Rolling({n_win}w, IS={is_years}y/OOS={oos_months}m/step={step_months}m): "
        f"WFE={wfe:.0%}, WinRate={win_rate:.0%}, SRdecay={sr_decay:.0%}, "
        f"meanOOS_SR={mean_oos_sr:.2f} → {'PASS' if passed else 'FAIL'}"
    )
    logger.info(detail)

    return {
        "passed": passed,
        "detail": detail,
        "windows": windows,
        "metrics": {
            "n_windows": n_win,
            "wfe": round(wfe, 4),
            "win_rate": round(win_rate, 4),
            "consistency": round(consistency, 4),
            "sr_decay": round(sr_decay, 4),
            "mean_oos_ar": round(mean_oos_ar, 4),
            "mean_oos_sr": round(mean_oos_sr, 4),
        },
    }


# ============================================================
# 7. CSCV/PBO — 过拟合概率量化（C21, Bailey et al. 2017）
# ============================================================

def cscv_pbo(
    returns: list[float],
    n_splits: int = 16,
    n_trials: int = 100,
) -> dict:
    """
    CSCV (Combinatorially Symmetric Cross-Validation) + PBO (Probability of Backtest Overfitting).

    核心思想：在所有可能的 IS/OOS 划分中，如果 IS 最优策略在 OOS 表现
    和随机差不多 → 过拟合概率高。

    步骤：
      1. 将收益序列切成 N = 2*n_splits 个等长子矩阵
      2. 随机抽样 n_splits 个为 IS，剩余为 OOS（共 C(N, n_splits) 种组合）
      3. 选 n_trials 组做 Monte Carlo 近似
      4. 对每次试验：排名 IS 表现 → 找到 IS 最优策略在 OOS 中的排名
      5. PBO = P(IS 最优在 OOS 中排名后半段的概率)

    参数
    ----------
    returns : list[float]
        策略收益序列
    n_splits : int
        CSCV 等分份数的一半（总份数 = 2*n_splits）
    n_trials : int
        Monte Carlo 抽样次数

    返回
    -------
    dict: {pbo, overfit_risk, interpretation}
    """
    n = len(returns)
    n_parts = 2 * n_splits

    if n < n_parts * 2:
        return {"pbo": 0.5, "overfit_risk": "数据不足", "n": n, "n_parts": n_parts}

    # 切成 n_parts 等份
    part_size = n // n_parts
    parts = []
    for i in range(n_parts):
        parts.append(returns[i * part_size:(i + 1) * part_size])

    is_best_in_oos_bottom_half = 0

    for _ in range(n_trials):
        # 随机选 n_splits 份为 IS
        indices = list(range(n_parts))
        np.random.shuffle(indices)
        is_idx = set(indices[:n_splits])
        oos_idx = set(indices[n_splits:])

        # IS 表现（平均收益）
        is_perf = []
        for j in range(n_parts):
            if j in is_idx:
                is_perf.append(np.mean(parts[j]))

        # IS 最优 part → 看它在 OOS part 中的排名
        is_best_idx = np.argmax(is_perf)
        oos_perfs = [np.mean(parts[j]) for j in range(n_parts) if j in oos_idx]
        oos_ranked = sorted(oos_perfs, reverse=True)
        is_best_oos_perf = np.mean(parts[is_best_idx]) if is_best_idx in oos_idx else None

        if is_best_oos_perf is not None:
            # 排名（1 = 最好）
            rank = sum(1 for p in oos_ranked if p > is_best_oos_perf) + 1
            if rank > len(oos_ranked) / 2:
                is_best_in_oos_bottom_half += 1

    pbo = is_best_in_oos_bottom_half / n_trials if n_trials > 0 else 0.5

    if pbo < 0.1:
        risk = "低过拟合风险 — 策略稳健"
    elif pbo < 0.3:
        risk = "中等过拟合风险 — 建议额外样本外验证"
    elif pbo < 0.5:
        risk = "高过拟合风险 — 需要更多数据和更严格的验证"
    else:
        risk = "极高过拟合风险 — 策略很可能只是数据挖掘产物"

    return {
        "pbo": round(pbo, 4),
        "overfit_risk": risk,
        "n": n,
        "n_parts": n_parts,
        "n_splits": n_splits,
        "n_trials": n_trials,
        "interpretation": f"PBO={pbo:.3f}: {risk}",
    }


def scipy_stats_norm_cdf(x: float) -> float:
    """正态分布 CDF（避免 scipy 依赖）"""
    import math
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


# ============================================================
# 8. 蒙特卡洛持仓风险评估
# ============================================================

def mc_risk_assess(
    returns: list[float],
    n_paths: int = 2000,
    horizon: int = 20,
    initial_capital: float = 100000,
    confidence_levels: tuple = (0.95, 0.99),
    seed: int = 42,
) -> dict:
    """
    蒙特卡洛持仓风险评估：参数 bootstrap + 路径模拟。

    步骤：
      1. 对历史日收益拟合正态分布（μ, σ）
      2. 参数 bootstrap 生成 n_paths 条价格路径（每条 horizon 天）
      3. 对每条路径计算终值、最大回撤
      4. 从终值分布计算 VaR、CVaR

    参数
    ----------
    returns : list[float]
        策略日收益序列
    n_paths : int
        模拟路径数
    horizon : int
        未来交易日天数
    initial_capital : float
        初始资金
    confidence_levels : tuple
        VaR/CVaR 的置信水平

    返回
    -------
    dict: {var, cvar, max_dd_distribution, path_stats, terminal_values}
    """
    if len(returns) < 20:
        return {"error": "收益数据不足（需要 ≥20 个交易日）", "n": len(returns)}

    rng = np.random.RandomState(seed)
    rets = np.array(returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)

    if sigma == 0:
        return {"error": "收益标准差为 0，无法模拟", "mu": mu}

    # 参数 bootstrap：从正态分布采样
    terminal_values = np.zeros(n_paths)
    max_drawdowns = np.zeros(n_paths)
    path_returns = np.zeros(n_paths)

    for i in range(n_paths):
        sim_rets = rng.normal(mu, sigma, horizon)
        cumulative = np.cumprod(1 + sim_rets)
        path = np.concatenate([[1.0], cumulative])
        terminal_values[i] = path[-1] * initial_capital

        # 最大回撤
        peak = np.maximum.accumulate(path)
        dd = (path - peak) / peak
        max_drawdowns[i] = abs(np.min(dd))

        path_returns[i] = cumulative[-1] - 1

    # VaR & CVaR
    var_results = {}
    cvar_results = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        var_val = np.percentile(terminal_values, alpha * 100)
        var_return = var_val / initial_capital - 1
        var_results[f"var_{int(cl*100)}"] = {
            "level": cl,
            "terminal_value": round(var_val, 2),
            "return": round(var_return, 4),
        }
        cvar_val = terminal_values[terminal_values <= var_val].mean()
        cvar_return = cvar_val / initial_capital - 1
        cvar_results[f"cvar_{int(cl*100)}"] = {
            "level": cl,
            "terminal_value": round(cvar_val, 2),
            "return": round(cvar_return, 4),
        }

    return {
        "method": "参数bootstrap（正态分布）",
        "mu": round(mu, 6),
        "sigma": round(sigma, 6),
        "horizon": horizon,
        "n_paths": n_paths,
        "initial_capital": initial_capital,
        "var": var_results,
        "cvar": cvar_results,
        "terminal_stats": {
            "mean": round(float(np.mean(terminal_values)), 2),
            "median": round(float(np.median(terminal_values)), 2),
            "min": round(float(np.min(terminal_values)), 2),
            "max": round(float(np.max(terminal_values)), 2),
            "std": round(float(np.std(terminal_values)), 2),
        },
        "max_dd_stats": {
            "mean": round(float(np.mean(max_drawdowns)), 4),
            "median": round(float(np.median(max_drawdowns)), 4),
            "p95": round(float(np.percentile(max_drawdowns, 95)), 4),
            "worst": round(float(np.max(max_drawdowns)), 4),
        },
        "return_stats": {
            "mean_return": round(float(np.mean(path_returns)), 4),
            "prob_profit": round(float(np.mean(path_returns > 0)), 4),
        },
    }


def mc_param_sensitivity(
    param_ranges: dict[str, tuple[float, float]],
    eval_fn,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """
    蒙特卡洛参数稳健性分析：对策略参数做随机扰动，评估表现分布。

    参数
    ----------
    param_ranges : dict
        {参数名: (low, high)}，在区间内均匀采样
    eval_fn : callable
        eval_fn(params_dict) → float（如夏普比率），越大越好
    n_samples : int
        采样次数
    seed : int
        随机种子

    返回
    -------
    dict: {samples, mean, std, cv, pct_worse_than_baseline, robust}
    """
    rng = np.random.RandomState(seed)
    param_names = list(param_ranges.keys())
    results = []

    for _ in range(n_samples):
        params = {}
        for name in param_names:
            low, high = param_ranges[name]
            params[name] = rng.uniform(low, high)
        try:
            score = eval_fn(params)
            results.append({"params": params, "score": score})
        except Exception:
            pass

    if not results:
        return {"error": "所有参数组合均失败", "samples": 0}

    scores = [r["score"] for r in results]
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    cv = std_score / abs(mean_score) if mean_score != 0 else float("inf")

    return {
        "samples": len(results),
        "mean_score": round(mean_score, 4),
        "std_score": round(std_score, 4),
        "cv": round(cv, 4),
        "best_score": round(float(np.max(scores)), 4),
        "worst_score": round(float(np.min(scores)), 4),
        "pct_positive": round(float(np.mean(np.array(scores) > 0)), 4),
        "robust": cv < 0.5 and mean_score > 0,
        "robustness_note": (
            "参数稳健（CV<0.5且均值>0）" if cv < 0.5 and mean_score > 0
            else f"参数敏感（CV={cv:.2f}），建议缩小参数范围或增加正则化"
        ),
    }


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/validate.py

def check_strategy_death(
    result: dict,
    strategy_name: str = "",
    backtest_baseline: dict = None,
) -> dict:
    """策略死亡信号检测 — 5 条规则回测阶段提前预警。

    规则:
      1. 连续亏损 >= threshold（默认 5 笔）
      2. 回撤超标（> max_drawdown）
      3. 连续亏损月（>= 3 个月）
      4. 信号频率偏离（实盘 vs 回测 > 50%）
      5. 月收益持续衰减（后半段 < 前半段*50%）

    Returns: {passed, risk_score, risk_level, death_signals, rules, detail}
    """
    if backtest_baseline is None:
        backtest_baseline = {}

    rules = {}
    death_signals = []
    risk_score = 0

    # Rule 1: 连续亏损
    consecutive_losses = result.get("consecutive_losses", 0)
    loss_threshold = backtest_baseline.get("max_consecutive_losses", 5)
    ok = consecutive_losses < loss_threshold
    rules["consecutive_losses"] = {"passed": ok, "current": consecutive_losses,
                                    "threshold": loss_threshold}
    if not ok:
        death_signals.append(f"连续亏损 {consecutive_losses} >= {loss_threshold}")
        risk_score += 2

    # Rule 2: 回撤超标
    dd = abs(result.get("drawdown", 0))
    dd_threshold = backtest_baseline.get("max_drawdown", 0.15)
    ok = dd < dd_threshold
    rules["drawdown"] = {"passed": ok, "current": round(dd, 4), "threshold": dd_threshold}
    if not ok:
        death_signals.append(f"回撤 {dd:.1%} >= {dd_threshold:.0%}")
        risk_score += 2

    # Rule 3: 连续亏损月
    monthly = result.get("monthly_returns", [])
    if monthly:
        max_loss_months = 0
        streak = 0
        for m in monthly:
            if m < 0:
                streak += 1
                max_loss_months = max(max_loss_months, streak)
            else:
                streak = 0
        ok = max_loss_months < 3
        rules["loss_months"] = {"passed": ok, "current": max_loss_months, "threshold": 3}
        if not ok:
            death_signals.append(f"连续亏损月 {max_loss_months} >= 3")
            risk_score += 1
    else:
        rules["loss_months"] = {"passed": True, "detail": "无月度数据"}

    # Rule 4: 信号频率偏离
    bt_freq = backtest_baseline.get("signal_frequency", 0)
    if bt_freq > 0:
        live_freq = result.get("signal_frequency", 0)
        deviation = abs(live_freq - bt_freq) / bt_freq
        ok = deviation < 0.5
        rules["signal_drift"] = {"passed": ok, "deviation": round(deviation, 4)}
        if not ok:
            death_signals.append(f"信号频率偏离 {deviation:.0%} > 50%")
            risk_score += 1
    else:
        rules["signal_drift"] = {"passed": True, "detail": "无回测基准"}

    # Rule 5: 收益衰减
    if len(monthly) >= 3:
        half = len(monthly) // 2
        first = np.mean(monthly[:half])
        second = np.mean(monthly[half:])
        decay = first - second if first > 0 else 0
        ok = decay < first * 0.5
        rules["return_decay"] = {
            "passed": ok,
            "first_half": round(float(first), 4),
            "second_half": round(float(second), 4),
        }
        if not ok:
            death_signals.append("月收益持续衰减")
            risk_score += 1
    else:
        rules["return_decay"] = {"passed": True, "detail": f"数据不足({len(monthly)}月)"}

    passed = risk_score < 3
    risk_level = "高" if risk_score >= 5 else ("中" if risk_score >= 3 else "低")

    detail = (
        f"死亡信号: {len(death_signals)}/{len(rules)} 触发, "
        f"风险分 {risk_score}, 级别 {risk_level}"
    )
    logger.info(f"[{strategy_name or '未知'}] {detail}")

    return {
        "passed": passed,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "death_signals": death_signals,
        "rules": rules,
        "detail": detail,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.engine.bt_runner import run_backtest
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    print("=" * 60)
    print("验证测试：双均线策略 (fast=5, slow=20)")
    print("=" * 60)

    df = fetch_index_daily("沪深300", "20200101", "20260522")

    # 测试1：样本外验证
    print("\n[测试1] 样本外验证（split=2023-01-01）")
    oos = out_of_sample_test(
        MaCrossStrategy, df,
        split_date="2023-01-01",
        fast=5, slow=20,
    )
    print(oos["detail"])

    # 测试2：参数稳健性
    print("\n[测试2] 参数稳健性（slow=20, 测试18/19/21/22）")
    train_df = df[df["date"] < "2023-01-01"]
    param = param_robustness_test(
        MaCrossStrategy, train_df,
        param_name="slow", base_value=20,
        fast=5,
    )
    print(param["detail"])
