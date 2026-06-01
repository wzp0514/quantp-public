"""
L3 组合优化 — 均值方差 / 风险平价 / Factor Timing

用法
--------
>>> from backtest.analysis.portfolio_optimizer import PortfolioOptimizer
>>> opt = PortfolioOptimizer(returns_df)        # returns: T×N 日收益矩阵
>>> w_mv = opt.mean_variance()                   # 最大夏普比
>>> w_rp = opt.risk_parity()                     # 等风险贡献
>>> w_ft = opt.factor_timing(factor_returns_df)  # 因子动量加权
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.log import get_logger

logger = get_logger("portfolio_optimizer")


class PortfolioOptimizer:
    """L3 组合优化器"""

    def __init__(self, returns: pd.DataFrame, risk_free: float = 0.02):
        """
        参数
        ----------
        returns : pd.DataFrame  T×N 日收益率（列=资产）
        risk_free : float       年化无风险利率
        """
        self.returns = returns
        self.rf_daily = risk_free / 252
        self.mu = returns.mean().values * 252
        self.Sigma = returns.cov().values * 252
        self.n = len(returns.columns)
        self.assets = returns.columns.tolist()

    # ═══════════════════════════════════════════
    # 均值方差优化
    # ═══════════════════════════════════════════

    def mean_variance(self, objective: str = "max_sharpe",
                      bounds: tuple = (0.01, 0.40)) -> dict:
        """
        Markowitz 均值方差优化。

        参数
        ----------
        objective : str  优化目标: max_sharpe / min_vol / efficient_frontier
        bounds : tuple   单资产权重上下界
        """
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bnds = [(bounds[0], bounds[1]) for _ in range(self.n)]
        w0 = np.ones(self.n) / self.n

        if objective == "min_vol":
            result = minimize(
                lambda w: np.sqrt(w @ self.Sigma @ w),
                w0, bounds=bnds, constraints=constraints, method="SLSQP"
            )
        elif objective == "max_sharpe":
            result = minimize(
                lambda w: -(w @ self.mu - self.rf_daily * 252) / np.sqrt(w @ self.Sigma @ w),
                w0, bounds=bnds, constraints=constraints, method="SLSQP"
            )
        else:
            raise ValueError(f"未知目标: {objective}")

        if not result.success:
            logger.warning(f"优化未收敛: {result.message}")

        w = result.x
        w = np.clip(w, bounds[0], bounds[1])
        w = w / w.sum()
        return self._summary(w, "mean_variance", objective)

    def efficient_frontier(self, n_points: int = 20,
                           bounds: tuple = (0.01, 0.40)) -> list[dict]:
        """计算有效前沿"""
        points = []
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bnds = [(bounds[0], bounds[1]) for _ in range(self.n)]

        w_min = self.mean_variance("min_vol", bounds)
        ret_min = w_min["expected_return"]
        ret_max = self.mu.max() * 0.8

        for target_ret in np.linspace(ret_min, ret_max, n_points):
            cons = constraints + [
                {"type": "eq", "fun": lambda w, r=target_ret: w @ self.mu - r}
            ]
            result = minimize(
                lambda w: np.sqrt(w @ self.Sigma @ w),
                np.ones(self.n) / self.n, bounds=bnds, constraints=cons, method="SLSQP"
            )
            if result.success:
                w = result.x / result.x.sum()
                points.append(self._summary(w, "efficient_frontier", str(target_ret)))

        return points

    # ═══════════════════════════════════════════
    # 风险平价 (Equal Risk Contribution)
    # ═══════════════════════════════════════════

    def risk_parity(self, bounds: tuple = (0.01, 0.40)) -> dict:
        """
        等风险贡献：每个资产对组合波动率的贡献相等。

        min Σ_i (w_i * (Σw)_i - target_rc)^2
        s.t.  Σw = 1
        """
        def _erc_objective(w):
            port_vol = np.sqrt(w @ self.Sigma @ w)
            marginal_contrib = self.Sigma @ w
            risk_contrib = w * marginal_contrib / port_vol
            target_rc = port_vol / self.n
            return np.sum((risk_contrib - target_rc) ** 2)

        bnds = [(bounds[0], bounds[1]) for _ in range(self.n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(_erc_objective, np.ones(self.n) / self.n,
                          bounds=bnds, constraints=constraints, method="SLSQP")

        w = result.x
        w = np.clip(w, bounds[0], bounds[1])
        w = w / w.sum()
        return self._summary(w, "risk_parity", "equal_risk_contribution")

    # ═══════════════════════════════════════════
    # Factor Timing
    # ═══════════════════════════════════════════

    def factor_timing(self, factor_returns: pd.DataFrame = None,
                      lookback: int = 60, method: str = "momentum") -> dict:
        """
        因子择时：根据因子近期表现动态分配权重。

        参数
        ----------
        factor_returns : pd.DataFrame  K×N 因子收益（如未提供则用资产收益本身）
        lookback : int                 回看窗口（交易日）
        method : str                   择时方法: momentum / vol_target / equal
        """
        if factor_returns is None:
            factor_returns = self.returns

        recent = factor_returns.iloc[-lookback:]
        n_factors = factor_returns.shape[1]

        if method == "momentum":
            score = recent.mean().values
            score = np.maximum(score, 0)
        elif method == "vol_target":
            vol = recent.std().values
            score = 1.0 / np.maximum(vol, 0.0001)
        elif method == "equal":
            w = np.ones(n_factors) / n_factors
            return self._summary(w, "factor_timing", method)
        else:
            raise ValueError(f"未知方法: {method}")

        if score.sum() == 0:
            score = np.ones(n_factors)

        w = score / score.sum()
        w = np.clip(w, 0.01, 0.40)
        w = w / w.sum()
        return self._summary(w, "factor_timing", method)

    # ═══════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════

    def _summary(self, weights: np.ndarray, optimizer: str,
                 objective: str) -> dict:
        w = weights
        port_ret = w @ self.mu
        port_vol = np.sqrt(w @ self.Sigma @ w)
        sharpe = (port_ret - self.rf_daily * 252) / port_vol

        risk_contrib = w * (self.Sigma @ w) / port_vol
        rc_pct = risk_contrib / risk_contrib.sum()

        return {
            "optimizer": optimizer,
            "objective": objective,
            "weights": dict(zip(self.assets, np.round(w, 4))),
            "expected_return": round(float(port_ret), 4),
            "expected_vol": round(float(port_vol), 4),
            "expected_sharpe": round(float(sharpe), 4),
            "risk_contribution": dict(zip(self.assets, np.round(rc_pct, 4))),
            "concentration": round(float(np.sum(w ** 2)), 4),
        }
