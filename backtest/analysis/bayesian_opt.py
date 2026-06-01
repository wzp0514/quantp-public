"""
贝叶斯优化策略参数调优 — 用概率模型替代网格搜索

原理：
  1. 高斯过程（Gaussian Process）对目标函数建模
  2. 采集函数（Expected Improvement）决定下一个采样点
  3. 每次采样都平衡"探索"（试不确定的区域）和"利用"（在已知好区域深挖）
  4. 通常 20-50 次采样就能找到接近最优的参数

对比网格搜索：
  - 网格搜索: 10×10 = 100 次回测（指数增长）
  - 贝叶斯优化: 30-50 次回测（线性增长，与参数维度关系小）

基于 Optuna 框架（Trial-based, pruning, 可视化内置）。

用法
--------
>>> from backtest.analysis.bayesian_opt import BayesianOptimizer
>>> bo = BayesianOptimizer(df, cash=100000)
>>> best = bo.optimize(strategy_class, param_space, n_trials=50)
>>> print(f"最优参数: {best['params']}, 夏普: {best['sharpe']:.2f}")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("bayesian_opt")


class BayesianOptimizer:
    """
    贝叶斯优化策略参数。

    用法
    --------
    >>> bo = BayesianOptimizer(df, cash=100000)
    >>> best = bo.optimize(MaCrossStrategy, {"fast": (3,20), "slow": (15,60)})
    """

    def __init__(self, df: pd.DataFrame, cash: float = 100000.0):
        self.df = df
        self.cash = cash
        self.study = None

    def optimize(
        self,
        strategy_class,
        param_space: dict,
        n_trials: int = 50,
        objective: str = "sharpe",
        direction: str = "maximize",
        timeout: int = 600,
        show_progress: bool = True,
    ) -> dict:
        """
        贝叶斯优化策略参数。

        参数
        ----------
        strategy_class : bt.Strategy
        param_space : dict
            参数空间: {"fast": (3, 20), "slow": (15, 60)}
            或进阶: {"fast": ("int", 3, 20), "slow": ("int", 15, 60)}
        n_trials : int
            采样次数（推荐 30-100）
        objective : str
            优化目标: "sharpe" | "annual_return" | "calmar" | "profit_factor"
        direction : str
            "maximize" | "minimize"
        timeout : int
            最大优化时间（秒）
        show_progress : bool
            是否显示进度条

        返回
        -------
        dict: {params, value, sharpe, drawdown, annual_return, trials}
        """
        import optuna

        from backtest.engine.bt_runner import run_backtest

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective_fn(trial):
            params = {}
            for name, spec in param_space.items():
                if isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], str):
                    # 格式: ("int", low, high) 或 ("float", low, high)
                    kind, low, high = spec
                    if kind == "int":
                        params[name] = trial.suggest_int(name, low, high)
                    elif kind == "float":
                        params[name] = trial.suggest_float(name, low, high)
                    elif kind == "log":
                        params[name] = trial.suggest_float(name, low, high, log=True)
                    elif kind == "categorical":
                        params[name] = trial.suggest_categorical(name, low)  # low = choices list
                elif isinstance(spec, tuple) and len(spec) == 2:
                    low, high = spec
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = trial.suggest_int(name, low, high)
                    else:
                        params[name] = trial.suggest_float(name, low, high)

            r = run_backtest(strategy_class, self.df, initial_cash=self.cash, **params)

            # 剪枝：如果回撤过大，提前终止
            dd = r.get("drawdown", 0)
            if dd > 0.35:
                trial.report(-99, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            trial.set_user_attr("sharpe", r.get("sharpe", 0) or 0)
            trial.set_user_attr("drawdown", r.get("drawdown", 0))
            trial.set_user_attr("annual_return", r.get("annual_return", 0))
            trial.set_user_attr("params", params)

            return self._objective(r, objective)

        # 创建 study
        if direction == "minimize":
            self.study = optuna.create_study(direction="minimize")
        else:
            self.study = optuna.create_study(direction="maximize")

        logger.info(f"贝叶斯优化启动: {n_trials}次采样, 目标={objective}, {len(param_space)}维参数")

        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        best_trial = self.study.best_trial
        result = {
            "params": best_trial.params,
            "value": best_trial.value,
            "sharpe": best_trial.user_attrs.get("sharpe", 0),
            "drawdown": best_trial.user_attrs.get("drawdown", 0),
            "annual_return": best_trial.user_attrs.get("annual_return", 0),
            "n_trials": len(self.study.trials),
        }

        logger.info(
            f"贝叶斯优化完成: 最优={result['value']:.4f}, "
            f"夏普={result['sharpe']:.2f}, 参数={result['params']}"
        )
        return result

    def get_importance(self) -> dict:
        """返回参数重要性排序（超参重要性）"""
        if self.study is None:
            return {}
        import optuna
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}

    def _objective(self, result: dict, metric: str) -> float:
        """目标函数"""
        if metric == "sharpe":
            s = result.get("sharpe", 0)
            return s if s else -99
        elif metric == "annual_return":
            r = result.get("annual_return", -1)
            dd = result.get("drawdown", 1)
            return r / (dd + 0.01)
        elif metric == "calmar":
            r = result.get("annual_return", -1)
            dd = result.get("drawdown", 1)
            return r / dd if dd > 0 else r
        elif metric == "profit_factor":
            trades = result.get("trades_df", pd.DataFrame())
            if trades.empty:
                return -99
            wins = trades[trades["pnl"] > 0]["pnl"].sum()
            losses = abs(trades[trades["pnl"] < 0]["pnl"].sum())
            return wins / losses if losses > 0 else wins
        return 0


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/bayesian_opt.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.strategies.builtin.bollinger import BollingerStrategy

    df = fetch_index_daily("沪深300", "20200101", "20250601")

    bo = BayesianOptimizer(df, cash=100000)
    best = bo.optimize(
        BollingerStrategy,
        param_space={"period": ("int", 10, 50), "devfactor": ("float", 1.0, 3.0)},
        n_trials=30,
    )

    print(f"\n最优参数: {best['params']}")
    print(f"夏普: {best['sharpe']:.2f}, 年化: {best['annual_return']:.2%}, 回撤: {best['drawdown']:.2%}")

    importance = bo.get_importance()
    if importance:
        print(f"\n参数重要性: {importance}")
