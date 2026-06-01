"""
遗传算法策略优化器 — 用进化替代暴力穷举

原理：
  1. 初始化种群（随机生成 N 组参数）
  2. 每组参数跑回测 → 计算适应度（夏普比率 / 年化收益 / 自定义）
  3. 选择（锦标赛选择）→ 交叉（均匀交叉）→ 变异（高斯扰动）
  4. 重复 N 代 → 输出最优个体

对比策略矿工（暴力排列组合）：
  - 矿工: 6×6×N 参数 = O(m×n×p) 穷举
  - 遗传: 种群大小 × 代数 = O(pop×gen) 智能搜索
  - 参数空间 > 1000 组合时, 遗传算法效率远超穷举

用法
--------
>>> from backtest.analysis.genetic_miner import GeneticOptimizer
>>> go = GeneticOptimizer(df, cash=100000)
>>> best = go.evolve(pop_size=30, generations=20)
>>> print(f"最优参数: {best['params']}, 夏普: {best['sharpe']:.2f}")
"""

import logging
import random
import time
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("genetic_miner")


class GeneticOptimizer:
    """
    遗传算法策略优化器。

    用法
    --------
    >>> go = GeneticOptimizer(df, cash=100000)
    >>> best = go.evolve(pop_size=30, generations=20)
    """

    def __init__(self, df: pd.DataFrame, cash: float = 100000.0):
        self.df = df
        self.cash = cash
        self.history: list[dict] = []  # 每代最佳个体

    def evolve(
        self,
        strategy_class,
        param_space: dict,
        pop_size: int = 30,
        generations: int = 20,
        fitness: str = "sharpe",
        elite_size: int = 3,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.1,
        early_stop: int = 5,
    ) -> dict:
        """
        遗传算法优化策略参数。

        参数
        ----------
        strategy_class : bt.Strategy 子类
            策略类
        param_space : dict
            参数空间, 如 {"fast": (3, 20), "slow": (15, 60)}
            int 范围 → 整数, float 范围 → 浮点
        pop_size : int
            种群大小（推荐 20-50）
        generations : int
            最大代数（推荐 10-30）
        fitness : str
            适应度函数: "sharpe" | "annual_return" | "calmar" | "profit_factor"
        elite_size : int
            精英保留数量
        mutation_rate : float
            变异概率（0-1）
        mutation_scale : float
            变异幅度（相对于参数范围的百分比）
        early_stop : int
            连续 N 代无改善则提前终止

        返回
        -------
        dict: {params, fitness_value, sharpe, drawdown, annual_return, generation}
        """
        from backtest.engine.bt_runner import run_backtest

        # 1. 初始化种群
        population = self._init_population(param_space, pop_size)
        best_overall = None
        best_fitness = -float("inf")
        no_improve = 0

        logger.info(f"遗传算法启动: 种群={pop_size}, 代数={generations}, 参数={len(param_space)}维")

        for gen in range(generations):
            t0 = time.time()

            # 2. 评估适应度
            scored = []
            for i, ind in enumerate(population):
                r = run_backtest(strategy_class, self.df, initial_cash=self.cash, **ind)
                score = self._fitness(r, fitness)
                scored.append({
                    "params": ind,
                    "fitness": score,
                    "sharpe": r.get("sharpe", 0) or 0,
                    "drawdown": r.get("drawdown", 0),
                    "annual_return": r.get("annual_return", 0),
                })

            # 排序（适应度越高越好）
            scored.sort(key=lambda x: x["fitness"], reverse=True)
            gen_best = scored[0]

            # 更新全局最优
            if gen_best["fitness"] > best_fitness:
                best_fitness = gen_best["fitness"]
                best_overall = deepcopy(gen_best)
                best_overall["generation"] = gen + 1
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.time() - t0
            logger.info(
                f"  第{gen+1}/{generations}代 | "
                f"最优适应度={gen_best['fitness']:.4f} | "
                f"夏普={gen_best['sharpe']:.2f} | "
                f"参数={gen_best['params']} | "
                f"{elapsed:.1f}s"
            )

            self.history.append(gen_best)

            # 提前终止
            if no_improve >= early_stop:
                logger.info(f"连续{early_stop}代无改善，提前终止")
                break

            # 3. 选择 + 交叉 + 变异 → 下一代
            if gen < generations - 1:
                population = self._next_generation(
                    scored, param_space, pop_size, elite_size,
                    mutation_rate, mutation_scale,
                )

        logger.info(f"遗传算法完成: 最优适应度={best_overall['fitness']:.4f}")
        return best_overall

    def _init_population(self, param_space: dict, size: int) -> list[dict]:
        """随机初始化种群"""
        pop = []
        for _ in range(size):
            ind = {}
            for name, (low, high) in param_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    ind[name] = random.randint(low, high)
                else:
                    ind[name] = random.uniform(low, high)
            pop.append(ind)
        return pop

    def _fitness(self, result: dict, metric: str) -> float:
        """计算适应度"""
        if metric == "sharpe":
            s = result.get("sharpe", 0)
            return s if s else -99
        elif metric == "annual_return":
            r = result.get("annual_return", -1)
            dd = result.get("drawdown", 1)
            return r / (dd + 0.01)  # 惩罚回撤
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

    def _next_generation(
        self, scored: list, param_space: dict,
        pop_size: int, elite_size: int,
        mutation_rate: float, mutation_scale: float,
    ) -> list[dict]:
        """生成下一代种群"""
        new_pop = []

        # 精英保留
        for i in range(min(elite_size, len(scored))):
            new_pop.append(deepcopy(scored[i]["params"]))

        # 锦标赛选择 + 交叉 + 变异
        while len(new_pop) < pop_size:
            p1 = self._tournament_select(scored)
            p2 = self._tournament_select(scored)
            child = self._crossover(p1, p2)
            child = self._mutate(child, param_space, mutation_rate, mutation_scale)
            new_pop.append(child)

        return new_pop[:pop_size]

    def _tournament_select(self, scored: list, k: int = 3) -> dict:
        """锦标赛选择：随机选 k 个，取最优"""
        candidates = random.sample(scored, min(k, len(scored)))
        return max(candidates, key=lambda x: x["fitness"])["params"]

    def _crossover(self, p1: dict, p2: dict) -> dict:
        """均匀交叉：每个参数随机从父或母继承"""
        child = {}
        for key in p1:
            child[key] = p1[key] if random.random() < 0.5 else p2[key]
        return child

    def _mutate(self, ind: dict, param_space: dict, rate: float, scale: float) -> dict:
        """高斯变异"""
        for name, (low, high) in param_space.items():
            if random.random() < rate:
                if isinstance(low, int) and isinstance(high, int):
                    delta = max(1, int((high - low) * scale))
                    ind[name] = max(low, min(high, ind[name] + random.randint(-delta, delta)))
                else:
                    delta = (high - low) * scale
                    ind[name] = max(low, min(high, ind[name] + random.uniform(-delta, delta)))
        return ind


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/genetic_miner.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy

    df = fetch_index_daily("沪深300", "20200101", "20250601")

    go = GeneticOptimizer(df, cash=100000)
    best = go.evolve(
        MaCrossStrategy,
        param_space={"fast": (3, 15), "slow": (15, 50)},
        pop_size=20,
        generations=10,
    )

    print(f"\n最优参数: {best['params']}")
    print(f"夏普: {best['sharpe']:.2f}, 年化: {best['annual_return']:.2%}, 回撤: {best['drawdown']:.2%}")
    print(f"在第 {best['generation']} 代发现")
