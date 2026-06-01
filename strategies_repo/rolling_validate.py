"""
滚动窗口验证 — 不停验证策略库中的每个策略

一次性样本外不够：2020-2023训练 → 2024-2025验证，碰巧好/坏概率大。
滚动窗口才是正确做法：

  窗口1: [2018-2020训练] → [2021验证]
  窗口2: [2019-2021训练] → [2022验证]
  窗口3: [2020-2022训练] → [2023验证]
  ...一直滚到数据用完

一个策略只有通过多个窗口的验证，才算真正可靠。

评估标准：
  - 年化收益 > 基准（超额为正）
  - 验证期收益衰减 ≤ 30%
  - 通过率 > 50%（6个窗口中至少4个通过）

用法
--------
>>> from strategies_repo.rolling_validate import RollingValidator
>>> rv = RollingValidator("布林带回归")
>>> result = rv.run()
>>> print(result["verdict"])  # "PASS" or "FAIL"
"""

import logging
import time
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger
logger = get_logger("rolling_validate")
class RollingValidator:
    """
    滚动窗口验证器。

    对单个策略，在多个滑动窗口上反复验证，
    返回是否稳定可靠。
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str = "沪深300",
        train_years: int = 3,
        test_years: int = 1,
        step_years: int = 1,
        cash: float = 100000.0,
    ):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years
        self.cash = cash

    def run(self) -> dict:
        """
        执行滚动验证。

        返回
        -------
        dict: {
            verdict: "PASS" | "FAIL",
            pass_rate: float,
            windows: [{train_start, train_end, test_start, test_end,
                      train_return, test_return, decay, passed}, ...],
            summary: str,
        }
        """
        from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
        from backtest.engine.bt_runner import run_backtest
        from backtest.strategy_market import ALL_STRATEGIES

        # 加载策略
        if self.strategy_name not in ALL_STRATEGIES:
            return {"verdict": "FAIL", "error": f"策略不存在: {self.strategy_name}"}

        info = ALL_STRATEGIES[self.strategy_name]
        strategy_class = info["class"]
        params = info.get("params", {})

        # 拉取全量数据
        logger.info(f"滚动验证: {self.strategy_name} on {self.symbol}")
        df = fetch_index_daily(self.symbol, "20100101", "")
        if df.empty:
            return {"verdict": "FAIL", "error": "无数据"}

        df["date"] = pd.to_datetime(df["date"])
        data_start = df["date"].min()
        data_end = df["date"].max()

        # 生成时间窗口
        windows = []
        train_start = data_start

        while True:
            train_end = train_start + pd.DateOffset(years=self.train_years)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(years=self.test_years)

            if test_end > data_end:
                break

            train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
            test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

            if len(train_df) < 50 or len(test_df) < 20:
                train_start += pd.DateOffset(years=self.step_years)
                continue

            windows.append({
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "train_bars": len(train_df),
                "test_bars": len(test_df),
            })

            train_start += pd.DateOffset(years=self.step_years)

        if len(windows) < 3:
            return {"verdict": "FAIL", "error": f"窗口不足 ({len(windows)} < 3)", "windows": []}

        logger.info(f"共 {len(windows)} 个窗口")
        start_time = time.time()

        # 逐窗口回测
        results = []
        passed_count = 0

        for i, w in enumerate(windows):
            train_df = df[(df["date"] >= pd.Timestamp(w["train_start"]))
                         & (df["date"] <= pd.Timestamp(w["train_end"]))]
            test_df = df[(df["date"] >= pd.Timestamp(w["test_start"]))
                        & (df["date"] <= pd.Timestamp(w["test_end"]))]

            # 训练期
            train_result = run_backtest(
                strategy_class, train_df,
                initial_cash=self.cash, a_share_mode=True, **params,
            )

            # 验证期
            test_result = run_backtest(
                strategy_class, test_df,
                initial_cash=self.cash, a_share_mode=True, **params,
            )

            train_ret = train_result["annual_return"]
            test_ret = test_result["annual_return"]

            # 基准 = 买入持有
            bh_test = test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1
            test_days = len(test_df)
            bh_test_annual = (1 + bh_test) ** (252 / test_days) - 1 if test_days > 0 else 0

            # 衰减
            decay = (train_ret - test_ret) / train_ret if train_ret > 0 else (1.0 if test_ret < train_ret else 0)

            # 通过条件
            excess_positive = test_ret > bh_test_annual  # 跑赢基准
            decay_ok = decay <= 0.30 or train_ret <= 0  # 衰减≤30%（训练期正的）
            passed = excess_positive and decay_ok

            if passed:
                passed_count += 1

            w_result = {
                **w,
                "train_return": train_ret,
                "test_return": test_ret,
                "benchmark_return": bh_test_annual,
                "excess": test_ret - bh_test_annual,
                "decay": decay,
                "excess_positive": excess_positive,
                "decay_ok": decay_ok,
                "passed": passed,
            }
            results.append(w_result)

            logger.info(
                f"  窗口{i+1}: train={train_ret:.2%} test={test_ret:.2%} "
                f"bm={bh_test_annual:.2%} decay={decay:.1%} "
                f"{'PASS' if passed else 'FAIL'}"
            )

        elapsed = time.time() - start_time
        pass_rate = passed_count / len(results)

        # 判定
        if pass_rate >= 0.6:
            verdict = "PASS"
            verdict_text = "稳定可靠 — 可进入纸上交易"
        elif pass_rate >= 0.4:
            verdict = "WEAK_PASS"
            verdict_text = "勉强通过 — 需要更多数据验证"
        else:
            verdict = "FAIL"
            verdict_text = "不稳定 — 不建议实盘"

        summary_lines = [
            "=" * 60,
            f"  滚动验证: {self.strategy_name}",
            "=" * 60,
            f"  窗口数: {len(results)} | 通过: {passed_count} | 通过率: {pass_rate:.0%}",
            f"  判定: {verdict} — {verdict_text}",
            f"  平均超额(vs基准): {np.mean([r['excess'] for r in results]):.2%}",
            f"  平均衰减: {np.mean([r['decay'] for r in results]):.1%}",
            f"  耗时: {elapsed:.0f}s",
            "=" * 60,
        ]

        output = {
            "strategy": self.strategy_name,
            "verdict": verdict,
            "verdict_text": verdict_text,
            "pass_rate": pass_rate,
            "window_count": len(results),
            "passed_count": passed_count,
            "windows": results,
            "summary": "\n".join(summary_lines),
        }

        print(output["summary"])
        return output


def validate_all_in_repo(top_n: int = 0) -> list[dict]:
    """
    验证策略仓库中的所有（或 Top N）策略。

    返回按通过率降序排列的结果列表。
    """
    from strategies_repo.repo import StrategyRepo
    repo = StrategyRepo()
    all_s = repo.list()

    strategies = [s["name"] for s in all_s]
    if top_n > 0:
        strategies = strategies[:top_n]

    logger.info(f"滚动验证全部 {len(strategies)} 个策略...")
    results = []
    for name in strategies:
        try:
            rv = RollingValidator(name)
            r = rv.run()
            results.append(r)
        except Exception as e:
            logger.error(f"{name} 验证失败: {e}")
            results.append({"strategy": name, "verdict": "ERROR", "error": str(e)})

    results.sort(key=lambda x: x.get("pass_rate", 0), reverse=True)
    return results


def get_battle_ready() -> list[str]:
    """
    获取所有通过滚动验证、可以上实盘的策略名列表。

    这就是你的"弹药库"。
    """
    results = validate_all_in_repo()
    ready = [r["strategy"] for r in results if r.get("verdict") in ("PASS", "WEAK_PASS")]
    logger.info(f"可实盘策略: {len(ready)}/{len(results)}")
    return ready


# ============================================================
# 命令行测试
# ============================================================
# python strategies_repo/rolling_validate.py

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    rv = RollingValidator("布林带回归", train_years=3, test_years=1, step_years=1)
    rv.run()
