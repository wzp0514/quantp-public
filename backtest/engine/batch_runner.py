"""
BatchRunner — 批量回测调度器

替代 shootout/market scan/full backtest/strategy miner 中
重复的"遍历策略→回测→收集结果"循环。

支持串行和并行（线程池）两种模式。

用法
--------
>>> runner = BatchRunner(data, cash=100000)
>>> results = runner.run(strategies, max_workers=4)
>>> print(runner.summary())
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from config.log import get_logger

logger = get_logger("batch_runner")


@dataclass
class BatchConfig:
    """单次批量任务的配置"""
    cash: float = 100000.0
    max_workers: int = 1          # 1=串行, >1=并行
    timeout_seconds: int = 600    # 单个策略超时
    min_trades: int = 3           # 最少交易笔数（太少的直接跳过）
    save_results: bool = False


@dataclass
class BatchResult:
    """批量回测汇总结果"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    elapsed: float = 0.0
    results: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Batch: {self.total} tested, {self.passed} passed, "
            f"{self.failed} failed, {self.elapsed:.1f}s"
        )

    def best(self):
        """返回夏普最高的结果"""
        valid = [r for r in self.results if r.get("sharpe") is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r.get("sharpe") or -999)

    def top_n(self, n: int = 10):
        """返回夏普最高的 N 个结果"""
        valid = [r for r in self.results if r.get("sharpe") is not None]
        valid.sort(key=lambda r: r.get("sharpe") or -999, reverse=True)
        return valid[:n]


class BatchRunner:
    """
    批量回测调度器。

    接口：
      runner.run(strategies) → BatchResult
      runner.summary()       → 文本报告

    策略输入格式: {name: (class, params_dict), ...}
    """

    def __init__(self, df: pd.DataFrame, config: BatchConfig = None):
        self.df = df
        self.config = config or BatchConfig()

    def run(self, strategies: dict) -> BatchResult:
        """
        批量执行回测。

        参数
        ----------
        strategies : dict
            {name: (strategy_class, params_dict), ...}

        返回
        -------
        BatchResult
        """
        from backtest.engine.bt_runner import run_backtest

        t0 = time.time()
        items = list(strategies.items())
        passed = []
        failed = []

        if self.config.max_workers <= 1:
            # 串行
            for name, (cls, params) in items:
                try:
                    r = run_backtest(cls, self.df.copy(), initial_cash=self.config.cash, **params)
                    if r.total_trades >= self.config.min_trades:
                        passed.append({"name": name, "result": r})
                    else:
                        failed.append({"name": name, "reason": f"trades={r.total_trades} < {self.config.min_trades}"})
                except Exception as e:
                    failed.append({"name": name, "reason": str(e)[:100]})
                    logger.debug(f"Batch: {name} failed: {e}")
        else:
            # 并行
            def _run_one(name, cls, params):
                try:
                    r = run_backtest(cls, self.df.copy(), initial_cash=self.config.cash, **params)
                    return name, r, None
                except Exception as e:
                    return name, None, str(e)[:100]

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(_run_one, n, c, p): n for n, (c, p) in items}
                for future in as_completed(futures):
                    name, result, err = future.result()
                    if err:
                        failed.append({"name": name, "reason": err})
                    elif result.total_trades >= self.config.min_trades:
                        passed.append({"name": name, "result": result})
                    else:
                        failed.append({"name": name, "reason": "too few trades"})

        elapsed = time.time() - t0
        logger.info(
            f"Batch complete: {len(items)} strategies, {len(passed)} passed, "
            f"{len(failed)} failed, {elapsed:.1f}s"
            f"{' (parallel x%d)' % self.config.max_workers if self.config.max_workers > 1 else ''}"
        )

        return BatchResult(
            total=len(items),
            passed=len(passed),
            failed=len(failed),
            elapsed=elapsed,
            results=passed + failed,
        )

    def summary(self, result: BatchResult = None, top_n: int = 10) -> str:
        """生成批量回测文本报告"""
        if result is None:
            return "No results yet."

        lines = [
            "=" * 70,
            f"  Batch Backtest Report ({result.total} strategies, {result.elapsed:.1f}s)",
            "=" * 70,
            f"  Passed: {result.passed} | Failed: {result.failed}",
            "─" * 70,
            f"  {'Rank':<4} {'Strategy':<22} {'AR':>6} {'DD':>5} {'SR':>6} {'Trades':>5}",
            "  " + "-" * 54,
        ]

        ranked = result.top_n(top_n)
        for i, entry in enumerate(ranked):
            r = entry["result"]
            s = r.get("sharpe")
            sr = f"{s:.2f}" if s is not None else "N/A"
            lines.append(
                f"  {i+1:<4} {entry['name']:<22} "
                f"{r.get('annual_return',0)*100:>5.1f}% "
                f"{r.get('drawdown',0)*100:>4.1f}% "
                f"{sr:>6} {r.get('total_trades',0):>5}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================
# 命令行测试
# ============================================================
# python backtest/engine/batch_runner.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fd
    from backtest.strategy_market import ALL_STRATEGIES
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    from backtest.strategies.builtin.bollinger import BollingerStrategy
    from backtest.strategies.experimental.resonance import ResonanceStrategy

    df = fd("沪深300", "20230101", "20260524")

    strategies = {
        "MA_Cross_5_20": (MaCrossStrategy, {"fast": 5, "slow": 20}),
        "MA_Cross_10_30": (MaCrossStrategy, {"fast": 10, "slow": 30}),
        "Bollinger": (BollingerStrategy, {"period": 20, "devfactor": 2.0}),
        "Resonance_2sig": (ResonanceStrategy, {"require_signals": 2, "regime_enabled": True}),
    }

    runner = BatchRunner(df, BatchConfig(cash=100000, max_workers=1))
    result = runner.run(strategies)
    print(runner.summary(result))
