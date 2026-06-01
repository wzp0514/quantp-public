"""
TaskRunner — 统一任务调度抽象

替代 run_strategy.py 的 9 个 cmd_* 和 interactive.py 的 8 个 menu_*。
调用方只构造 TaskSpec，TaskRunner 负责"取数据→调 runner→格式化输出"。

设计：一个 TaskSpec → 一个 TaskRunner.run() → 一个 TaskResult。
新增分析模式只需注册新的 TaskSpec 类型，不需要改 run_strategy + interactive 两处。

用法
--------
>>> runner = TaskRunner()
>>> spec = TaskSpec(kind="backtest", strategy="ma_cross", symbol="沪深300")
>>> result = runner.run(spec)
>>> print(result.summary)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from config.log import get_logger

logger = get_logger("task_runner")


# ── TaskSpec: 描述"做什么" ──

@dataclass
class TaskSpec:
    """统一任务描述"""
    kind: str                    # backtest / shootout / mine / paper / replay / ga / bayes / rl / live
    strategy: str = ""           # 策略名（在 ALL_STRATEGIES 中）
    symbol: str = "沪深300"      # 标的
    start: str = "20230101"      # 数据起始
    end: str = ""                # 数据截止（空=今天）
    cash: float = 100000.0
    plot: bool = False
    save_path: str = ""

    # 挖掘/优化专属
    max_combos: int = 50
    min_trades: int = 10
    pop_size: int = 20
    generations: int = 10
    n_trials: int = 30
    timesteps: int = 30000

    # 回放专属
    replay_days: int = 0         # 0=全部
    replay_indicators: list[str] = field(default_factory=list)

    # 额外参数（透传到策略）
    extra_params: dict = field(default_factory=dict)


# ── TaskResult: 封装"产出什么" ──

@dataclass
class TaskResult:
    """统一任务结果"""
    success: bool = False
    kind: str = ""
    data: any = None             # 核心产出（BacktestResult / BatchResult / 挖掘结果 / ...）
    elapsed: float = 0.0
    summary: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)


# ── TaskRunner ──

class TaskRunner:
    """
    统一任务调度器。

    新增分析模式只需注册新的 handler:
      runner.register("new_mode", handler_function)
    """

    def __init__(self):
        self._handlers = {}
        self._register_defaults()

    def _register_defaults(self):
        self._handlers["backtest"] = self._run_backtest
        self._handlers["shootout"] = self._run_shootout
        self._handlers["mine"] = self._run_mine
        self._handlers["paper"] = self._run_paper
        self._handlers["replay"] = self._run_replay
        self._handlers["ga"] = self._run_ga
        self._handlers["bayes"] = self._run_bayes
        self._handlers["batch"] = self._run_batch
        self._handlers["paper_live"] = self._run_paper_live
        self._handlers["paper_daemon"] = self._run_paper_daemon
        self._handlers["paper_diff"] = self._run_paper_diff
        self._handlers["paper_league"] = self._run_paper_league

    def register(self, kind: str, handler):
        self._handlers[kind] = handler

    def run(self, spec: TaskSpec) -> TaskResult:
        """执行一个任务，返回 TaskResult"""
        t0 = time.time()
        handler = self._handlers.get(spec.kind)
        if handler is None:
            return TaskResult(success=False, kind=spec.kind, error=f"Unknown task kind: {spec.kind}")

        try:
            result = handler(spec)
            result.elapsed = time.time() - t0
            result.kind = spec.kind
            return result
        except Exception as e:
            logger.error(f"Task '{spec.kind}' failed: {e}")
            return TaskResult(
                success=False, kind=spec.kind,
                error=str(e), elapsed=time.time() - t0,
            )

    # ── 内置 handler ──

    def _get_data(self, spec: TaskSpec):
        from data.fetchers.fallback import fetch_index_daily_safe as fd
        return fd(spec.symbol, spec.start, spec.end or "")

    def _get_strategy_class(self, name: str):
        from backtest.strategy_market import ALL_STRATEGIES
        info = ALL_STRATEGIES.get(name)
        if info:
            return info.get("class")
        return None

    def _run_backtest(self, spec: TaskSpec) -> TaskResult:
        from backtest.engine.bt_runner import run_backtest
        from backtest.analysis.report import generate_report

        df = self._get_data(spec)
        cls = self._get_strategy_class(spec.strategy) if spec.strategy else None
        if cls is None:
            # fallback: import by module path if strategy specified
            cls = self._resolve_strategy(spec.strategy)

        kwargs = spec.extra_params.copy()
        result = run_backtest(cls, df, initial_cash=spec.cash, **kwargs)
        report = generate_report(result.to_dict())

        if spec.plot:
            from backtest.analysis.report import plot_kline
            path = spec.save_path or f"reports/kline_{spec.strategy or 'backtest'}.png"
            plot_kline(df, indicators=spec.extra_params.get("indicators", ["ma20", "ma60"]),
                      days=120, title=f"{spec.strategy} on {spec.symbol}", save_path=path)

        return TaskResult(success=True, data=result, summary=report)

    def _run_shootout(self, spec: TaskSpec) -> TaskResult:
        from backtest.engine.batch_runner import BatchRunner, BatchConfig
        from backtest.strategy_market import ALL_STRATEGIES
        df = self._get_data(spec)
        strategies = {}
        for name, info in ALL_STRATEGIES.items():
            cls = info.get("class")
            if cls:
                strategies[name] = (cls, info.get("params", {}))
        runner = BatchRunner(df, BatchConfig(cash=spec.cash, max_workers=1))
        batch_result = runner.run(strategies)
        return TaskResult(success=True, data=batch_result, summary=runner.summary(batch_result))

    def _run_mine(self, spec: TaskSpec) -> TaskResult:
        from backtest.strategy_miner import StrategyMiner
        df = self._get_data(spec)
        miner = StrategyMiner(df, cash=spec.cash)
        result = miner.mine(max_combinations=spec.max_combos, min_trades=spec.min_trades)
        return TaskResult(success=True, data=result, summary=result.get("summary", ""))

    def _run_paper(self, spec: TaskSpec) -> TaskResult:
        from live.paper_trader import PaperTrader
        df = self._get_data(spec)
        cls = self._get_strategy_class(spec.strategy)
        pt = PaperTrader(df, cls, cash=spec.cash)
        result = pt.run()
        return TaskResult(success=True, data=result, summary=result.get("summary", ""))

    def _run_replay(self, spec: TaskSpec) -> TaskResult:
        from live.replay.engine import ReplayEngine
        df = self._get_data(spec)
        engine = ReplayEngine(df, cash=spec.cash)
        engine.run(max_steps=spec.replay_days)
        return TaskResult(success=True, data=engine._snapshots, summary=engine.report())

    def _run_ga(self, spec: TaskSpec) -> TaskResult:
        from backtest.analysis.genetic_miner import GeneticOptimizer
        df = self._get_data(spec)
        cls = self._get_strategy_class(spec.strategy)
        go = GeneticOptimizer(df, cash=spec.cash)
        best = go.evolve(cls, pop_size=spec.pop_size, generations=spec.generations,
                         param_space=spec.extra_params.get("param_space", {}) if spec.extra_params else {})
        return TaskResult(success=True, data=best, summary=f"GA best: {best}")

    def _run_bayes(self, spec: TaskSpec) -> TaskResult:
        from backtest.analysis.bayesian_opt import BayesianOptimizer
        df = self._get_data(spec)
        cls = self._get_strategy_class(spec.strategy)
        bo = BayesianOptimizer(df, cash=spec.cash)
        best = bo.optimize(cls, n_trials=spec.n_trials,
                          param_space=spec.extra_params.get("param_space", {}) if spec.extra_params else {})
        return TaskResult(success=True, data=best, summary=f"Bayes best: {best}")

    def _run_batch(self, spec: TaskSpec) -> TaskResult:
        """批量回测：自定义策略列表（非全局扫描）"""
        from backtest.engine.batch_runner import BatchRunner, BatchConfig
        df = self._get_data(spec)
        strategies = {}
        for name, (cls, params) in spec.extra_params.get("strategies", {}).items():
            strategies[name] = (cls, params)
        runner = BatchRunner(df, BatchConfig(cash=spec.cash, max_workers=spec.extra_params.get("max_workers", 1)))
        batch_result = runner.run(strategies)
        return TaskResult(success=True, data=batch_result, summary=runner.summary(batch_result))

    def _run_paper_live(self, spec: TaskSpec) -> TaskResult:
        from live.feed.realtime_feed import make_index_feed
        from live.paper_trader import run_paper_trading_live
        cls = self._get_strategy_class(spec.strategy)
        feed = make_index_feed(spec.symbol or "沪深300", poll_interval=10.0)
        result = run_paper_trading_live(cls, feed, spec.symbol, spec.cash)
        return TaskResult(success=True, data=result, summary=result.get("summary", ""))

    def _run_paper_daemon(self, spec: TaskSpec) -> TaskResult:
        from live.paper_daemon import PaperDaemon
        daemon = PaperDaemon()
        iid = daemon.start_instance(spec.strategy, spec.symbol, spec.cash,
                                     spec.extra_params or {})
        return TaskResult(success=True, data={"instance_id": iid},
                          summary=f"PaperDaemon instance #{iid} started")

    def _run_paper_diff(self, spec: TaskSpec) -> TaskResult:
        from live.analysis.paper_diff import compare_paper_backtest, generate_diff_text
        from live.execution.paper_store import PaperTradeStore
        df = self._get_data(spec)
        cls = self._get_strategy_class(spec.strategy)
        from backtest.engine.bt_runner import run_backtest
        bt_result = run_backtest(cls, df, initial_cash=spec.cash,
                                 **spec.extra_params.get("strategy_params", {}))
        store = PaperTradeStore()
        iid = spec.extra_params.get("instance_id", 1) if spec.extra_params else 1
        paper_data = store.load_instance(iid)
        store.close()
        paper_result = {
            "daily_log": paper_data.get("daily_log", []),
            "signals": [{"date": l.get("date"), "signal": l.get("signal")}
                         for l in paper_data.get("daily_log", []) if l.get("signal")],
            "orders": paper_data.get("orders", []),
        }
        report = compare_paper_backtest(paper_result, bt_result, spec.strategy, spec.symbol)
        return TaskResult(success=True, data=report, summary=generate_diff_text(report))

    def _run_paper_league(self, spec: TaskSpec) -> TaskResult:
        from live.paper_league import PaperLeague, LeagueConfig
        df = self._get_data(spec)
        strategies = {}
        for name, (cls_name, params) in (spec.extra_params or {}).get("strategies", {}).items():
            cls = self._get_strategy_class(cls_name)
            strategies[name] = (cls, params)
        config = LeagueConfig(strategies=strategies, symbol=spec.symbol,
                              initial_cash_per_strategy=spec.cash, data_df=df)
        league = PaperLeague(config)
        league.run_history()
        lb = league.leaderboard()
        summary = "\n".join(f"  #{e.rank} {e.strategy}: {e.total_return:+.2%}" for e in lb)
        return TaskResult(success=True, data={"leaderboard": lb}, summary=summary)

    def _resolve_strategy(self, name: str):
        """从模块路径动态加载策略类"""
        if not name:
            return None
        try:
            import importlib
            mod = importlib.import_module(f"backtest.strategies.{name}")
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and hasattr(obj, 'params'):
                    return obj
        except Exception:
            logger.exception(f"Failed to load strategy module: {name}")
        return None


# ============================================================
# 命令行测试
# ============================================================
# python core/runner.py

if __name__ == "__main__":
    print("=" * 60)
    print("TaskRunner Demo")
    print("=" * 60)

    runner = TaskRunner()

    # Quick backtest
    spec = TaskSpec(kind="backtest", strategy="双均线交叉", symbol="沪深300",
                    start="20250101", extra_params={"fast": 5, "slow": 20})
    result = runner.run(spec)
    print(f"Backtest: success={result.success}, elapsed={result.elapsed:.1f}s")
    if result.success and hasattr(result.data, 'summary'):
        print(result.data.summary())

    # Batch
    print()
    spec2 = TaskSpec(kind="batch", extra_params={
        "strategies": {
            "MA_5_20": (__import__("backtest.strategies.ma_cross", fromlist=["MaCrossStrategy"]).MaCrossStrategy, {"fast": 5, "slow": 20}),
            "MA_10_30": (__import__("backtest.strategies.ma_cross", fromlist=["MaCrossStrategy"]).MaCrossStrategy, {"fast": 10, "slow": 30}),
        }
    })
    result2 = runner.run(spec2)
    print(f"Batch: success={result2.success}, {result2.summary[:80]}")
