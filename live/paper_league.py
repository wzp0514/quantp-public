"""
多策略纸上交易联赛（Paper League）

N策略并行纸上交易，共享实时行情源，独立资金，每日排行榜。

用法
--------
>>> from live.paper_league import PaperLeague, LeagueConfig
>>> config = LeagueConfig(
...     strategies={"双均线": (MaCrossStrategy, {"fast": 5, "slow": 20}),
...                  "布林带": (BollStrategy, {"period": 20})},
...     symbol="000300",
...     initial_cash_per_strategy=100000,
... )
>>> league = PaperLeague(config)
>>> league.start()
>>> lb = league.leaderboard()
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from config.log import get_logger

logger = get_logger("paper_league")


@dataclass
class LeagueConfig:
    strategies: dict  # name -> (strategy_class, params_dict)
    symbol: str = "000300"
    initial_cash_per_strategy: float = 100000.0
    poll_interval: float = 10.0
    max_concurrent: int = 10
    data_df: Optional[pd.DataFrame] = None  # 历史模式用


@dataclass
class LeaderboardEntry:
    rank: int = 0
    strategy: str = ""
    total_return: float = 0.0
    current_value: float = 0.0
    drawdown: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0
    guard_blown: bool = False
    guard_reason: str = ""
    signals: int = 0


class PaperLeague:
    """多策略纸上交易联赛。

    每个策略独立 PaperTrader 实例 + 独立资金 + 独立熔断。
    共享一个行情源，逐笔推送至所有实例。
    """

    def __init__(self, config: LeagueConfig):
        self.config = config
        self.traders: Dict[str, object] = {}
        self.results: Dict[str, dict] = {}
        self._running = False
        self._lock = threading.Lock()
        self._feed = None

    # ── 联赛控制 ───────────────────────────────────────────

    def start(self):
        """启动联赛。

        创建所有策略的 PaperTrader 实例并初始化。
        历史模式：逐日回放数据
        实时模式：启动行情轮询
        """
        from live.paper_trader import PaperTrader

        with self._lock:
            for name, (cls, params) in self.config.strategies.items():
                df = self.config.data_df if self.config.data_df is not None else pd.DataFrame()
                trader = PaperTrader(
                    cls, df,
                    initial_cash=self.config.initial_cash_per_strategy,
                    enable_guard=True,
                    **params,
                )
                self.traders[name] = trader
                logger.info(f"联赛选手 {name} 已就绪: {cls.__name__}")

        self._running = True
        logger.info(f"联赛已启动，{len(self.traders)} 位选手")

    def start_with_feed(self, feed):
        """以实时行情源启动联赛。"""
        self._feed = feed
        self.start()

        def on_snapshot(snap: dict):
            self.step_all(snap)

        feed.start(on_snapshot)
        logger.info("联赛实时模式运行中...")

    def step_all(self, snapshot: dict = None):
        """对所有策略执行一步（历史模式：逐日；实时模式：逐快照）。"""
        with self._lock:
            for name, trader in list(self.traders.items()):
                try:
                    if snapshot:
                        sym = self.config.symbol[-6:]
                        info = snapshot.get(sym, {})
                        if info:
                            trader.step_live(self.config.symbol, info)
                except Exception as e:
                    logger.error(f"选手 {name} 异常: {e}")

    def run_history(self):
        """历史数据模式：对所有策略执行完整历史回放。"""
        self.start()
        for name, trader in self.traders.items():
            try:
                result = trader.run()
                self.results[name] = result
                logger.info(f"选手 {name}: {result['total_return']:+.2%}")
            except Exception as e:
                logger.error(f"选手 {name} 失败: {e}")
                self.results[name] = {"error": str(e)}
        self._running = False

    def stop(self):
        """停止联赛。"""
        if self._feed:
            self._feed.stop()
        self._running = False
        logger.info("联赛已停止")

    # ── 排行榜 ─────────────────────────────────────────────

    def leaderboard(self) -> list[LeaderboardEntry]:
        """当前排行榜。"""
        entries = []
        with self._lock:
            for name, trader in self.traders.items():
                result = self.results.get(name, {})
                pos = trader.position_mgr.get_position("PAPER")

                # 当前资产
                if not trader.data.empty:
                    last_price = float(trader.data.iloc[-1]["close"])
                else:
                    last_price = 0
                total_value = trader.position_mgr.get_total_value({"PAPER": last_price})
                total_return = total_value / trader.initial_cash - 1

                # 回撤
                dd = 0.0
                if trader.guard:
                    dd = trader.guard.metrics.max_drawdown

                # 胜率
                filled = [o for o in trader.order_mgr.orders
                          if o.get("status") == "filled"]
                wins = sum(1 for o in filled
                          if o.get("price", 0) > 0 and o.get("side") == "sell"
                          and result.get("daily_log", []))

                # 简化胜率：从daily_log推算
                win_rate = 0.0
                if trader.daily_log:
                    win_trades = len([l for l in trader.daily_log
                                      if l.get("action") == "sell"
                                      and l.get("close", 0) > l.get("position_cost", 0)])
                    all_trades = len([l for l in trader.daily_log
                                      if l.get("action") in ("buy", "sell")]) / 2
                    win_rate = win_trades / max(all_trades, 1)

                entries.append(LeaderboardEntry(
                    rank=0,
                    strategy=name,
                    total_return=total_return,
                    current_value=total_value,
                    drawdown=dd,
                    trades=len(filled),
                    win_rate=min(win_rate, 1.0),
                    signals=len(trader.signals),
                    guard_blown=trader.guard.is_blown() if trader.guard else False,
                    guard_reason=trader.guard.reason if trader.guard else "",
                ))

        entries.sort(key=lambda e: e.total_return, reverse=True)
        for i, e in enumerate(entries, 1):
            e.rank = i
        return entries

    def print_leaderboard(self):
        """打印排行榜到控制台。"""
        entries = self.leaderboard()
        print("\n" + "=" * 70)
        print("                   纸上交易联赛排行榜")
        print("=" * 70)
        print(f"{'排名':<4} {'策略':<16} {'收益':>8} {'资产':>10} {'回撤':>8} {'交易':>5} {'胜率':>7} {'熔断'}")
        print("-" * 70)

        for e in entries:
            guard = "⚠️" if e.guard_blown else "✅"
            print(f"#{e.rank:<3} {e.strategy:<16} {e.total_return:>+7.2%} "
                  f"{e.current_value:>10,.0f} {e.drawdown:>7.2%} "
                  f"{e.trades:>5} {e.win_rate:>6.1%}  {guard}")

        print("=" * 70)

    # ── 状态 ────────────────────────────────────────────────

    def get_strategy_status(self, name: str) -> dict:
        """获取指定策略的详细状态。"""
        trader = self.traders.get(name)
        if not trader:
            return {"error": f"策略 {name} 不存在"}

        pos = trader.position_mgr.get_position("PAPER")
        return {
            "strategy": name,
            "cash": trader.position_mgr.cash,
            "position": pos,
            "signals": len(trader.signals),
            "orders": len(trader.order_mgr.orders),
            "guard": trader.guard.status() if trader.guard else {},
            "daily_log_count": len(trader.daily_log),
        }

    def get_all_status(self) -> dict:
        """获取所有策略的状态摘要。"""
        return {
            "running": self._running,
            "strategy_count": len(self.traders),
            "entries": [dict(
                strategy=name,
                cash=trader.position_mgr.cash,
                signals=len(trader.signals),
                guard_blown=trader.guard.is_blown() if trader.guard else False,
            ) for name, trader in self.traders.items()],
        }


def run_shootout_paper(
    strategies: dict,
    df: pd.DataFrame,
    cash_per: float = 100000.0,
) -> dict:
    """历史数据联赛：对所有策略运行纸上交易并返回排行榜。

    参数
    ----------
    strategies : {name: (strategy_class, params_dict)}
    df : 历史行情 DataFrame
    cash_per : 每策略初始资金

    返回
    -------
    {"leaderboard": [...], "league": PaperLeague}
    """
    config = LeagueConfig(
        strategies=strategies,
        symbol="沪深300",
        initial_cash_per_strategy=cash_per,
        data_df=df,
    )
    league = PaperLeague(config)
    league.run_history()
    return {
        "leaderboard": league.leaderboard(),
        "league": league,
    }


# ═══════════════════════════════════════════
# S5: 策略健康追踪 — 周频快照 + 退化检测 + 自动审查
# ═══════════════════════════════════════════

@dataclass
class WeeklySnapshot:
    """单周绩效快照"""
    strategy: str
    week: str                # ISO 周 "2026-W22"
    total_return: float = 0.0
    sharpe: float = 0.0
    drawdown: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    signals: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


class HealthTracker:
    """策略健康追踪器 — 周频快照持久化 + 退化检测 + 审查标记。

    用法
    --------
    >>> ht = HealthTracker()
    >>> ht.snapshot("MA_5_20", 0.05, 0.3, 0.12, 8, 0.40, 12)
    >>> issues = ht.check("MA_5_20")
    """

    def __init__(self, storage_path: str = "data/vault/vault_data/health_snapshots.jsonl"):
        self.path = storage_path
        self._snapshots: dict[str, list[WeeklySnapshot]] = {}
        self._load()

    def snapshot(self, strategy: str, total_return: float, sharpe: float,
                 drawdown: float, trades: int, win_rate: float, signals: int):
        """记录周频快照。每周调用一次。"""
        from datetime import date
        week = date.today().strftime("%Y-W%W")
        snap = WeeklySnapshot(
            strategy=strategy, week=week,
            total_return=total_return, sharpe=sharpe,
            drawdown=drawdown, trades=trades,
            win_rate=win_rate, signals=signals,
            timestamp=datetime.now().isoformat(),
        )
        self._snapshots.setdefault(strategy, []).append(snap)
        self._save_one(snap)
        logger.debug(f"快照已保存: {strategy} @ {week}")

    def check(self, strategy: str, lookback_weeks: int = 4) -> list[dict]:
        """检测策略是否出现绩效退化。

        检查项：
        1. 连续亏损 — 最近N周收益持续下降
        2. 回撤恶化 — 回撤连续加深
        3. 信号枯竭 — 信号数量持续下降
        4. 夏普退化 — 夏普显著低于历史基准
        """
        snaps = self._snapshots.get(strategy, [])
        if len(snaps) < 3:
            return []

        issues = []
        recent = snaps[-lookback_weeks:] if len(snaps) >= lookback_weeks else snaps
        baseline = snaps[:-lookback_weeks] if len(snaps) > lookback_weeks else []

        # 1. 连续亏损检测
        returns = [s.total_return for s in recent]
        if len(returns) >= 3 and all(r2 < r1 for r1, r2 in zip(returns, returns[1:])):
            issues.append({"type": "consecutive_loss", "severity": "warning",
                           "detail": f"连续{len(returns)}周收益下降: {returns[-1]:.2%} → {returns[0]:.2%}"})

        # 2. 回撤恶化
        dds = [s.drawdown for s in recent]
        if len(dds) >= 3 and all(d2 > d1 for d1, d2 in zip(dds, dds[1:])):
            issues.append({"type": "drawdown_worsening", "severity": "danger",
                           "detail": f"回撤连续{len(dds)}周加深: {dds[0]:.2%} → {dds[-1]:.2%}"})

        # 3. 信号枯竭
        signals = [s.signals for s in recent]
        if len(signals) >= 4 and all(s <= signals[0] * 0.3 for s in signals[-2:]):
            issues.append({"type": "signal_starvation", "severity": "warning",
                           "detail": f"信号大幅减少: 最近2周均值={sum(signals[-2:])/2:.0f}, 初始={signals[0]}"})

        # 4. 夏普退化
        if baseline:
            baseline_sharpe = sum(s.sharpe for s in baseline) / len(baseline)
            recent_sharpe = sum(s.sharpe for s in recent) / len(recent)
            if baseline_sharpe > 0.2 and recent_sharpe < baseline_sharpe * 0.3:
                issues.append({"type": "sharpe_decay", "severity": "danger",
                               "detail": f"夏普退化: {baseline_sharpe:.2f} → {recent_sharpe:.2f}"})

        if issues:
            logger.warning(f"策略 {strategy} 检出 {len(issues)} 项退化: "
                          f"{[i['type'] for i in issues]}")

        return issues

    def mark_for_review(self, strategy: str, issues: list[dict]) -> bool:
        """对检测到退化的策略标记"需审查"。返回 True 表示需要人工审查。"""
        if not issues:
            return False

        dangers = [i for i in issues if i["severity"] == "danger"]
        if dangers:
            logger.info(f"策略 {strategy} 已被自动标记为需审查 "
                       f"({len(dangers)} 项 danger: {[d['type'] for d in dangers]})")
            self._tag_strategy(strategy, "needs_review", dangers)
            return True

        warnings = [i for i in issues if i["severity"] == "warning"]
        if len(warnings) >= 2:
            logger.info(f"策略 {strategy} 累积 {len(warnings)} 项 warning，标记需关注")
            self._tag_strategy(strategy, "needs_attention", warnings)
            return False

        return False

    def health_report(self) -> str:
        """生成所有策略的健康报告。"""
        lines = ["=" * 70, "  策略健康追踪报告", "=" * 70]
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"  时间: {now} | 追踪策略: {len(self._snapshots)} 个")
        lines.append("─" * 70)
        lines.append(f"  {'策略':<20} {'快照':>5} {'退化':>5} {'状态'}")
        lines.append("  " + "-" * 50)

        for name, snaps in sorted(self._snapshots.items()):
            issues = self.check(name)
            n_issues = len(issues)
            dangers = sum(1 for i in issues if i["severity"] == "danger")
            if dangers > 0:
                status = "需审查"
            elif n_issues > 0:
                status = "需关注"
            else:
                status = "正常"
            lines.append(f"  {name:<20} {len(snaps):>5} {n_issues:>5}  {status}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _tag_strategy(self, strategy: str, tag: str, issues: list[dict]):
        """在策略仓库中标记需审查/需关注。"""
        try:
            from strategies_repo.repo import StrategyRepo
            repo = StrategyRepo()
            tags = [f"{tag}:{i['type']}" for i in issues]
            repo.tag_strategy(strategy, tags)
        except Exception as e:
            logger.debug(f"策略标记失败: {strategy} ({e})")

    def _save_one(self, snap: WeeklySnapshot):
        import json, os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap.to_dict(), ensure_ascii=False) + "\n")

    def _load(self):
        import json
        if not __import__("os").path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    snap = WeeklySnapshot(**{k: d.get(k) for k in WeeklySnapshot.__dataclass_fields__})
                    self._snapshots.setdefault(snap.strategy, []).append(snap)
                except Exception:
                    pass


def check_health(strategies: dict, df, cash: float = 100000.0) -> dict:
    """对策略列表执行一次性健康检查（回测 + 退化检测）。

    返回 {"report": str, "issues": {name: [...]}, "league": PaperLeague}
    """
    from backtest.engine.bt_runner import run_backtest

    config = LeagueConfig(
        strategies=strategies, symbol="沪深300",
        initial_cash_per_strategy=cash, data_df=df,
    )
    league = PaperLeague(config)
    league.run_history()

    tracker = HealthTracker()
    all_issues = {}

    for name, trader in league.traders.items():
        result = league.results.get(name, {})
        if "error" in result:
            continue
        sr = result.get("sharpe") or 0
        dd = result.get("drawdown") or 0
        tr = result.get("total_return") or 0

        # 估算周胜率（简化）
        filled = [o for o in trader.order_mgr.orders if o.get("status") == "filled"]
        n_sells = sum(1 for o in filled if o.get("side") == "sell")
        n_trades = max(len(filled) // 2, 1)
        n_signals = len(trader.signals)

        tracker.snapshot(name, tr, sr, dd, n_trades, 0.4, n_signals)
        issues = tracker.check(name)
        if issues:
            all_issues[name] = issues
            tracker.mark_for_review(name, issues)

    report = tracker.health_report()
    return {"report": report, "issues": all_issues, "league": league}


# ── 命令行测试 ────────────────────────────────────────────
# python live/paper_league.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy

    print("联赛测试: 3个双均线变体")
    print("=" * 60)

    df = fetch_index("沪深300", "20240101", "20250601")

    strategies = {
        "MA(5,20)": (MaCrossStrategy, {"fast": 5, "slow": 20}),
        "MA(10,30)": (MaCrossStrategy, {"fast": 10, "slow": 30}),
        "MA(5,60)": (MaCrossStrategy, {"fast": 5, "slow": 60}),
    }

    result = run_shootout_paper(strategies, df, cash_per=100000)
    league = result["league"]
    league.print_leaderboard()
