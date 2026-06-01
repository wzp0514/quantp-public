"""
ReplayEngine — 历史回放引擎（量化飞行模拟器）

把历史数据按时间顺序逐条"播放"，程序假装不知道后面的行情，实时做决策。
外层可以渲染成任何形式：文本日志、K线图标注、Streamlit 互动。

接口
--------
>>> engine = ReplayEngine(data, strategy_class, cash=100000)
>>> for snapshot in engine.run():
...     print(snapshot)          # 每推进一步拿到当前状态
...     if engine.is_finished():
...         break
>>> print(engine.report())

改造 paper_trader: PaperTrader.run() 变成 ReplayEngine 的 consumer，
不再需要预先跑一遍完整回测来提取信号。
"""

import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("replay_engine")


@dataclass
class ReplaySnapshot:
    """每个时间步的状态快照"""
    date: str
    bar_index: int
    total_bars: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    position: int = 0            # 持仓数量（>0 持多）
    position_pct: float = 0.0    # 仓位比例
    equity: float = 0.0          # 当前权益
    cash: float = 0.0            # 剩余现金
    signal: str = ""             # 当前信号: buy/sell/hold
    signal_reason: str = ""      # 信号原因
    pnl_pct: float = 0.0         # 浮动盈亏
    is_bear_market: bool = False # 是否熊市


class ReplayEngine:
    """
    历史回放引擎。

    逐 bar 推进，每个 bar 调一次策略逻辑，返回 snapshot。
    策略可以是：
      - Backtrader Strategy 子类（callback 模式）
      - 纯函数 compute(data_slice, context) → action

    Parameters
    ----------
    df : DataFrame
        含 date/open/high/low/close/volume
    cash : float
        初始资金
    stop_loss_pct : float
        硬止损比例（0.05 = 5%）
    on_step : callable
        每步回调 snapshot, engine → 可用于画图/日志
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cash: float = 100000.0,
        stop_loss_pct: float = 0.05,
        on_step: Callable = None,
        bear_threshold: float = -0.05,   # 60日跌幅 < -5% = 熊市
    ):
        self.df = df.sort_values("date").reset_index(drop=True).copy()
        self.cash = cash
        self.initial_cash = cash
        self.stop_loss_pct = stop_loss_pct
        self.on_step = on_step
        self.bear_threshold = bear_threshold

        # 内部状态
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._bar_index = 0
        self._trades: list[dict] = []
        self._equity_curve: list[dict] = []
        self._snapshots: list[ReplaySnapshot] = []
        self._stopped = False
        self._stop_reason = ""

    @property
    def position(self) -> int:
        return self._position

    @property
    def equity(self) -> float:
        if self._position > 0 and self._bar_index < len(self.df):
            return self.cash + self._position * self.df["close"].iloc[self._bar_index]
        return self.cash

    def is_finished(self) -> bool:
        return self._bar_index >= len(self.df) - 1 or self._stopped

    def is_bear(self) -> bool:
        """简易区制判断：60日收益 < 阈值 = 熊市"""
        i = self._bar_index
        if i < 60:
            return False
        ret_60d = self.df["close"].iloc[i] / self.df["close"].iloc[i - 60] - 1
        return ret_60d < self.bear_threshold

    def step(self) -> Optional[ReplaySnapshot]:
        """推进一个 bar，返回当前快照。结束返回 None。"""
        if self.is_finished():
            return None

        i = self._bar_index
        row = self.df.iloc[i]
        close = float(row["close"])
        open_ = float(row.get("open", close))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        volume = float(row.get("volume", 0))
        date = str(row.get("date", ""))[:10]

        # 止损检查
        if self._position > 0 and self._entry_price > 0:
            loss = close / self._entry_price - 1
            if loss < -self.stop_loss_pct:
                self._close_position(close, date, "stop_loss")
                self._stopped = True
                self._stop_reason = f"止损触发 @ {date}"

        snapshot = ReplaySnapshot(
            date=date,
            bar_index=i,
            total_bars=len(self.df),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            position=self._position,
            position_pct=self._position * close / (self.cash + self._position * close + 1e-10),
            equity=self.equity,
            cash=self.cash,
            is_bear_market=self.is_bear(),
        )
        self._snapshots.append(snapshot)
        self._equity_curve.append({"date": date, "equity": self.equity})

        if self.on_step:
            self.on_step(snapshot, self)

        self._bar_index += 1
        return snapshot

    def run(self, max_steps: int = 0) -> list[ReplaySnapshot]:
        """运行完整回放，返回所有 snapshot。max_steps=0 表示全部。"""
        steps = 0
        while not self.is_finished():
            self.step()
            steps += 1
            if max_steps and steps >= max_steps:
                break
        logger.info(f"Replay finished: {steps} steps, {len(self._trades)} trades")
        return self._snapshots

    def send_signal(self, action: str, price: float = 0.0, reason: str = ""):
        """外部（策略）调用：提交交易信号。replay/paper 模式下为次日开盘价成交。"""
        if action == "buy" and self._position == 0:
            price = price or self.df["close"].iloc[self._bar_index]
            size = int(self.cash / price)
            if size > 0:
                self._position = size
                self.cash -= size * price * 1.00025  # 佣金
                self._entry_price = price
                self._peak_price = price
                self._snapshots[-1].signal = "buy"
                self._snapshots[-1].signal_reason = reason

        elif action == "sell" and self._position > 0:
            price = price or self.df["close"].iloc[self._bar_index]
            self._close_position(price, self._snapshots[-1].date if self._snapshots else "", reason)

    def _close_position(self, price: float, date: str, reason: str):
        """平仓"""
        pnl = self._position * (price - self._entry_price)
        cost = self._position * price * (0.00025 + 0.0005)  # 佣金+印花税
        self.cash += self._position * price - cost
        pnl_pct = (price / self._entry_price - 1) * 100
        self._trades.append({
            "date": date, "action": "sell", "price": price,
            "pnl": pnl, "pnl_pct": pnl_pct, "reason": reason,
        })
        self._position = 0
        self._entry_price = 0

    def report(self) -> str:
        """生成回放结果报告"""
        n = len(self._trades)
        wins = sum(1 for t in self._trades if t["pnl"] > 0)
        final_equity = self.equity
        total_return = (final_equity / self.initial_cash - 1) * 100
        max_equity = self.initial_cash
        max_dd = 0.0
        for s in self._equity_curve:
            max_equity = max(max_equity, s["equity"])
            dd = (max_equity - s["equity"]) / max_equity
            max_dd = max(max_dd, dd)

        lines = [
            "=" * 50,
            "  Replay Report",
            "=" * 50,
            f"  Steps: {len(self._snapshots)} | Trades: {n}",
            f"  Return: {total_return:.1f}% | Max DD: {max_dd:.1%}",
            f"  Win rate: {wins}/{n} ({wins/n*100:.0f}%)" if n > 0 else "  No trades",
            f"  Final equity: {final_equity:,.0f}",
        ]
        if self._stop_reason:
            lines.append(f"  Stopped: {self._stop_reason}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# 命令行测试
# ============================================================
# python live/replay/engine.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fd

    df = fd("沪深300", "20230101", "20260524")

    print("=" * 60)
    print("ReplayEngine Demo — 模拟逐日推进")
    print("=" * 60)

    engine = ReplayEngine(df, cash=100000)

    # 简单的 MA 交叉信号策略
    snapshots = []
    fast_ma = None
    slow_ma = None

    for snapshot in engine.run():
        # 简易 MA 交叉策略（在回放循环内计算，无未来函数）
        i = snapshot.bar_index
        if i >= 20:
            fast_ma = df["close"].iloc[i-4:i+1].mean() if i >= 4 else df["close"].iloc[:i+1].mean()
            slow_ma = df["close"].iloc[i-19:i+1].mean() if i >= 19 else df["close"].iloc[:i+1].mean()
            if fast_ma > slow_ma and engine.position == 0:
                engine.send_signal("buy", snapshot.close, "MA cross up")
            elif fast_ma < slow_ma and engine.position > 0:
                engine.send_signal("sell", snapshot.close, "MA cross down")

        snapshots.append(snapshot)

        if snapshot.bar_index % 50 == 0 and snapshot.bar_index > 0:
            print(f"  [{snapshot.date}] equity={snapshot.equity:,.0f} "
                  f"pos={snapshot.position} bear={snapshot.is_bear_market}")

    print()
    print(engine.report())
