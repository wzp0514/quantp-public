"""
纸上交易 vs 回测 差异分析

四维对比：
  1. 收益偏差 — 总收益率/夏普/最大回撤
  2. 信号对齐 — 信号日期的匹配度
  3. 成交价差 — 同笔交易的成交价偏差
  4. 回撤相关性 — 每日回撤序列的Pearson相关系数
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DiffReport:
    """纸上 vs 回测差异报告。"""
    strategy: str = ""
    symbol: str = ""
    period: tuple = ("", "")

    # 收益偏差
    backtest_return: float = 0
    paper_return: float = 0
    return_deviation: float = 0           # abs(paper - bt) / abs(bt)

    backtest_drawdown: float = 0
    paper_drawdown: float = 0
    drawdown_deviation: float = 0

    backtest_sharpe: float = 0
    paper_sharpe: float = 0

    # 信号对齐
    backtest_signals: int = 0
    paper_signals: int = 0
    matching_signals: int = 0
    extra_bt_signals: int = 0
    extra_paper_signals: int = 0
    signal_alignment_pct: float = 0

    # 成交价差
    avg_fill_price_diff: float = 0        # 平均 |paper_price - bt_price| / bt_price

    # 回撤序列
    daily_return_corr: float = 0          # 日收益率相关系数
    drawdown_series_corr: float = 0       # 回撤序列相关系数

    # 标记
    warnings: list[str] = field(default_factory=list)
    grade: str = ""                       # A/B/C/D/F


def compare_paper_backtest(
    paper_result: dict,
    backtest_result: dict,
    strategy_name: str = "",
    symbol: str = "",
) -> DiffReport:
    """四维对比纸上交易与回测结果。

    参数
    ----------
    paper_result : dict
        PaperTrader.run() 的返回值
    backtest_result : dict
        run_backtest() 的返回值
    strategy_name : str
        策略名（可选）
    symbol : str
        标的代码（可选）

    返回
    -------
    DiffReport
    """
    r = DiffReport(strategy=strategy_name, symbol=symbol)

    # ── 1. 收益维度 ──────────────────────────────────────
    _compare_returns(r, paper_result, backtest_result)

    # ── 2. 信号维度 ──────────────────────────────────────
    _compare_signals(r, paper_result, backtest_result)

    # ── 3. 成交价维度 ────────────────────────────────────
    _compare_fills(r, paper_result, backtest_result)

    # ── 4. 回撤维度 ──────────────────────────────────────
    _compare_drawdowns(r, paper_result, backtest_result)

    # ── 综合评级 ─────────────────────────────────────────
    _grade(r)

    return r


def _compare_returns(r: DiffReport, paper: dict, bt: dict):
    """收益维度对比。"""
    r.paper_return = paper.get("total_return", 0)
    r.backtest_return = bt.get("total_return", 0)

    denom = max(abs(r.backtest_return), 0.001)
    r.return_deviation = abs(r.paper_return - r.backtest_return) / denom

    if r.return_deviation > 0.30:
        r.warnings.append(f"收益偏差 {r.return_deviation:.1%} > 30%，需排查策略逻辑一致性")

    # 夏普
    r.paper_sharpe = _calc_sharpe_from_log(paper.get("daily_log", []))
    r.backtest_sharpe = bt.get("sharpe", bt.get("sharpe_ratio", 0))

    # 最大回撤
    r.paper_drawdown = _calc_max_drawdown(paper.get("daily_log", []))
    r.backtest_drawdown = bt.get("max_drawdown", bt.get("max_dd", 0))

    dd_denom = max(abs(r.backtest_drawdown), 0.001)
    r.drawdown_deviation = abs(r.paper_drawdown - r.backtest_drawdown) / dd_denom

    if r.drawdown_deviation > 0.50:
        r.warnings.append(f"回撤偏差 {r.drawdown_deviation:.1%} > 50%，风控执行可能不一致")


def _compare_signals(r: DiffReport, paper: dict, bt: dict):
    """信号维度对比。"""
    # 纸上信号日期
    paper_signals = paper.get("signals", [])
    paper_dates = set()
    for s in paper_signals:
        d = s.get("date", "")
        if hasattr(d, "strftime"):
            d = d.strftime("%Y-%m-%d")
        paper_dates.add(str(d)[:10])

    r.paper_signals = len(paper_dates)

    # 回测信号日期
    trades_df = bt.get("trades_df", pd.DataFrame())
    bt_dates = set()
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            d = t.get("date", "")
            if hasattr(d, "strftime"):
                d = d.strftime("%Y-%m-%d")
            bt_dates.add(str(d)[:10])

    r.backtest_signals = len(bt_dates)

    # 匹配/多余
    r.matching_signals = len(paper_dates & bt_dates)
    r.extra_bt_signals = len(bt_dates - paper_dates)
    r.extra_paper_signals = len(paper_dates - bt_dates)

    total = max(len(paper_dates | bt_dates), 1)
    r.signal_alignment_pct = r.matching_signals / total

    if r.signal_alignment_pct < 0.80:
        r.warnings.append(f"信号对齐率 {r.signal_alignment_pct:.0%} < 80%，策略执行路径可能不同")


def _compare_fills(r: DiffReport, paper: dict, bt: dict):
    """成交价维度对比。"""
    # 纸上成交价来自订单
    paper_orders = paper.get("orders", [])
    paper_fills = {}
    for o in paper_orders:
        if o.get("status") == "filled" and o.get("price"):
            d = o.get("filled_at", o.get("created_at", ""))
            if hasattr(d, "strftime"):
                d = d.strftime("%Y-%m-%d")
            paper_fills[str(d)[:10]] = o["price"]

    # 回测成交价
    trades_df = bt.get("trades_df", pd.DataFrame())
    bt_fills = {}
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            d = t.get("date", "")
            if hasattr(d, "strftime"):
                d = d.strftime("%Y-%m-%d")
            price = t.get("price", t.get("fill_price", 0))
            bt_fills[str(d)[:10]] = float(price or 0)

    diffs = []
    for d in paper_fills:
        if d in bt_fills and bt_fills[d] > 0:
            diffs.append(abs(paper_fills[d] - bt_fills[d]) / bt_fills[d])

    r.avg_fill_price_diff = float(np.mean(diffs)) if diffs else 0

    if r.avg_fill_price_diff > 0.03:
        r.warnings.append(f"平均成交价差 {r.avg_fill_price_diff:.1%} > 3%，模拟成交行为显著不同")


def _compare_drawdowns(r: DiffReport, paper: dict, bt: dict):
    """回撤序列维度对比。"""
    paper_log = paper.get("daily_log", [])
    if not paper_log:
        return

    # 纸上日收益率序列
    paper_returns = []
    prev_val = paper.get("initial_cash", 100000)
    for entry in paper_log:
        close = entry.get("close", entry.get("close_price", 0))
        pos_size = entry.get("position_size", 0)
        pos_cost = entry.get("position_cost", 0)
        cash = entry.get("cash", prev_val)
        val = cash + pos_size * (close if close > 0 else pos_cost)
        if prev_val > 0:
            paper_returns.append(val / prev_val - 1)
        prev_val = val

    # 回测日收益率序列（从 equity_curve）
    bt_equity = bt.get("equity_curve", bt.get("equity", []))
    bt_returns = []
    if isinstance(bt_equity, list) and len(bt_equity) > 1:
        bt_returns = [(bt_equity[i] / bt_equity[i - 1] - 1) for i in range(1, len(bt_equity))]
    elif isinstance(bt_equity, pd.DataFrame) and not bt_equity.empty:
        col = "equity" if "equity" in bt_equity.columns else bt_equity.columns[0]
        vals = bt_equity[col].values
        bt_returns = [(vals[i] / vals[i - 1] - 1) for i in range(1, len(vals))]

    # 计算相关系数
    if paper_returns and bt_returns:
        min_len = min(len(paper_returns), len(bt_returns))
        try:
            r.daily_return_corr = float(np.corrcoef(
                paper_returns[:min_len], bt_returns[:min_len]
            )[0, 1])
            if np.isnan(r.daily_return_corr):
                r.daily_return_corr = 0
        except Exception:
            r.daily_return_corr = 0

    # 回撤序列相关性
    paper_dd = _calc_drawdown_series(paper_returns)
    bt_dd = _calc_drawdown_series(bt_returns)
    if paper_dd and bt_dd:
        min_len = min(len(paper_dd), len(bt_dd))
        try:
            r.drawdown_series_corr = float(np.corrcoef(
                paper_dd[:min_len], bt_dd[:min_len]
            )[0, 1])
            if np.isnan(r.drawdown_series_corr):
                r.drawdown_series_corr = 0
        except Exception:
            r.drawdown_series_corr = 0

    if r.daily_return_corr < 0.70:
        r.warnings.append(f"日收益率相关性 {r.daily_return_corr:.2f} < 0.70，两套执行逻辑存在系统偏差")


def _grade(r: DiffReport):
    """综合评级。"""
    score = 100
    if r.return_deviation > 0.30:
        score -= 30
    elif r.return_deviation > 0.15:
        score -= 15

    if r.signal_alignment_pct < 0.80:
        score -= 25
    elif r.signal_alignment_pct < 0.90:
        score -= 10

    if r.avg_fill_price_diff > 0.03:
        score -= 20
    elif r.avg_fill_price_diff > 0.01:
        score -= 10

    if r.daily_return_corr < 0.70:
        score -= 15
    elif r.daily_return_corr < 0.85:
        score -= 5

    r.grade = ("A" if score >= 90 else "B" if score >= 75
               else "C" if score >= 60 else "D" if score >= 40 else "F")


def generate_diff_text(report: DiffReport) -> str:
    """生成人类可读的对比报告文本。"""
    lines = [
        "=" * 64,
        f"  纸上交易 vs 回测 差异分析: {report.strategy}",
        "=" * 64,
        "",
        "── 1. 收益维度 ──",
        f"  回测收益:     {report.backtest_return:+.2%}",
        f"  纸上收益:     {report.paper_return:+.2%}",
        f"  收益偏差:     {report.return_deviation:.2%}",
        f"  回测夏普:     {report.backtest_sharpe:.2f}",
        f"  纸上夏普:     {report.paper_sharpe:.2f}",
        f"  回测回撤:     {report.backtest_drawdown:.2%}",
        f"  纸上回撤:     {report.paper_drawdown:.2%}",
        "",
        "── 2. 信号维度 ──",
        f"  回测信号数:   {report.backtest_signals}",
        f"  纸上信号数:   {report.paper_signals}",
        f"  匹配信号:     {report.matching_signals}",
        f"  仅回测有:     {report.extra_bt_signals}",
        f"  仅纸上有:     {report.extra_paper_signals}",
        f"  信号对齐率:   {report.signal_alignment_pct:.1%}",
        "",
        "── 3. 成交价维度 ──",
        f"  平均价差:     {report.avg_fill_price_diff:.3%}",
        "",
        "── 4. 回撤维度 ──",
        f"  日收益相关:   {report.daily_return_corr:.3f}",
        f"  回撤序列相关: {report.drawdown_series_corr:.3f}",
        "",
        f"── 综合评级: {report.grade} ──",
    ]

    if report.warnings:
        lines.append("")
        lines.append("⚠️ 警告:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


# ── 辅助计算 ──────────────────────────────────────────────

def _calc_sharpe_from_log(daily_log: list[dict]) -> float:
    """从每日日志计算夏普比率。"""
    if len(daily_log) < 2:
        return 0
    returns = []
    prev_val = None
    for entry in daily_log:
        close = entry.get("close", entry.get("close_price", 0))
        pos = entry.get("position_size", 0)
        cash = entry.get("cash", 0)
        val = cash + pos * close
        if prev_val and prev_val > 0:
            returns.append(val / prev_val - 1)
        prev_val = val
    if not returns:
        return 0
    arr = np.array(returns)
    mean = arr.mean()
    std = arr.std(ddof=1)
    return float(mean / std * np.sqrt(252)) if std > 0 else 0


def _calc_max_drawdown(daily_log: list[dict]) -> float:
    """从每日日志计算最大回撤。"""
    if not daily_log:
        return 0
    peak = 0
    max_dd = 0
    for entry in daily_log:
        close = entry.get("close", entry.get("close_price", 0))
        pos = entry.get("position_size", 0)
        cash = entry.get("cash", 0)
        val = cash + pos * close
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return abs(max_dd)


def _calc_drawdown_series(returns: list[float]) -> list[float]:
    """从收益率序列计算回撤序列。"""
    if not returns:
        return []
    cum = [1.0]
    for r in returns:
        cum.append(cum[-1] * (1 + r))
    peak = 0
    dds = []
    for v in cum:
        if v > peak:
            peak = v
        dds.append((v - peak) / peak if peak > 0 else 0)
    return dds


# ── 命令行测试 ────────────────────────────────────────────
# python live/analysis/paper_diff.py

if __name__ == "__main__":
    # 用假数据演示
    paper = {
        "total_return": 0.12,
        "sharpe": 1.1,
        "max_drawdown": 0.15,
        "initial_cash": 100000,
        "signals": [
            {"date": "2025-01-15", "signal": "buy"},
            {"date": "2025-02-20", "signal": "sell"},
        ],
        "orders": [
            {"status": "filled", "price": 3500.0, "filled_at": "2025-01-16"},
            {"status": "filled", "price": 3700.0, "filled_at": "2025-02-21"},
        ],
        "daily_log": [
            {"date": "2025-01-15", "close": 3500, "cash": 100000, "position_size": 0, "position_cost": 0},
            {"date": "2025-01-16", "close": 3520, "cash": 30000, "position_size": 20, "position_cost": 3500},
            {"date": "2025-02-20", "close": 3700, "cash": 30000, "position_size": 20, "position_cost": 3500},
            {"date": "2025-02-21", "close": 3680, "cash": 104000, "position_size": 0, "position_cost": 0},
        ],
    }

    bt = {
        "total_return": 0.15,
        "sharpe": 1.3,
        "max_drawdown": 0.12,
        "trades_df": pd.DataFrame([
            {"date": "2025-01-15", "type": "buy", "price": 3520},
            {"date": "2025-02-20", "type": "sell", "price": 3680},
        ]),
    }

    report = compare_paper_backtest(paper, bt, "双均线策略", "000300")
    print(generate_diff_text(report))
