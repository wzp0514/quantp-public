"""
回测报告生成

从 bt_runner.run_backtest() 返回的结果中提取关键指标，
生成可读的报告文本和图表。

指标清单：
  - 总收益率、年化收益率
  - 最大回撤、卡玛比率（年化收益/最大回撤）
  - 夏普比率
  - 交易次数、胜率、盈亏比、利润因子
  - 权益曲线图
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_report(result: dict, regime_info: dict = None) -> str:
    """
    生成纯文本回测报告

    参数
    ----------
    result : dict
        bt_runner.run_backtest() 的返回值
    regime_info : dict, optional
        regime_filter.regime_summary() 的返回值，附加区制分析

    返回
    -------
    str : 格式化的报告文本
    """
    total_ret = result.get("total_return", 0)
    annual_ret = result.get("annual_return", 0)
    drawdown = result.get("drawdown", 0)
    sharpe = result.get("sharpe")
    total_trades = result.get("total_trades", 0)
    won = result.get("won_trades", 0)
    lost = result.get("lost_trades", 0)
    win_rate = result.get("win_rate", 0)
    final_value = result.get("final_value", 0)

    # 卡玛比率 = 年化收益 / 最大回撤（绝对值）
    calmar = annual_ret / abs(drawdown) if drawdown else float("inf")

    # 盈亏比：平均盈利 / 平均亏损
    trades_df = result.get("trades_df", None)
    profit_factor = _calc_profit_factor(trades_df)

    # 基准对比（策略 vs 买入持有）
    bh_return = _calc_benchmark_return(result)
    alpha = annual_ret - bh_return

    lines = [
        "=" * 60,
        "                    回测报告",
        "=" * 60,
        "",
        "收益指标",
        "-" * 40,
        f"  总收益率:    {total_ret * 100:8.2f}%",
        f"  年化收益率:  {annual_ret * 100:8.2f}%",
        f"  最终资金:    {final_value:12,.2f} 元",
        "",
        "基准对比 (vs 买入持有)",
        "-" * 40,
        f"  策略年化:    {annual_ret * 100:8.2f}%",
        f"  基准年化:    {bh_return * 100:8.2f}%",
        f"  Alpha(超额):  {(alpha) * 100:+8.2f}%  ({'跑赢' if alpha > 0 else '跑输'}基准)",
        "",
        "风险指标",
        "-" * 40,
        f"  最大回撤:    {drawdown * 100:8.2f}%",
        f"  夏普比率:    {sharpe if sharpe else 'N/A':>12}",
        f"  卡玛比率:    {calmar:12.2f}  (年化收益/最大回撤, >0.5及格, >1优秀)",
        "",
        "交易统计",
        "-" * 40,
        f"  总交易次数:  {total_trades:>12}",
        f"  盈利次数:    {won:>12}",
        f"  亏损次数:    {lost:>12}",
        f"  胜率:        {win_rate * 100:11.1f}%",
        f"  利润因子:    {profit_factor:12.2f}  (>1.3及格, >2.0优秀)",
        "",
        "评价",
        "-" * 40,
    ]

    # 自动评价
    if annual_ret < 0.03:
        lines.append("  年化收益低于 3%（及格线），策略盈利能力不足")
    elif annual_ret > 0.15:
        lines.append("  年化收益超过 15%（优秀线），收益率表现优秀")

    if abs(drawdown) > 0.25:
        lines.append("  最大回撤超过 25%（超标），风险过大")
    elif abs(drawdown) < 0.15:
        lines.append("  最大回撤小于 15%（优秀），风险控制良好")

    if sharpe and sharpe < 0.5:
        lines.append("  夏普比率偏低（<0.5），单位风险回报不足")
    elif sharpe and sharpe > 1.5:
        lines.append("  夏普比率 > 1.5（优秀），风险调整后收益优秀")

    if profit_factor < 1.3:
        lines.append("  利润因子 < 1.3（不及格），盈利不足以覆盖亏损")
    elif profit_factor > 2.0:
        lines.append("  利润因子 > 2.0（优秀），盈利能力强劲")

    lines.append("")
    lines.append("=" * 60)

    lines.append("")
    lines.append("注意事项")
    lines.append("-" * 40)
    lines.append("  幸存者偏差: 回测使用当前仍在交易的标的，收益可能被高估10-30%。")
    lines.append("  实盘建议: 回测收益打7折作为预期。参照05文档陷阱3。")
    lines.append("  止损假设: 回测中止损以固定0.1%滑点成交，实盘极端行情可能无法成交。")
    lines.append("  参照05文档失败2(500万归零)+失败4(马丁爆仓)。")

    # ── 区制分析（如果有） ──
    if regime_info and regime_info.get("by_regime"):
        lines.append("")
        lines.append("区制分布（策略在什么市场环境下交易）")
        lines.append("-" * 40)
        by_regime = regime_info["by_regime"]
        for regime in ["Bull", "Sideways", "Bear"]:
            r = by_regime.get(regime, {})
            cnt = r.get("count", 0)
            buy_cnt = r.get("buy_count", 0)
            if cnt > 0:
                lines.append(f"  {regime:<9} 卖出 {cnt} 笔, 买入 {buy_cnt} 笔")
            else:
                lines.append(f"  {regime:<9} 无交易")
        lines.append(f"  建议: {regime_info.get('recommendation', '')}")

    # ── 验证结果（如果有） ──
    validation = result.get("validation")
    if validation:
        lines.append("")
        lines.append("自动验证")
        lines.append("-" * 40)
        oos = validation.get("out_of_sample", {})
        if oos:
            lines.append(f"  样本外: {'PASS' if oos.get('passed') else 'FAIL'} | {oos.get('detail', '')}")
        st = validation.get("statistical", {})
        if st:
            lines.append(f"  显著性: {'PASS' if st.get('significant') else 'FAIL'} | {st.get('interpretation', '')}")
        lines.append(f"  综合: {validation.get('overall', 'UNKNOWN')}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def scorecard(result: dict) -> str:
    """
    生成策略评分卡片 — 新手友好，颜色+ PASS/FAIL + 一句话评价

    评分权重: 年化收益30% 最大回撤30% 夏普比率20% 利润因子10% 胜率10%
    """
    total_ret = result.get("total_return", 0)
    annual_ret = result.get("annual_return", 0)
    drawdown = abs(result.get("drawdown", 0))
    sharpe = result.get("sharpe") or 0
    total_trades = result.get("total_trades", 0)
    won = result.get("won_trades", 0)
    lost = result.get("lost_trades", 0)
    win_rate = result.get("win_rate", 0)
    trades_df = result.get("trades_df")
    pf = _calc_profit_factor_from_result(result)

    # ── 分项评分（0-100，clamp到0-100） ──
    def _clamp(v, lo=0, hi=100):
        return max(lo, min(hi, v))

    ret_score = _clamp(annual_ret / 0.15 * 100) if annual_ret > 0 else 0
    dd_score = _clamp((1 - drawdown / 0.25) * 100)
    sharpe_score = _clamp(sharpe / 1.5 * 100) if sharpe > 0 else 0
    pf_score = _clamp(pf / 2.0 * 100) if pf > 0 else 0
    wr_score = _clamp(win_rate / 0.60 * 100)

    total = ret_score * 0.3 + dd_score * 0.3 + sharpe_score * 0.2 + pf_score * 0.1 + wr_score * 0.1

    # ── 分项阈值 (readme 4.4 节) ──
    def _grade(label, value, pass_line, good_line, unit, higher_is_better):
        if higher_is_better:
            if value >= good_line:
                return f"  [PASS 优秀] {label}: {value:.2f}{unit} (优秀线: {good_line:.2f}{unit})"
            elif value >= pass_line:
                return f"  [PASS 及格] {label}: {value:.2f}{unit} (及格线: {pass_line:.2f}{unit})"
            else:
                return f"  [FAIL]      {label}: {value:.2f}{unit} (不及格，需 {pass_line:.2f}{unit})"
        else:
            if value <= good_line:
                return f"  [PASS 优秀] {label}: {value:.2f}{unit} (优秀线: ≤{good_line:.2f}{unit})"
            elif value <= pass_line:
                return f"  [PASS 及格] {label}: {value:.2f}{unit} (及格线: ≤{pass_line:.2f}{unit})"
            else:
                return f"  [FAIL]      {label}: {value:.2f}{unit} (超标，需 ≤{pass_line:.2f}{unit})"

    checks = [
        _grade("年化收益", annual_ret, 0.03, 0.15, "%", True),
        _grade("最大回撤", drawdown, 0.25, 0.15, "%", False),
        _grade("夏普比率", sharpe, 1.0, 1.5, "", True),
        _grade("利润因子", pf, 1.3, 2.0, "", True),
        _grade("胜率    ", win_rate, 0.35, 0.50, "%", True),
    ]

    # ── 综合评价 ──
    if total >= 75:
        judge = "优秀——综合表现突出，可以考虑纸上交易验证"
    elif total >= 50:
        judge = "中等——有一定盈利能力，建议优化参数或换策略"
    elif total >= 30:
        judge = "偏弱——建议继续挖掘其他策略"
    else:
        judge = "较差——不建议用于实盘"

    failed = sum(1 for c in checks if "[FAIL]" in c)
    if failed > 0:
        judge += f"（{failed}项未达标）"

    lines = [
        "",
        "=" * 50,
        f"  策略评分卡  |  综合: {total:.0f}/100  |  {judge}",
        "=" * 50,
    ] + checks + [
        "-" * 50,
        f"  交易次数: {total_trades:>5}  盈利: {won:>4}  亏损: {lost:>4}",
        "=" * 50,
    ]

    return "\n".join(lines)


def _calc_profit_factor_from_result(result: dict) -> float:
    """从回测结果计算利润因子"""
    trades_df = result.get("trades_df")
    if trades_df is None or trades_df.empty:
        return 0.0
    try:
        buys = trades_df[trades_df["type"] == "buy"]
        sells = trades_df[trades_df["type"] == "sell"]
        if len(buys) == 0 or len(sells) == 0:
            return 0.0
        avg_buy = buys["price"].mean()
        avg_sell = sells["price"].mean()
        total_profit = (avg_sell - avg_buy) * len(sells)
        total_loss = abs(avg_buy - avg_sell) * len(sells)
        if total_loss <= 0:
            return 2.0  # 无亏损 → 优秀
        return total_profit / total_loss if total_profit > 0 else 0.0
    except Exception:
        return 0.0


def plot_equity_curve(
    result: dict,
    save_path: str = "",
    title: str = "策略权益曲线",
) -> None:
    """
    绘制权益曲线（账户资金变化图）

    参数
    ----------
    result : dict
        bt_runner.run_backtest() 的返回值
    save_path : str
        保存路径，如 "notebooks/equity.png"。不填则弹出窗口显示
    title : str
        图表标题
    """
    equity_df = result.get("equity_df")

    if equity_df is None or equity_df.empty:
        # 如果没有 equity_df，尝试从 Backtrader 自带的绘图
        print("无权益曲线数据，请确保回测结果中包含 equity_df")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(equity_df["date"], equity_df["equity"], linewidth=0.8, color="#1f77b4")
    ax.fill_between(
        equity_df["date"],
        equity_df["equity"],
        equity_df["equity"].iloc[0],
        alpha=0.1,
        color="#1f77b4",
    )

    # 标记最高点和最低点
    peak_idx = equity_df["equity"].idxmax()
    trough_idx = equity_df["equity"].idxmin()
    ax.scatter(
        equity_df["date"].iloc[peak_idx],
        equity_df["equity"].iloc[peak_idx],
        color="green", s=30, zorder=5, label=f'最高: {equity_df["equity"].iloc[peak_idx]:,.0f}'
    )

    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("账户资金（元）")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"权益曲线已保存: {save_path}")
    else:
        plt.show()


# ============================================================
# K线图生成（mplfinance，参考 TradingView 的静态替代）
# ============================================================

def plot_kline(
    df,
    trades=None,
    indicators=None,
    save_path="",
    title="",
    days=120,
):
    """
    生成 K 线图（纯 matplotlib，零额外依赖）。

    含成交量柱状图 + 均线叠加 + 买卖点标记。
    参考 TradingView 的静态替代——生成 PNG 文件，不需要 GUI/浏览器。

    用法
    --------
    >>> plot_kline(df, indicators=['ma20','ma60'], days=120, save_path='kline.png')
    """
    ohlc = df[["date", "open", "high", "low", "close", "volume"]].copy()
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    ohlc = ohlc.set_index("date").sort_index()
    if days and len(ohlc) > days:
        ohlc = ohlc.iloc[-days:]

    close = ohlc["close"].values
    idx = range(len(ohlc))
    dates = [d.strftime("%m/%d") if hasattr(d, "strftime") else str(d)[:5]
             for d in ohlc.index]
    date_labels = ohlc.index

    # 双面板布局: 上面K线+均线, 下面成交量
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ---- K线 (candlestick) ----
    width = 0.6
    for i in idx:
        o, h, l, c = ohlc["open"].iloc[i], ohlc["high"].iloc[i], ohlc["low"].iloc[i], ohlc["close"].iloc[i]
        color = "red" if c >= o else "green"
        # 影线
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
        # 实体
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height > 0:
            ax1.add_patch(plt.Rectangle((i - width / 2, body_bottom), width, body_height,
                                        facecolor=color if c >= o else "white",
                                        edgecolor=color, linewidth=0.5))
        else:
            ax1.plot([i - width / 2, i + width / 2], [c, c], color=color, linewidth=0.8)

    # ---- 均线 ----
    if indicators:
        colors = {"ma5": "#f5a623", "ma10": "#e8541e", "ma20": "#4a90d9",
                  "ma60": "#7b68ee", "ma120": "#2ca02c"}
        for ind in indicators:
            if ind.startswith("ma"):
                period = int(ind[2:])
                ma = ohlc["close"].rolling(period).mean()
                ax1.plot(idx, ma.values, color=colors.get(ind, "gray"),
                        linewidth=0.8, label=ind.upper())
        ax1.legend(loc="upper left", fontsize=8)

    # ---- 买卖标记 ----
    if trades:
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for t in trades:
            try:
                d = pd.Timestamp(t.get("date", t.get("entry_date", "")))
                p = float(t.get("price", t.get("entry_price", 0)))
                action = str(t.get("action", t.get("side", ""))).lower()
                if d in ohlc.index:
                    pos = list(ohlc.index).index(d)
                    if "buy" in action or "long" in action or "entry" in action:
                        buy_x.append(pos); buy_y.append(p)
                    elif "sell" in action or "exit" in action:
                        sell_x.append(pos); sell_y.append(p)
            except Exception:
                continue
        if buy_x:
            ax1.scatter(buy_x, buy_y, marker="^", color="red", s=60, zorder=5, label="Buy")
        if sell_x:
            ax1.scatter(sell_x, sell_y, marker="v", color="green", s=60, zorder=5, label="Sell")
        if buy_x or sell_x:
            ax1.legend(loc="upper left", fontsize=8)

    # ---- 成交量柱状图 ----
    vol = ohlc["volume"].values
    colors_vol = ["red" if ohlc["close"].iloc[i] >= ohlc["open"].iloc[i] else "green" for i in idx]
    ax2.bar(idx, vol, width=width, color=colors_vol, alpha=0.6)
    ax2.set_ylabel("Volume", fontsize=9)

    # ---- 格式化 ----
    tick_step = max(1, len(idx) // 10)
    tick_positions = idx[::tick_step]
    tick_labels = [dates[i] for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=30, fontsize=8)
    ax1.set_ylabel("Price", fontsize=10)
    ax1.set_title(title or "K-line Chart", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if not save_path:
        save_path = f"reports/kline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    import os as _os
    _os.makedirs(_os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[report] K-line saved: {save_path}")
    return save_path


def _calc_benchmark_return(result: dict) -> float:
    """
    计算基准收益 = 买入持有的年化收益率。
    从回测结果的 equity curve 推算（第一笔到最后笔的总收益年化）。
    如果没有权益曲线数据，返回 0。
    """
    equity = result.get("equity_curve", result.get("equity_df"))
    if equity is None or (hasattr(equity, '__len__') and len(equity) == 0):
        return 0.0
    try:
        if hasattr(equity, 'iloc'):
            start_val = equity.iloc[0].get("equity", 0) if hasattr(equity.iloc[0], 'get') else equity.iloc[0]
            end_val = equity.iloc[-1].get("equity", 0) if hasattr(equity.iloc[-1], 'get') else equity.iloc[-1]
        elif isinstance(equity, list) and len(equity) > 0:
            start_val = equity[0].get("equity", result.get("start_value", 100000))
            end_val = equity[-1].get("equity", result.get("end_value", 100000))
        else:
            return 0.0

        total_days = result.get("total_bars", 252)
        if total_days > 0 and start_val > 0:
            total_ret = (end_val / start_val) - 1
            years = total_days / 252
            if years > 0:
                return (1 + total_ret) ** (1 / years) - 1
    except Exception:
        pass
    return 0.0


def _calc_profit_factor(trades_df) -> float:
    """利润因子 = 总盈利 / 总亏损（从 trades_df 配对计算）"""
    if trades_df is None or trades_df.empty:
        return 0.0
    try:
        # trades_df 可能包含 buy/sell 行，需要配对
        if "pnl" in trades_df.columns:
            wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
            losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
            return wins / losses if losses > 0 else float("inf")
        # 如果只有价格和 size，配对计算
        if "price" in trades_df.columns and "size" in trades_df.columns:
            buys = trades_df[trades_df["size"] > 0]
            sells = trades_df[trades_df["size"] < 0]
            if len(buys) > 0 and len(sells) > 0:
                win_trades = sum(1 for i in range(min(len(buys), len(sells)))
                                if sells.iloc[i]["price"] > buys.iloc[i]["price"])
                return win_trades / max(len(buys), 1)
    except Exception:
        pass
    return 0.0


# ============================================================
# 命令行测试
# ============================================================
# python backtest/analysis/report.py

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.engine.bt_runner import run_backtest
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy

    print("运行双均线策略回测...")
    df = fetch_index_daily("沪深300", "20230101", "20250601")
    result = run_backtest(
        MaCrossStrategy, df,
        initial_cash=100000,
        fast=5, slow=20,
    )

    # 生成并打印报告
    report = generate_report(result)
    print(report)

    # 画权益曲线
    plot_equity_curve(result, save_path="notebooks/equity_ma_cross.png")
