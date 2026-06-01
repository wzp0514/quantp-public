"""
策略信号监控页面 — 实时信号 vs 回测预期对比

参考 Freqtrade 的实时交易面板。
显示: 信号频率、回测偏差（漂移检测）、信号质量评估。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def show():
    st.title("策略信号监控")
    st.caption("参考 Freqtrade 实时面板 — 信号频率 vs 回测预期")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.selectbox("标的", ["沪深300", "中证500", "创业板指", "上证50"])
    with col2:
        lookback = st.selectbox("观察周期", ["近1月", "近3月", "近6月", "近1年"], index=1)

    if st.button("拉取信号数据", type="primary"):
        with st.spinner("拉取数据并分析信号..."):
            try:
                from data.fetchers.fallback import fetch_index_daily_safe
                from backtest.strategy_market import ALL_STRATEGIES
                from backtest.engine.bt_runner import run_backtest

                df = fetch_index_daily_safe(symbol, "20200101")
                if df.empty:
                    st.error(f"无法获取 {symbol} 数据（所有数据源均失败）")
                    return

                source = df.attrs.get("source", "unknown")
                st.caption(f"数据源: {source}, {len(df)} 条, "
                          f"{pd.to_datetime(df['date']).min().date()} ~ "
                          f"{pd.to_datetime(df['date']).max().date()}")

                # 对每个可用策略跑回测，提取信号统计
                signals_data = []
                for name, info in ALL_STRATEGIES.items():
                    try:
                        strategy_class = info["class"]
                        result = run_backtest(
                            strategy_class, df, initial_cash=100000, **info.get("params", {})
                        )

                        trades_df = result.get("trades_df", pd.DataFrame())
                        if trades_df.empty:
                            continue

                        total_days = len(df)
                        trade_count = len(trades_df)
                        signals_per_month = trade_count / max(total_days / 21, 1)
                        win_count = len(trades_df[trades_df.get("pnl", 0) > 0]) if "pnl" in trades_df.columns else 0
                        win_rate = win_count / trade_count if trade_count > 0 else 0

                        signals_data.append({
                            "策略": name,
                            "类型": info.get("type", ""),
                            "交易次数": trade_count,
                            "月均信号": f"{signals_per_month:.1f}",
                            "胜率": f"{win_rate:.1%}",
                            "年化收益": f"{result['annual_return']:.2%}",
                            "最大回撤": f"{result['drawdown']:.2%}",
                            "夏普": f"{result['sharpe']:.2f}" if result['sharpe'] else "N/A",
                            "_signals_pm": signals_per_month,
                            "_winrate": win_rate,
                        })
                    except Exception as e:
                        st.debug(f"{name} 回测失败: {e}")

                if not signals_data:
                    st.warning("无可用策略信号")
                    return

                # ── 信号总览表 ──
                st.subheader("信号总览")
                display_df = pd.DataFrame(signals_data)
                st.dataframe(
                    display_df[[c for c in display_df.columns if not c.startswith("_")]],
                    use_container_width=True,
                    hide_index=True,
                )

                # ── 可视化: 信号频率 vs 年化收益 ──
                st.subheader("信号频率 vs 策略收益")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                plt.rcParams["font.sans-serif"] = ["SimHei"]
                plt.rcParams["axes.unicode_minus"] = False

                names = [d["策略"] for d in signals_data]
                sig_freq = [d["_signals_pm"] for d in signals_data]
                rets = [float(d["年化收益"].strip("%")) / 100 for d in signals_data]
                wrs = [d["_winrate"] for d in signals_data]
                colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

                # 频率 vs 收益
                axes[0].scatter(sig_freq, [r * 100 for r in rets], c=colors, s=100, edgecolors="white")
                for i, n in enumerate(names):
                    axes[0].annotate(n[:8], (sig_freq[i], rets[i] * 100), fontsize=8, alpha=0.8)
                axes[0].axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
                axes[0].set_xlabel("月均信号数")
                axes[0].set_ylabel("年化收益 (%)")
                axes[0].set_title("信号频率 vs 收益（越多≠越好）")
                axes[0].grid(True, alpha=0.3)

                # 信号频率 vs 胜率
                axes[1].scatter(sig_freq, [w * 100 for w in wrs], c=colors, s=100, edgecolors="white")
                for i, n in enumerate(names):
                    axes[1].annotate(n[:8], (sig_freq[i], wrs[i] * 100), fontsize=8, alpha=0.8)
                axes[1].set_xlabel("月均信号数")
                axes[1].set_ylabel("胜率 (%)")
                axes[1].set_title("信号频率 vs 胜率")
                axes[1].grid(True, alpha=0.3)

                fig.tight_layout()
                st.pyplot(fig)

                # ── 漂移检测提示 ──
                st.divider()
                st.subheader("策略漂移提示")
                st.info(
                    "实盘中使用 Guardian 的 check_strategy_drift() 检测：\n\n"
                    "如果实盘信号频率偏离回测 > 50%，或胜率偏离 > 50%，"
                    "系统自动告警。这说明策略可能过拟合或市场环境已改变。\n\n"
                    "在 interactive.py 中选择菜单项 10（启动守护）可开启自动漂移检测。"
                )

                # ── 数据质量 ──
                st.divider()
                st.subheader("数据质量")
                from data.fetchers.fallback import check_data_completeness, get_source_health
                completeness = check_data_completeness(df)
                health = get_source_health()

                c1, c2 = st.columns(2)
                with c1:
                    if completeness["complete"]:
                        st.success(f"数据完整 — 最近 {completeness['missing_days']} 个交易日缺失")
                    else:
                        st.warning(completeness["warning"])
                    st.caption(f"最新数据日期: {completeness['last_date']}")

                with c2:
                    for src, h in health.items():
                        icon = "OK" if h["status"] == "ok" else ("DOWN" if h["status"] == "down" else "?")
                        st.metric(f"数据源: {src}", icon)

            except Exception as e:
                st.error(f"分析失败: {e}")

    else:
        st.info("点击「拉取信号数据」查看各策略在选定标的上的信号统计。")
        st.markdown("""
        **信号监控要点：**
        - 月均信号太少（<0.5）→ 策略太保守，资金利用率低
        - 月均信号太多（>10）→ 可能过度交易，手续费吃掉利润
        - 胜率 < 30% → 策略可能需要重新审视入场逻辑
        - 信号频率偏离回测 > 50% → 警告：策略可能过拟合
        """)
