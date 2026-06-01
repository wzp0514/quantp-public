"""
盈亏分析页面 — 参考 BsStrategy 绩效评估 + MoneySkills 风险可视化

显示策略收益归因、风险指标、权益曲线。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def show():
    st.title("盈亏分析")
    st.caption("参考 BsStrategy 绩效评估 + MoneySkills 风险可视化")

    st.info("选择标的和策略后，展示完整的盈亏归因分析。")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("标的", ["沪深300", "中证500", "创业板指"])
    with col2:
        strategy = st.selectbox("策略", ["双均线交叉", "布林带回归", "动量策略", "均值回归", "网格交易"])
    with col3:
        cash = st.number_input("初始资金（万）", value=10, step=1) * 10000

    if st.button("开始分析", type="primary"):
        with st.spinner("运行回测 + 归因分析..."):
            from data.fetchers.fallback import fetch_index_daily_safe
            from backtest.engine.bt_runner import run_backtest
            from backtest.strategy_market import ALL_STRATEGIES
            from backtest.analysis.attribution import simple_attribution, compare_to_benchmark

            info = ALL_STRATEGIES[strategy]
            strategy_class = info["class"]

            df = fetch_index_daily_safe(symbol, "20230101", "20250601")
            result = run_backtest(strategy_class, df, initial_cash=cash, **info["params"])

            # ---- 收益指标 ----
            st.subheader("收益概览")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("总收益", f"{result['total_return']:.2%}")
            with m2:
                st.metric("年化收益", f"{result['annual_return']:.2%}")
            with m3:
                st.metric("最大回撤", f"{result['drawdown']:.2%}")
            with m4:
                sharpe = f"{result['sharpe']:.2f}" if result['sharpe'] else "N/A"
                st.metric("夏普比率", sharpe)

            # ---- 归因分析 ----
            st.subheader("绩效归因")
            st.caption("把收益拆成 市场Beta（大盘涨跌） + 超额Alpha（策略真本事）")

            try:
                bench_returns = df.set_index("date")["close"].pct_change().dropna()
                if not result["equity_df"].empty:
                    eq = result["equity_df"]
                    eq["daily_ret"] = eq["equity"].pct_change()
                    strategy_rets = eq.set_index("date")["daily_ret"].dropna()

                    attr = simple_attribution(strategy_rets, bench_returns)
                    if "error" not in attr:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            alpha_val = attr["annual_alpha"] * 100
                            st.metric("年化 Alpha", f"{alpha_val:+.2f}%",
                                     delta=f"{alpha_val:+.1f}%" if alpha_val > 0 else None)
                        with c2:
                            st.metric("Beta", f"{attr['beta']:.2f}")
                        with c3:
                            st.metric("R²", f"{attr['r_squared']:.2f}")

                        st.info(attr["interpretation"])

                        # 基准对比
                        buy_hold_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1
                        days = len(df)
                        buy_hold_annual = (1 + buy_hold_return) ** (252 / days) - 1 if days > 0 else 0
                        peak = df["close"].expanding().max()
                        dd = (df["close"] - peak) / peak
                        buy_hold_dd = dd.min()

                        report = compare_to_benchmark(
                            result["total_return"], result["annual_return"], result["drawdown"],
                            buy_hold_return, buy_hold_annual, buy_hold_dd,
                        )
                        st.code(report)
            except Exception as e:
                st.warning(f"归因分析失败: {e}")

            # ---- 权益曲线 ----
            st.subheader("权益曲线")
            if not result["equity_df"].empty:
                fig, ax = plt.subplots(figsize=(10, 3))
                plt.rcParams["font.sans-serif"] = ["SimHei"]
                plt.rcParams["axes.unicode_minus"] = False
                ax.plot(result["equity_df"]["date"], result["equity_df"]["equity"], linewidth=0.8)
                ax.set_title(f"{strategy} — 权益曲线")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

            # ---- 月度收益热力图 ----
            st.subheader("月度收益热力图")
            if not result["equity_df"].empty:
                eq = result["equity_df"].copy()
                eq["date"] = pd.to_datetime(eq["date"])
                eq.set_index("date", inplace=True)

                # 计算日收益
                daily_ret = eq["equity"].pct_change().dropna()

                # 构建月度收益矩阵
                monthly = daily_ret.resample("ME").apply(
                    lambda x: (1 + x).prod() - 1
                )
                monthly_df = monthly.to_frame("return")
                monthly_df["year"] = monthly_df.index.year
                monthly_df["month"] = monthly_df.index.month

                # 透视成热力图格式
                heatmap_data = monthly_df.pivot_table(
                    values="return", index="year", columns="month", aggfunc="first"
                )

                if not heatmap_data.empty:
                    fig, ax = plt.subplots(figsize=(10, max(3, len(heatmap_data) * 0.6)))
                    plt.rcParams["font.sans-serif"] = ["SimHei"]
                    plt.rcParams["axes.unicode_minus"] = False

                    # 绘制热力图
                    im = ax.imshow(heatmap_data.values * 100, cmap="RdYlGn",
                                   aspect="auto", vmin=-10, vmax=10)

                    # 标注
                    month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                                   "7月", "8月", "9月", "10月", "11月", "12月"]
                    year_labels = [str(y) for y in heatmap_data.index]

                    ax.set_xticks(range(12))
                    ax.set_xticklabels(month_labels, rotation=45)
                    ax.set_yticks(range(len(heatmap_data)))
                    ax.set_yticklabels(year_labels)

                    # 在每个单元格中标注数值
                    for y_idx in range(len(heatmap_data)):
                        for x_idx in range(12):
                            val = heatmap_data.iloc[y_idx, x_idx]
                            if not pd.isna(val):
                                color = "white" if abs(val) > 0.05 else "black"
                                ax.text(x_idx, y_idx, f"{val*100:+.1f}%",
                                       ha="center", va="center", fontsize=8, color=color)

                    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label("月度收益 (%)")

                    ax.set_title(f"{strategy} — 月度收益热力图（绿色=盈利, 红色=亏损）")
                    fig.tight_layout()
                    st.pyplot(fig)

                    # 统计信息
                    valid_months = monthly_df["return"].dropna()
                    pos_months = (valid_months > 0).sum()
                    total_months = len(valid_months)
                    win_month_pct = pos_months / total_months * 100 if total_months > 0 else 0
                    best_month = valid_months.max()
                    worst_month = valid_months.min()

                    hm1, hm2, hm3, hm4 = st.columns(4)
                    with hm1:
                        st.metric("盈利月占比", f"{win_month_pct:.0f}%")
                    with hm2:
                        st.metric("最佳月度", f"{best_month:.2%}")
                    with hm3:
                        st.metric("最差月度", f"{worst_month:.2%}")
                    with hm4:
                        avg_month = valid_months.mean()
                        st.metric("月均收益", f"{avg_month:.2%}")

            # ---- 交易统计 ----
            st.subheader("交易统计")
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                st.metric("总交易", result["total_trades"])
            with tc2:
                st.metric("盈利笔数", result["won_trades"])
            with tc3:
                st.metric("亏损笔数", result["lost_trades"])
            with tc4:
                st.metric("胜率", f"{result['win_rate']:.1%}")

    else:
        st.info("选择标的和策略后点击「开始分析」。")
        st.caption("分析包含: 收益概览 → 绩效归因（Alpha/Beta） → 基准对比 → 权益曲线 → 月度热力图 → 交易统计")
