"""
策略对比页面 — 参考 Sattoro Hub 并行分析引擎

跑全部策略，展示排名表、雷达图对比、各策略权益曲线。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def show(symbol: str, cash: float):
    st.title("策略对比")

    # 可用策略
    from backtest.strategy_market import ALL_STRATEGIES

    strategy_names = list(ALL_STRATEGIES.keys())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption("参考 Sattoro Hub — 12 个并行分析引擎。此处全部策略同时跑。")
    with col2:
        start_date = st.date_input("开始日期", value=pd.Timestamp("2023-01-01"))
    with col3:
        end_date = st.date_input("结束日期", value=pd.Timestamp("2025-06-01"))

    run_clicked = st.button("开始策略大比武", type="primary") or st.session_state.get("run_shootout", False)

    if run_clicked:
        st.session_state["run_shootout"] = False

        with st.spinner("拉取数据..."):
            from data.fetchers.fallback import fetch_index_daily_safe
            df = fetch_index_daily_safe(
                symbol,
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d"),
            )

        st.info(f"数据: {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")

        with st.spinner(f"正在并行回测 {len(strategy_names)} 个策略..."):
            from backtest.analysis.shootout import run_shootout
            result = run_shootout(df, cash=cash)

        ranking = result["ranking"]

        # ---- 排名表 ----
        st.subheader("排名")
        rank_data = []
        for i, r in enumerate(ranking):
            rank_data.append({
                "排名": i + 1,
                "策略": r["name"],
                "类型": r["type"],
                "年化收益": f"{r['annual_return']:.2%}",
                "最大回撤": f"{r['drawdown']:.2%}",
                "夏普": f"{r['sharpe']:.2f}" if r['sharpe'] else "N/A",
                "交易次数": r['total_trades'],
                "胜率": f"{r['win_rate']:.1%}",
            })
        st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

        # ---- 推荐 ----
        best = ranking[0]
        st.success(f"**推荐策略: {best['name']}** — {best['desc']}")

        # ---- 可视化对比 ----
        st.subheader("可视化对比")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        names = [r["name"] for r in ranking]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

        # 收益对比
        returns = [r["annual_return"] * 100 for r in ranking]
        bars = axes[0].barh(names, returns, color=colors)
        axes[0].axvline(x=0, color="gray", linewidth=0.5)
        axes[0].set_title("年化收益率 (%)")
        for bar, val in zip(bars, returns):
            axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8)

        # 回撤对比
        drawdowns = [abs(r["drawdown"]) * 100 for r in ranking]
        bars = axes[1].barh(names, drawdowns, color=colors)
        axes[1].set_title("最大回撤 (%)")
        for bar, val in zip(bars, drawdowns):
            axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8)

        # 散点图：收益 vs 回撤
        axes[2].scatter(
            [abs(r["drawdown"]) * 100 for r in ranking],
            [r["annual_return"] * 100 for r in ranking],
            c=range(len(ranking)), cmap="Set2", s=100,
        )
        for i, r in enumerate(ranking):
            axes[2].annotate(r["name"][:4],
                           (abs(r["drawdown"]) * 100, r["annual_return"] * 100),
                           fontsize=8)
        axes[2].set_xlabel("最大回撤 (%)")
        axes[2].set_ylabel("年化收益 (%)")
        axes[2].axhline(y=0, color="gray", linewidth=0.5)
        axes[2].set_title("收益 vs 风险")

        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.info("点击「开始策略大比武」按钮，将对全部 5 个策略进行回测对比。")
        st.caption("首次运行需要拉取数据，约需 30-60 秒。")
