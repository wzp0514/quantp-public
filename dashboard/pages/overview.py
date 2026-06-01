"""
总览页面 — 参考 MoneySkills 市场数据监控 + 风险可视化

显示：市场快照、策略运行状态、风控仪表盘
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime


def show(symbol: str, cash: float):
    st.title(f"总览")
    st.caption(f"更新于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ============================================================
    # 第一行：关键指标卡片
    # ============================================================
    col1, col2, col3, col4, col5 = st.columns(5)

    # 尝试拉取实时数据（自动降级: AkShare → Tushare → Baostock）
    try:
        from data.fetchers.fallback import fetch_index_daily_safe
        df = fetch_index_daily_safe(symbol, "20240101")
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        daily_change = (latest["close"] / prev["close"] - 1) * 100
        recent_high = df["close"].tail(20).max()
        recent_low = df["close"].tail(20).min()
        ret_1m = (latest["close"] / df["close"].iloc[-21] - 1) * 100 if len(df) >= 21 else 0
        ret_3m = (latest["close"] / df["close"].iloc[-63] - 1) * 100 if len(df) >= 63 else 0

        data_ok = True
    except Exception:
        data_ok = False

    if data_ok:
        change_color = "green" if daily_change >= 0 else "red"
        with col1:
            st.metric(
                f"{symbol} 最新价",
                f"{latest['close']:.2f}",
                f"{daily_change:+.2f}%",
            )
        with col2:
            st.metric("20日最高", f"{recent_high:.2f}")
        with col3:
            st.metric("20日最低", f"{recent_low:.2f}")
        with col4:
            st.metric("近1月涨跌", f"{ret_1m:+.2f}%")
        with col5:
            st.metric("近3月涨跌", f"{ret_3m:+.2f}%")

        # ============================================================
        # 第二行：走势图 + 风险仪表盘
        # ============================================================
        left, right = st.columns([2, 1])

        with left:
            st.subheader("近期走势")
            chart_df = df.tail(120).set_index("date")
            st.line_chart(chart_df["close"], height=300)

        with right:
            st.subheader("风控仪表盘")
            try:
                from live.risk.risk_engine import get_params
                params = get_params()

                st.metric("仓位上限", f"{params['max_position_pct']*100:.0f}%")
                st.metric("单笔止损", f"{params['max_single_loss_pct']*100:.0f}%")
                st.metric("日亏损熔断", f"{params['max_daily_loss_pct']*100:.0f}%")
                st.metric("总回撤报废", f"{params['max_drawdown_pct']*100:.0f}%")

                risk_level = "保守" if params['max_single_loss_pct'] <= 0.01 else "适中"
                st.info(f"当前风控等级: **{risk_level}**")
            except Exception:
                st.warning("风控参数未加载")

        # ============================================================
        # 第三行：最近信号（简化）
        # ============================================================
        st.divider()
        st.subheader("操作指引")
        st.markdown(f"""
        - 选中「策略对比」标签页 → 5 个策略在 **{symbol}** 上跑一遍，看排名
        - 选中「持仓监控」标签页 → 查看当前持仓和成本
        - 选中「盈亏分析」标签页 → 查看策略盈亏归因
        - 侧边栏点击「策略对比」按钮 → 立即开始策略大比武
        """)

    else:
        st.warning("无法获取实时数据，请检查网络连接。")
        st.info("侧边栏选择标签页可查看其他功能。")
