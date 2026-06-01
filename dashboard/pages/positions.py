"""
持仓监控页面 — 参考 BsStrategy 组合监控

显示当前持仓、成本、盈亏、风险敞口。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime


def show():
    st.title("持仓监控")
    st.caption("参考 BsStrategy — 组合监控")

    # 模拟持仓数据（实际接入 position_mgr 后替换）
    st.info("当前为模拟数据。实盘模式下，此页面从 PositionManager 读取实际持仓。")

    # 模拟持仓
    mock_positions = [
        {"标的": "沪深300 ETF", "代码": "510300", "持仓量": 2500, "均价": 3.85, "现价": 3.95, "市值": 9875, "盈亏": 250.0},
        {"标的": "贵州茅台", "代码": "600519", "持仓量": 50, "均价": 1680.0, "现价": 1720.0, "市值": 86000, "盈亏": 2000.0},
    ]

    if mock_positions:
        df = pd.DataFrame(mock_positions)
        # 格式化列
        st.dataframe(
            df.style.format({
                "均价": "{:.2f}",
                "现价": "{:.2f}",
                "市值": "¥{:,.0f}",
                "盈亏": "{:+.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        total_value = sum(p["市值"] for p in mock_positions)
        total_pnl = sum(p["盈亏"] for p in mock_positions)
        cash = 50000.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("持仓市值", f"¥{total_value:,.0f}")
        with col2:
            st.metric("可用现金", f"¥{cash:,.0f}")
        with col3:
            st.metric("浮动盈亏", f"¥{total_pnl:+,.0f}")

        st.metric("总资产", f"¥{total_value + cash:,.0f}")

        # 风险敞口
        exposure = total_value / (total_value + cash) * 100
        st.progress(exposure / 100, text=f"风险敞口: {exposure:.1f}%")
    else:
        st.info("当前无持仓")

    st.divider()
    st.subheader("最近交易")
    st.caption("从 OrderManager 读取最近订单")
    st.text("暂无交易记录（模拟数据）")

    # 后续接入真实数据:
    # from live.execution.position_mgr import position_mgr
    # from live.execution.order_mgr import order_mgr
