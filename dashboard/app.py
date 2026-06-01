r"""
量化鲲鹏 (QuantP) 监控仪表盘

启动方法:
    cd .
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config.loader import is_crypto_enabled

st.set_page_config(
    page_title="QuantP",
    page_icon="[chart]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 侧边栏导航
# ============================================================
st.sidebar.title("量化鲲鹏")
st.sidebar.caption("QuantP · 个人量化交易系统")

if not is_crypto_enabled():
    st.sidebar.warning("加密市场未启用。若需使用，设置 markets.crypto=true")

page = st.sidebar.radio(
    "导航",
    ["总览", "策略对比", "信号监控", "持仓监控", "盈亏分析"],
    label_visibility="collapsed",
)

st.sidebar.divider()

# 快速操作区
with st.sidebar.expander("快速操作", expanded=True):
    symbol = st.selectbox("标的", ["沪深300", "中证500", "创业板指"])
    cash = st.number_input("初始资金（万）", value=10, step=1) * 10000

    col1, col2 = st.columns(2)
    with col1:
        if st.button("跑回测", use_container_width=True):
            st.session_state["run_backtest"] = True
    with col2:
        if st.button("策略对比", use_container_width=True):
            st.session_state["run_shootout"] = True

st.sidebar.divider()
st.sidebar.caption("参考: BsStrategy + MoneySkills + Sattoro Hub")

# ============================================================
# 页面路由
# ============================================================
if page == "总览":
    from dashboard.pages import overview
    overview.show(symbol, cash)
elif page == "策略对比":
    from dashboard.pages import compare
    compare.show(symbol, cash)
elif page == "信号监控":
    from dashboard.pages import signals
    signals.show()
elif page == "持仓监控":
    from dashboard.pages import positions
    positions.show()
elif page == "盈亏分析":
    from dashboard.pages import pnl
    pnl.show()


# ============================================================
# 底部状态栏
# ============================================================
st.sidebar.divider()
st.sidebar.caption("Phase 0-4 完成 | 2026-05-22")
