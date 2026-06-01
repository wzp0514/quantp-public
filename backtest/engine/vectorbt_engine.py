"""
VectorBT 加速引擎（C10）— 全量回测/大比武的向量化加速

VectorBT 比 Backtrader 快 100-200 倍（向量化回测 vs 事件驱动），
适合全量回测和大比武等批量场景。单策略精确回测仍用 Backtrader。

用法
--------
>>> from backtest.engine.vectorbt_engine import run_vbt_backtest
>>> result = run_vbt_backtest(df, fast=5, slow=20)
>>> print(result["sharpe"])
"""

import logging
import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("vectorbt_engine")


def has_vectorbt() -> bool:
    """检查 VectorBT 是否可用"""
    try:
        import vectorbt as vbt
        return True
    except ImportError:
        return False


def _vbt_ma_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> dict:
    """VectorBT 双均线交叉策略"""
    try:
        import vectorbt as vbt
    except ImportError:
        return {"error": "VectorBT not installed"}

    close = df["close"].values
    fast_ma = pd.Series(close).rolling(fast).mean().values
    slow_ma = pd.Series(close).rolling(slow).mean().values
    entries = fast_ma > slow_ma
    exits = fast_ma < slow_ma

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        exits=exits,
        freq="1D",
    )

    stats = pf.stats()
    return {
        "annual_return": float(stats.get("Annual Return [%]", 0) / 100),
        "total_return": float(stats.get("Total Return [%]", 0) / 100),
        "drawdown": float(stats.get("Max Drawdown [%]", 100) / 100),
        "sharpe": float(stats.get("Sharpe Ratio", 0)),
        "total_trades": int(stats.get("Total Trades", 0)),
        "win_rate": float(stats.get("Win Rate [%]", 0) / 100),
        "final_value": float(pf.value().iloc[-1]),
        "engine": "vectorbt",
    }


def _vbt_generic_signals(
    df: pd.DataFrame,
    entry_signal: np.ndarray,
    exit_signal: np.ndarray,
) -> dict:
    """VectorBT 通用信号回测"""
    try:
        import vectorbt as vbt
    except ImportError:
        return {"error": "VectorBT not installed"}

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entry_signal,
        exits=exit_signal,
        freq="1D",
    )

    stats = pf.stats()
    return {
        "annual_return": float(stats.get("Annual Return [%]", 0) / 100),
        "total_return": float(stats.get("Total Return [%]", 0) / 100),
        "drawdown": float(stats.get("Max Drawdown [%]", 100) / 100),
        "sharpe": float(stats.get("Sharpe Ratio", 0)),
        "total_trades": int(stats.get("Total Trades", 0)),
        "win_rate": float(stats.get("Win Rate [%]", 0) / 100),
        "final_value": float(pf.value().iloc[-1]),
        "engine": "vectorbt",
    }


def run_vbt_fast(df: pd.DataFrame, strategy_type: str = "ma_cross",
                 **kwargs) -> dict:
    """
    C10: VectorBT 快速回测入口。

    当前支持：ma_cross（双均线交叉）
    其他策略类型逐步迁移（见 _strategy_builders）。
    """
    if not has_vectorbt():
        return {"error": "VectorBT 未安装。pip install vectorbt"}

    if strategy_type == "ma_cross":
        return _vbt_ma_cross(df, **kwargs)

    return {"error": f"不支持的类型: {strategy_type}，当前仅支持 ma_cross"}


def estimate_vbt_speedup(n_strategies: int = 11) -> str:
    """估算 VectorBT vs Backtrader 的加速比"""
    bt_estimate = n_strategies * 2  # Backtrader: ~2s/策略
    vbt_estimate = max(n_strategies / 10, 0.1)  # VectorBT: ~0.1s/策略
    speedup = bt_estimate / vbt_estimate
    return (
        f"Backtrader 预计: ~{bt_estimate}s | "
        f"VectorBT 预计: ~{vbt_estimate:.1f}s | "
        f"加速: ~{speedup:.0f}×"
    )


# ============================================================
# 策略注册映射（C10: 从 ALL_STRATEGIES 到 VBT 加速器的路由）
# ============================================================

VBT_STRATEGY_MAP = {
    "双均线交叉": ("ma_cross", {"fast": 5, "slow": 20}),
    # 更多映射在实现 VectorBT 等价逻辑后添加
    # "布林带回归": ("bollinger", {"period": 20, "devfactor": 2.0}),
    # etc.
}


def run_vbt_shootout(df: pd.DataFrame, strategies: list[str] = None) -> list[dict]:
    """
    C10: VectorBT 加速版策略大比武。
    只对有 VBT 映射的策略使用 VectorBT，无映射的跳过。

    返回和 shootout.run_shootout 兼容的 records 列表。
    """
    if not has_vectorbt():
        logger.warning("VectorBT 不可用，回退到 Backtrader")
        return []

    available = [s for s in (strategies or VBT_STRATEGY_MAP.keys())
                 if s in VBT_STRATEGY_MAP]

    results = []
    for name in available:
        vbt_type, kwargs = VBT_STRATEGY_MAP[name]
        r = run_vbt_fast(df, strategy_type=vbt_type, **kwargs)
        if "error" not in r:
            results.append({
                "name": f"{name}(VBT)",
                "type": "趋势跟踪",
                "source": "builtin",
                "engine": "vectorbt",
                **{k: v for k, v in r.items() if k != "engine"},
            })

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("VectorBT acceleration check:")
    if has_vectorbt():
        print("  VectorBT available")
        print(f"  {estimate_vbt_speedup(11)}")
    else:
        print("  VectorBT not installed. pip install vectorbt")
