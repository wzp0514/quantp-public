"""
Freqtrade→Backtrader 指标改写 — 5个代表性指标。

从 Freqtrade 源码提取指标公式，改写为 MyTT/NumPy 实现，可接入策略体系。

用法
--------
>>> from backtest.freqtrade_indicators import RMI, VIDYA, STC
>>> df["rmi"] = RMI(df["close"].values, 14, 5)
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
# 1. RMI — Relative Momentum Index
# 类似 RSI，但用动量（close - close_{N}）代替价格变化
# ═══════════════════════════════════════════════════════════

def RMI(close: np.ndarray, period: int = 14, momentum: int = 5) -> np.ndarray:
    """
    Relative Momentum Index。

    RMI = 100 - 100/(1 + U/D)
    其中 U/D 是 N 期内动量变化的平均涨/平均跌。
    """
    mom = np.zeros_like(close)
    mom[momentum:] = close[momentum:] - close[:-momentum]
    up = np.where(mom > 0, mom, 0)
    down = np.where(mom < 0, -mom, 0)
    avg_up = pd.Series(up).rolling(period).mean().values
    avg_down = pd.Series(down).rolling(period).mean().values
    rs = np.divide(avg_up, avg_down, out=np.zeros_like(avg_up), where=avg_down != 0)
    return 100 - 100 / (1 + rs)


# ═══════════════════════════════════════════════════════════
# 2. VIDYA — Variable Index Dynamic Average
# 自适应均线，用 Chande Momentum Oscillator 动态调节平滑因子
# ═══════════════════════════════════════════════════════════

def VIDYA(close: np.ndarray, period: int = 14, smoothing: int = 9) -> np.ndarray:
    """
    Variable Index Dynamic Average (Chande 版)。

    CMO = 100 * (U-D)/(U+D)  → alpha = abs(CMO/100) / smoothing
    VIDYA_{t} = alpha * close + (1-alpha) * VIDYA_{t-1}
    """
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0)
    down = np.where(diff < 0, -diff, 0)
    sum_up = pd.Series(up).rolling(period).sum().values
    sum_down = pd.Series(down).rolling(period).sum().values
    total = sum_up + sum_down
    cmo = np.divide(100 * (sum_up - sum_down), total, out=np.zeros_like(sum_up), where=total != 0)

    alpha = np.abs(cmo) / 100 / smoothing
    vidya = np.full_like(close, np.nan)
    vidya[period] = close[period]
    for i in range(period + 1, len(close)):
        vidya[i] = alpha[i] * close[i] + (1 - alpha[i]) * vidya[i - 1]
    return vidya


# ═══════════════════════════════════════════════════════════
# 3. STC — Schaff Trend Cycle
# MACD 再应用 Stochastic 公式，更快的趋势转折信号
# ═══════════════════════════════════════════════════════════

def STC(close: np.ndarray, fast: int = 23, slow: int = 50, cycle: int = 10,
        smooth1: int = 3, smooth2: int = 3) -> np.ndarray:
    """
    Schaff Trend Cycle (Doug Schaff, 1999)。

    1. MACD = EMA(fast) - EMA(slow)
    2. 将 MACD 映射到 0-100 范围 (Stochastic formula applied to MACD)
    3. 双重 EMA 平滑
    """
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    macd = ema_fast - ema_slow

    # Stochastic: %K = (MACD - LL) / (HH - LL) * 100, over 'cycle' periods
    stc = np.full_like(close, np.nan)
    for i in range(slow + cycle, len(close)):
        window = macd[i - cycle:i + 1]
        ll = np.min(window)
        hh = np.max(window)
        denom = hh - ll
        if denom != 0:
            stc[i] = (macd[i] - ll) / denom * 100
        else:
            stc[i] = stc[i - 1] if i > 0 else 50

    # 双重平滑（用 EMA 替换原文的 SMA）
    stc = pd.Series(stc).ewm(span=smooth1, adjust=False).mean().values
    stc = pd.Series(stc).ewm(span=smooth2, adjust=False).mean().values
    return stc


# ═══════════════════════════════════════════════════════════
# 4. TKE — Triple Keltner (Keltner Channel 三线)
# ═══════════════════════════════════════════════════════════

def TKE(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 20, atr_period: int = 10, multiplier: float = 1.5) -> dict:
    """
    Triple Keltner Channel。

    中线 = EMA(typical_price, period)
    typical_price = (high + low + close) / 3
    ATR = Average True Range
    通道 = 中线 ± ATR * multiplier

    返回 dict: {"mid": ..., "upper": ..., "lower": ..., "bandwidth_pct": ...}
    """
    tp = (high + low + close) / 3
    mid = pd.Series(tp).ewm(span=period, adjust=False).mean().values

    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values

    upper = mid + atr * multiplier
    lower = mid - atr * multiplier
    bandwidth = (upper - lower) / mid * 100

    return {"mid": mid, "upper": upper, "lower": lower, "bandwidth_pct": bandwidth}


# ═══════════════════════════════════════════════════════════
# 5. MADR — Moving Average Distance Ratio
# 价格偏离多均线的程度，用于判断超买超卖
# ═══════════════════════════════════════════════════════════

def MADR(close: np.ndarray, periods: tuple = (5, 10, 20, 60, 120)) -> dict:
    """
    Moving Average Distance Ratio。

    计算价格相对于多根均线的偏离百分比，输出综合偏离度。

    返回 dict: {"dist_{p}": ..., "composite": ...}
        composite > 0 → 价格在均线上方（偏多）
        composite < 0 → 价格在均线下方（偏空）
    """
    result = {}
    distances = []
    for p in periods:
        ma = pd.Series(close).rolling(p).mean().values
        dist = (close - ma) / ma * 100
        result[f"dist_{p}"] = dist
        distances.append(dist)

    # 等权合成
    composite = np.nanmean(distances, axis=0)
    result["composite"] = composite
    return result
