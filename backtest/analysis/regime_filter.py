"""
Regime filter — Markov regime confirmation for any strategy.

Usage as pre-trade filter:
    from backtest.analysis.regime_filter import get_regime_signal, should_trade
    sig = get_regime_signal(close_series)
    if should_trade(sig, direction="long", min_confidence=0.1):
        # execute buy

Usage as post-backtest analysis:
    from backtest.analysis.regime_filter import label_trades_with_regime, regime_summary
    labeled = label_trades_with_regime(trades_df, close_series)
    summary = regime_summary(labeled)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from backtest.analysis.markov_regime import analyze, label_regimes, STATES


# --------------------------------------------------------------------------- #
# Pre-trade filter
# --------------------------------------------------------------------------- #

def get_regime_signal(
    close: pd.Series,
    window: int = 20,
    threshold: float = 0.05,
) -> dict:
    """Quick regime check for pre-trade confirmation.

    Returns a dict with current_regime, signal, next-state probabilities,
    and a ready-to-use should_trade bool.
    """
    result = analyze(close, source="pre_trade", window=window, threshold=threshold, hmm=False)
    sig = result["signal"]
    regime = result["current_regime"]

    # Long bias: bullish regime + positive signal
    long_ok = regime == "Bull" or (regime == "Sideways" and sig > 0)
    # Short bias: bearish regime + negative signal
    short_ok = regime == "Bear" or (regime == "Sideways" and sig < 0)

    return {
        "current_regime": regime,
        "signal": sig,
        "bull_prob": result["next_state_probabilities"]["bull"],
        "bear_prob": result["next_state_probabilities"]["bear"],
        "long_ok": long_ok,
        "short_ok": short_ok,
        "persistence": result["persistence_diagonal"],
        "stationary": result["stationary_distribution"],
    }


def should_trade(
    regime_info: dict,
    direction: str = "long",
    allowed_regimes: tuple[str, ...] = ("Bull", "Sideways"),
    min_confidence: float = 0.0,
) -> bool:
    """Simple gate: should we execute a trade given the current regime?

    Parameters
    ----------
    regime_info : dict
        Output from get_regime_signal().
    direction : str
        "long" or "short".
    allowed_regimes : tuple
        Which regimes to allow trading in (default: Bull + Sideways).
    min_confidence : float
        Minimum |signal| to trade (default 0 = no filter).
    """
    if regime_info["current_regime"] not in allowed_regimes:
        return False

    if direction == "long" and not regime_info["long_ok"]:
        return False
    if direction == "short" and not regime_info["short_ok"]:
        return False

    if abs(regime_info["signal"]) < min_confidence:
        return False

    return True


# --------------------------------------------------------------------------- #
# Post-backtest trade labeling
# --------------------------------------------------------------------------- #

def label_trades_with_regime(
    trades_df: pd.DataFrame,
    close: pd.Series,
    window: int = 20,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Add a 'regime' column to each trade based on the market regime at trade date.

    Parameters
    ----------
    trades_df : DataFrame
        Must have 'date' column. From bt_runner result['trades_df'].
    close : pd.Series
        Daily close prices with DatetimeIndex (the same data fed to backtest).
    window, threshold : int, float
        Passed to label_regimes().

    Returns
    -------
    DataFrame with added columns: regime, regime_signal
    """
    if trades_df is None or trades_df.empty:
        return trades_df

    labels = label_regimes(close, window=window, threshold=threshold)
    label_map = {0: "Bear", 1: "Sideways", 2: "Bull"}

    df = trades_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    regimes = []
    signals = []
    for _, row in df.iterrows():
        d = row["date_dt"]
        # Find the closest label on or before this date
        matching = labels[labels.index <= d]
        if len(matching) > 0:
            state = int(matching.iloc[-1])
        else:
            state = 1  # default Sideways
        regimes.append(label_map.get(state, "Sideways"))

        # Compute signal from transition matrix built up to this point
        if len(matching) > 20:
            from backtest.analysis.markov_regime import build_transition_matrix, signal_from_matrix
            P = build_transition_matrix(matching)
            sig = signal_from_matrix(P, state)
        else:
            sig = 0.0
        signals.append(sig)

    df["regime"] = regimes
    df["regime_signal"] = signals
    df.drop(columns=["date_dt"], inplace=True)
    return df


def regime_summary(trades_df: pd.DataFrame) -> dict:
    """Break down trade performance by regime.

    Parameters
    ----------
    trades_df : DataFrame
        Output from label_trades_with_regime() — must have 'regime', 'type', 'value' columns.

    Returns
    -------
    dict with keys: by_regime (per-regime stats), recommendation (str)
    """
    if trades_df is None or trades_df.empty or "regime" not in trades_df.columns:
        return {"by_regime": {}, "recommendation": "no regime data — run label_trades_with_regime() first"}

    df = trades_df.copy()
    # Only look at sell trades for P&L (buy entries don't have P&L directly)
    sells = df[df["type"] == "sell"].copy()

    by_regime = {}
    for regime in ["Bull", "Sideways", "Bear"]:
        subset = sells[sells["regime"] == regime]
        n = len(subset)
        if n == 0:
            by_regime[regime] = {"count": 0, "avg_value": 0, "regime": regime}
        else:
            by_regime[regime] = {
                "count": n,
                "avg_value": float(subset["value"].mean()),
                "regime": regime,
            }

    # Also count buy entries by regime
    buys = df[df["type"] == "buy"]
    for regime in ["Bull", "Sideways", "Bear"]:
        n_buys = len(buys[buys["regime"] == regime])
        if regime in by_regime:
            by_regime[regime]["buy_count"] = n_buys

    # Recommendation
    bull_count = by_regime.get("Bull", {}).get("count", 0)
    bear_count = by_regime.get("Bear", {}).get("count", 0)
    total = max(bull_count + bear_count + by_regime.get("Sideways", {}).get("count", 0), 1)

    if bear_count > total * 0.4:
        recommendation = "策略在Bear区制交易过多——建议叠加区制过滤器，Bear区不交易"
    elif bull_count > total * 0.5:
        recommendation = "策略大部分交易集中在Bull区制——牛市依赖度高，注意熊市表现"
    else:
        recommendation = "区制分布较均衡"

    return {"by_regime": by_regime, "recommendation": recommendation}


# --------------------------------------------------------------------------- #
# CLI test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.fetchers.fallback import fetch_index_daily_safe

    print("=" * 60)
    print("Regime Filter 测试")
    print("=" * 60)

    # 1. Pre-trade signal
    df = fetch_index_daily_safe("沪深300", "20240101", "20250601")
    close = df.set_index("date")["close"]
    sig = get_regime_signal(close)
    print(f"\n当前区制: {sig['current_regime']}")
    print(f"信号: {sig['signal']:+.4f}")
    print(f"Bull概率: {sig['bull_prob']:.1%}  Bear概率: {sig['bear_prob']:.1%}")
    print(f"做多允许: {sig['long_ok']}  做空允许: {sig['short_ok']}")
    print(f"区制粘性: Bear={sig['persistence']['bear']:.1%} "
          f"Sideways={sig['persistence']['sideways']:.1%} "
          f"Bull={sig['persistence']['bull']:.1%}")

    # 2. Trade labeling
    from backtest.engine.bt_runner import run_backtest
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy

    result = run_backtest(MaCrossStrategy, df, fast=5, slow=20)
    if not result["trades_df"].empty:
        labeled = label_trades_with_regime(result["trades_df"], close)
        summary = regime_summary(labeled)
        print(f"\n区制分布: {summary['by_regime']}")
        print(f"建议: {summary['recommendation']}")
