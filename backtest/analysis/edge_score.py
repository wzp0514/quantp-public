"""
Edge Score — 统一门控得分，融合区制+环境+信号共振为单一边缘得分。

用法
--------
>>> from backtest.analysis.edge_score import EdgeScore
>>> es = EdgeScore(df)
>>> result = es.compute()
>>> print(result["edge"], result["should_trade"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("edge_score")


class EdgeScore:
    """
    融合三层过滤为单一边缘得分 0-1。

    三层：
      1. 区制置信度（regime_filter） — 当前市场状态是否适合交易
      2. 环境仓位乘数（env_filter） — L4+Agent 综合评估
      3. 信号共振度 — 多指标方向一致性

    默认阈值 0.6，超过才建议交易。
    """

    def __init__(self, df: pd.DataFrame, symbol: str = "",
                 threshold: float = 0.6,
                 regime_weight: float = 0.35,
                 env_weight: float = 0.35,
                 signal_weight: float = 0.30):
        self.df = df
        self.symbol = symbol
        self.threshold = threshold
        self.regime_weight = regime_weight
        self.env_weight = env_weight
        self.signal_weight = signal_weight

    def compute(self, use_llm: bool = False) -> dict:
        """计算边缘得分。返回 edge, should_trade, breakdown, recommendation。"""
        regime_conf = self._regime_confidence()
        env_mult = self._env_multiplier(use_llm=use_llm)
        signal_agree = self._signal_agreement()

        edge = round(
            self.regime_weight * regime_conf +
            self.env_weight * env_mult +
            self.signal_weight * signal_agree, 4
        )

        return {
            "edge": edge,
            "should_trade": edge >= self.threshold,
            "threshold": self.threshold,
            "breakdown": {
                "regime_confidence": round(regime_conf, 4),
                "env_multiplier": round(env_mult, 4),
                "signal_agreement": round(signal_agree, 4),
            },
            "recommendation": self._recommend(edge),
        }

    # -------------------------------------------------------------------
    # 子组件
    # -------------------------------------------------------------------

    def _regime_confidence(self) -> float:
        """区制置信度：当前 regime 的 next-state 概率作为置信度。"""
        try:
            from backtest.analysis.regime_filter import get_regime_signal
            close = self.df.set_index("date")["close"] if "date" in self.df.columns else self.df["close"]
            sig = get_regime_signal(close)
            regime = sig["current_regime"]
            if regime == "Bull":
                conf = sig["bull_prob"]
            elif regime == "Bear":
                conf = sig["bear_prob"]
            else:
                conf = 0.5 + abs(sig["signal"]) * 0.5
            return float(np.clip(conf, 0, 1))
        except Exception as e:
            logger.warning(f"区制置信度计算失败: {e}，返回中性值 0.5")
            return 0.5

    def _env_multiplier(self, use_llm: bool = False) -> float:
        """环境仓位乘数：复用 env_filter 的 L4+Agent 评估。"""
        try:
            from backtest.analysis.env_filter import EnvironmentFilter
            ef = EnvironmentFilter(self.df, symbol=self.symbol, use_llm=use_llm)
            result = ef.assess()
            return result.get("position_multiplier", 0.5)
        except Exception as e:
            logger.warning(f"环境评估失败: {e}，返回中性值 0.5")
            return 0.5

    def _signal_agreement(self) -> float:
        """信号共振度：趋势/动量/量能/RSI 四个独立指标的方向一致性。"""
        try:
            close = self.df["close"].values
            n = len(close)
            if n < 30:
                return 0.5

            signals = []

            # 1. 趋势：短期均线 > 长期均线
            ma_short = np.mean(close[-10:])
            ma_long = np.mean(close[-30:])
            signals.append(1 if ma_short > ma_long else 0)

            # 2. 动量：5日收益为正
            ret_5d = (close[-1] / close[-min(6, n)] - 1)
            signals.append(1 if ret_5d > 0 else 0)

            # 3. 量能：近期放量
            if "volume" in self.df.columns:
                vol = self.df["volume"].values
                vol_recent = np.mean(vol[-5:])
                vol_prev = np.mean(vol[-20:-5]) if n >= 20 else vol_recent
                signals.append(1 if vol_recent > vol_prev else 0)

            # 4. RSI 不极端（30-70 之间 = 信号有效）
            if n >= 14:
                delta = np.diff(close[-15:])
                gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
                loss = abs(np.mean(delta[delta < 0])) if any(delta < 0) else 1e-9
                rs = gain / loss if loss > 0 else 1
                rsi = 100 - 100 / (1 + rs)
                signals.append(1 if 30 < rsi < 70 else 0)

            if not signals:
                return 0.5

            return float(np.clip(sum(signals) / len(signals), 0, 1))
        except Exception as e:
            logger.warning(f"信号共振度计算失败: {e}，返回中性值 0.5")
            return 0.5

    def _recommend(self, edge: float) -> str:
        if edge >= 0.8:
            return "强边缘 — 建议正常仓位交易"
        elif edge >= self.threshold:
            return "中等边缘 — 建议半仓交易"
        elif edge >= 0.4:
            return "弱边缘 — 建议观望或极小仓位试探"
        else:
            return "无边缘 — 不建议交易"


# -------------------------------------------------------------------
# CLI 测试
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.fetchers.fallback import fetch_index_daily_safe

    print("=" * 60)
    print("Edge Score 测试")
    print("=" * 60)

    df = fetch_index_daily_safe("沪深300", "20240101", "20250601")
    es = EdgeScore(df)
    result = es.compute()

    print(f"\n边缘得分: {result['edge']:.4f}")
    print(f"是否交易: {result['should_trade']}")
    print(f"阈值: {result['threshold']}")
    print(f"\n分解:")
    for k, v in result["breakdown"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\n建议: {result['recommendation']}")
