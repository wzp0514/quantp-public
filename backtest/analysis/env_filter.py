"""
环境过滤器 — L4信号链 + Agent决策层 作为管线前置过滤器。

不直接下单，不参与选股。只管三件事：
  1. 今天适不适合交易？（should_trade）
  2. 什么方向？（direction: long/short）
  3. 用多少仓位？（position_multiplier: 0-1）

用法
--------
>>> from backtest.analysis.env_filter import EnvironmentFilter
>>> ef = EnvironmentFilter(df)
>>> result = ef.assess()  # 不调LLM，L4做定量 + Agent规则引擎
>>> print(result["should_trade"], result["direction"], result["position_multiplier"])
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("env_filter")


class EnvironmentFilter:
    """
    管线前置环境过滤器。

    L4（定量：另类数据+ML因子）→ 方向信号
    Agent（定性：规则引擎或LLM多角色）→ 市场环境判断
    两者综合 → 能否交易 + 方向 + 建议仓位乘数
    """

    def __init__(self, df: pd.DataFrame, symbol: str = "",
                 use_llm: bool = False, factor_signals: Optional[dict] = None,
                 sentiment_text: str = ""):
        self.df = df
        self.symbol = symbol
        self.use_llm = use_llm
        self.factor_signals = factor_signals or {}
        self.sentiment_text = sentiment_text

        # 预计算 L4 信号（不调 RL，只跑另类+ML，秒级）
        self._l4_result: Optional[dict] = None
        self._agent_result: Optional[dict] = None

    def assess(self, use_rl: bool = False) -> dict:
        """
        运行环境评估。

        返回
        -------
        dict:
            should_trade: bool       — 是否建议交易
            direction: str           — long / short / hold
            position_multiplier: float — 建议仓位乘数 (0-1)
            l4_signal: dict          — L4 原始输出
            agent_signal: dict       — Agent 原始输出
            reasoning: str           — 综合理由
            risk_level: str          — low / medium / high
        """
        # 1. L4 定量信号
        self._l4_result = self._run_l4(use_rl=use_rl)
        l4_dir = self._l4_result.get("action", "hold")
        l4_conf = self._l4_result.get("confidence", 0.5)
        l4_signal = self._l4_result.get("signal", 0.5)

        # 2. Agent 定性判断
        self._agent_result = self._run_agent()
        agent_dir = self._agent_result.get("action", "hold")
        agent_conf = self._agent_result.get("confidence", 0.5)

        # 3. 综合决策
        # 方向一致性：L4 和 Agent 同向 → 置信度取均值；反向 → 保守处理
        if l4_dir == agent_dir and l4_dir in ("buy", "sell"):
            direction = "long" if l4_dir == "buy" else "short"
            consensus_conf = (l4_conf + agent_conf) / 2
            position_multiplier = round(consensus_conf * 0.3, 2)  # 最多 30%
            should_trade = consensus_conf > 0.5
        elif l4_dir == "hold" and agent_dir == "hold":
            direction = "hold"
            position_multiplier = 0
            should_trade = False
        else:
            # 方向不一致 → 降低置信度，不交易
            direction = "hold"
            position_multiplier = 0.05  # 极小试探仓位
            should_trade = False

        # 风险等级
        risk_level = "low"
        agent_flags = self._agent_result.get("risk_flags", [])
        if agent_flags:
            risk_level = "high" if len(agent_flags) > 2 else "medium"
        if not should_trade:
            risk_level = "high"

        # 持仓建议（与开仓信号独立）
        # position_action: keep(维持) / reduce(减仓) / close(平仓)
        # 方向反转 + 高风险 → 建议平仓；高风险只→减仓；低风险→维持
        if risk_level == "high" and direction == "hold":
            position_action = "reduce"
            position_note = "环境风险高，不建议新开仓。现有持仓建议减至半仓以下。"
        elif risk_level == "high":
            position_action = "close"
            position_note = "方向信号与高风险并存，建议平仓观望。"
        elif risk_level == "medium":
            position_action = "reduce"
            position_note = "风险中等，建议控制仓位上限。"
        else:
            position_action = "keep"
            position_note = "环境风险低，可维持现有持仓。"

        reasoning = (
            f"L4: {l4_dir} (conf={l4_conf:.2f}, signal={l4_signal:.3f}) | "
            f"Agent: {agent_dir} (conf={agent_conf:.2f}, mode={self._agent_result.get('mode','')}) | "
            f"→ trade={'YES' if should_trade else 'NO'} dir={direction} pos={position_multiplier:.0%} "
            f"| position: {position_action}"
        )

        result = {
            # 开仓信号
            "should_trade": should_trade,
            "direction": direction,
            "position_multiplier": position_multiplier,
            # 持仓建议（独立维度）
            "position_action": position_action,
            "position_note": position_note,
            # 子组件输出
            "l4_signal": self._l4_result,
            "agent_signal": self._agent_result,
            # 元数据
            "reasoning": reasoning,
            "risk_level": risk_level,
            "risk_flags": agent_flags,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(reasoning)
        return result

    def _run_l4(self, use_rl: bool = False) -> dict:
        """跑 L4 信号链（默认不训练RL）"""
        try:
            from backtest.analysis.l4_integration import L4SignalChain
            l4 = L4SignalChain(self.df, use_rl=use_rl, symbol=self.symbol)
            return l4.run()
        except Exception as e:
            logger.warning(f"L4运行失败: {e}，返回中性信号")
            return {"action": "hold", "confidence": 0.5, "signal": 0.5,
                    "components": {"alt": 0.5, "rl": 0.5, "ml": 0.5}}

    def _run_agent(self) -> dict:
        """跑 Agent 决策层"""
        try:
            from backtest.analysis.agent_decision import AgentDecision
            ad = AgentDecision(self.df, factor_signals=self.factor_signals,
                               sentiment_text=self.sentiment_text)
            return ad.decide(use_llm=self.use_llm)
        except Exception as e:
            logger.warning(f"Agent运行失败: {e}，返回中性信号")
            return {"action": "hold", "confidence": 0.5, "position": 0,
                    "mode": "error", "reasoning": str(e)[:80]}

    def as_strategy_filter(self, direction: str) -> tuple[bool, float]:
        """
        给具体策略的方向过滤器。

        返回 (是否允许该方向, 仓位乘数)
        """
        if self._l4_result is None:
            self.assess()

        l4_dir = self._l4_result.get("action", "hold") if self._l4_result else "hold"
        agent_dir = self._agent_result.get("action", "hold") if self._agent_result else "hold"

        # 策略方向必须与 L4 信号方向一致
        l4_ok = (direction == "long" and l4_dir == "buy") or \
                (direction == "short" and l4_dir == "sell") or \
                (l4_dir == "hold")  # hold时不限制，让回测结果说话

        # Agent 不反对
        agent_risk = self._agent_result.get("risk_flags", []) if self._agent_result else []
        agent_ok = len(agent_risk) < 3

        allowed = l4_ok and agent_ok
        multiplier = 1.0 if allowed else 0.3

        return allowed, multiplier

    @staticmethod
    def check_direction_change(prev_direction: str, current_direction: str) -> bool:
        """
        检测方向是否反转。

        long→short 或 short→long 返回 True。
        hold→X 不算反转（之前没方向）。
        """
        opposites = {("long", "short"), ("short", "long")}
        return (prev_direction, current_direction) in opposites
