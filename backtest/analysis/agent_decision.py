"""
Agent决策层 — 多Agent LLM分析-辩论-决策架构。

参考 TradingAgents 的 LangGraph 架构，实现：
  分析师(技术面/因子/情绪，并行) → 研究员(多空辩论) → 交易员 → 风控官(3角色) → PM审批

LangGraph 可用时自动走状态图编排，不可用时回退到顺序LLM调用。
LLM 完全不可用时回退到规则引擎。

用法
--------
>>> from backtest.analysis.agent_decision import AgentDecision
>>> ad = AgentDecision(df, factor_signals)
>>> result = ad.decide(use_llm=True)
>>> print(result["action"], result["reasoning"])
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("agent_decision")

# AgentState 需在模块级别可访问（LangGraph 用 get_type_hints 解析类型注解）
try:
    from backtest.analysis.agent_schemas import AgentState  # noqa: F401
except ImportError:
    AgentState = None

# LangGraph 是否可用
_HAS_LANGGRAPH = False
try:
    import langgraph  # noqa: F401
    _HAS_LANGGRAPH = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════
# LLM 调用（通过统一 provider 接口）
# ═══════════════════════════════════════════════════════════

def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 500,
              model: str = "", temperature: float = 0.3) -> str:
    """调 LLM，失败返回空字符串。支持指定 model 和 temperature。"""
    try:
        from backtest.analysis.llm_provider import call_llm
        return call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            model=model,
            temperature=temperature,
        )
    except ImportError:
        pass

    # 回退：原始方式
    try:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            client = Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text
    except Exception:
        pass

    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
    except Exception:
        pass

    return ""


def _parse_json(text: str, default: dict = None) -> dict:
    """从 LLM 文本输出中提取 JSON"""
    if default is None:
        default = {}
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 JSON 块
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return default


# ═══════════════════════════════════════════════════════════
# AgentDecision 主类
# ═══════════════════════════════════════════════════════════

class AgentDecision:
    """
    多 Agent 决策框架。

    两种模式:
      - rule_based: 规则引擎（默认，无依赖）
      - LLM-driven: 走 LangGraph 状态图（langgraph 可用时）或顺序 LLM 调用

    流水线: 分析师(3角色并行) → 研究员辩论 → 交易员下单 → 风控审核 → PM批准
    """

    def __init__(self, df: pd.DataFrame, factor_signals: Optional[dict] = None,
                 sentiment_text: str = ""):
        self.df = df.copy()
        self.factor_signals = factor_signals or {}
        self.sentiment_text = sentiment_text

        latest = df.iloc[-1] if len(df) > 0 else {}
        self.snapshot = {
            "date": str(latest.get("date", "")),
            "price": float(latest.get("close", 0)),
            "change_5d": 0.0,
            "change_20d": 0.0,
            "volatility_20d": 0.0,
            "volume_ratio": 1.0,
        }
        if len(df) >= 5:
            self.snapshot["change_5d"] = round(
                float(df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100, 2)
        if len(df) >= 20:
            self.snapshot["change_20d"] = round(
                float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100, 2)
            returns = df["close"].pct_change().tail(20)
            self.snapshot["volatility_20d"] = round(
                float(returns.std() * np.sqrt(252) * 100), 1)
            avg_vol = float(df["volume"].tail(20).mean())
            if avg_vol > 0:
                self.snapshot["volume_ratio"] = round(
                    float(df["volume"].iloc[-1]) / avg_vol, 2)

    def _build_context(self) -> str:
        """构建市场数据上下文（给 LLM 的输入）"""
        ctx = json.dumps(self.snapshot, ensure_ascii=False, indent=2)
        if self.factor_signals:
            ctx += f"\n因子信号: {json.dumps(self.factor_signals, ensure_ascii=False)}"
        if self.sentiment_text:
            ctx += f"\n外部情绪: {self.sentiment_text}"
        return ctx

    def _build_tech_view(self) -> str:
        """技术分析师专属视角：仅K线结构+技术指标，不给因子/情绪数据"""
        s = self.snapshot
        # 计算更多技术指标
        close = self.df["close"]
        volume = self.df["volume"]
        high = self.df.get("high", close)
        low = self.df.get("low", close)

        # RSI 14日
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = float((ema12 - ema26).iloc[-1])
        signal = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

        # 布林带位置
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_position = float((close.iloc[-1] - ma20.iloc[-1]) / (2 * std20.iloc[-1] + 1e-10))

        # 最近5日K线形态
        last5 = self.df.tail(5)
        candles = []
        for _, row in last5.iterrows():
            body = abs(float(row["close"] - row["open"]))
            upper = float(row["high"] - max(row["close"], row["open"]))
            lower = float(min(row["close"], row["open"]) - row["low"])
            direction = "阳" if row["close"] >= row["open"] else "阴"
            candles.append(f"  {direction}线 实体{body:.2f} 上影{upper:.2f} 下影{lower:.2f}")

        # 均线位置
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20_val = float(ma20.iloc[-1])
        price = s["price"]

        lines = [
            "【技术分析专属数据 — 仅K线和指标】",
            f"价格: {price:.2f} | MA5: {ma5:.2f} | MA20: {ma20_val:.2f}",
            f"RSI(14): {rsi:.1f} | MACD: {macd:.4f} (signal: {signal:.4f})",
            f"布林带位置: {bb_position:.2f}σ (>1超买 <-1超卖)",
            f"5日涨跌: {s['change_5d']:+.1f}% | 20日涨跌: {s['change_20d']:+.1f}%",
            f"20日波动率: {s['volatility_20d']:.1f}% | 量比: {s['volume_ratio']:.2f}",
            "最近5日K线形态:",
        ] + candles + [
            "",
            "请基于上述技术指标分析，判断短期方向。注意：你没有因子数据也没有新闻数据，只能从价格和指标中判断。",
        ]
        return "\n".join(lines)

    def _build_factor_view(self) -> str:
        """因子分析师专属视角：仅因子IC/IR排名，不给K线/情绪数据"""
        fs = self.factor_signals

        # 构建 IC 排名表
        top_factors = fs.get("top_factors", [])
        ic_table = []
        if top_factors:
            for i, f in enumerate(top_factors[:10]):
                name = f.get("factor", f.get("name", f"因子{i}"))
                ic = f.get("ic", 0)
                rank_ic = f.get("rank_ic", 0)
                ir = f.get("ic_ir", 0)
                ic_table.append(f"  {i+1}. {name}: IC={ic:+.4f} RankIC={rank_ic:+.4f} IR={ir:+.3f}")
        else:
            ic_table.append("  (无详细因子数据)")

        lines = [
            "【因子分析专属数据 — 仅因子量化指标】",
            f"强因子数: {fs.get('strong_count', 0)} | 有效因子数(ICIR): {fs.get('icir_valid_count', 0)}",
            "因子IC排名 (前10):",
        ] + ic_table + [
            "",
            "请基于上述因子数据，判断因子面对市场方向的指示。",
            f"判断规则参考: |IC|>0.05视为有效预测，IC为正→看多，为负→看空。",
            "注意：你没有K线数据也没有新闻数据，只能从因子IC中判断。",
        ]
        return "\n".join(lines)

    def _build_sentiment_view(self) -> str:
        """情绪分析师专属视角：仅情绪/新闻文本+涨跌方向，不给因子/技术指标"""
        s = self.snapshot
        sentiment = self.sentiment_text if self.sentiment_text else "（无外部情绪数据，请基于涨跌方向给出市场情绪判断）"

        lines = [
            "【情绪分析专属数据 — 仅新闻和市场情绪】",
            f"5日涨跌: {s['change_5d']:+.1f}%",
            f"20日涨跌: {s['change_20d']:+.1f}%",
            f"量比: {s['volume_ratio']:.2f} (>1.5放量 <0.5缩量)",
            f"外部情绪/新闻:",
            f"  {sentiment}",
            "",
            "请基于上述情绪数据，判断市场情绪方向。",
            "注意：你没有K线技术指标也没有因子数据，只能从情绪/新闻文本和涨跌方向中判断。",
        ]
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 规则引擎回退
    # ═══════════════════════════════════════════════════════════

    def _rule_based_decision(self) -> dict:
        """基于规则的决策引擎，不依赖 LLM"""
        s = self.snapshot
        reasons = []

        tech_score = 50
        if s["change_5d"] > 2:
            tech_score += 15
            reasons.append(f"5日涨幅{s['change_5d']}%→动量偏多")
        elif s["change_5d"] < -3:
            tech_score -= 15
            reasons.append(f"5日跌幅{abs(s['change_5d'])}%→超卖可能反弹")
        if s["volume_ratio"] > 1.5:
            tech_score += 5
            reasons.append("放量→资金关注")

        strong_count = self.factor_signals.get("strong_count", 0)
        if strong_count >= 5:
            tech_score += 10
            reasons.append(f"{strong_count}个强因子→因子面偏多")
        elif strong_count < 2:
            tech_score -= 5

        risk_flags = []
        if s["volatility_20d"] > 40:
            risk_flags.append(f"高波动({s['volatility_20d']}%)")
        if abs(s["change_20d"]) > 20:
            risk_flags.append(f"20日涨跌幅异常({s['change_20d']}%)")

        tech_score = max(0, min(100, tech_score))
        if tech_score >= 60:
            action, confidence = "buy", round(tech_score / 100, 2)
        elif tech_score <= 40:
            action, confidence = "sell", round((100 - tech_score) / 100, 2)
        else:
            action, confidence = "hold", round(1 - abs(tech_score - 50) / 50, 2)

        position = round(confidence * 0.3, 2) if action != "hold" else 0

        return {
            "action": action,
            "confidence": confidence,
            "position": position,
            "reasoning": "; ".join(reasons),
            "risk_flags": risk_flags,
            "tech_score": tech_score,
            "mode": "rule_based",
            "agents": {
                "tech_analyst": {"score": tech_score, "bias": "bullish" if tech_score > 50 else "bearish"},
                "factor_analyst": {"strong_count": strong_count},
                "sentiment_analyst": {"available": bool(self.sentiment_text)},
                "risk_officer": {"flags": risk_flags, "approved": len(risk_flags) < 3},
            },
        }

    # ═══════════════════════════════════════════════════════════
    # LangGraph 模式
    # ═══════════════════════════════════════════════════════════

    # 角色配置表：角色 → {view, model, temperature, expertise, personality}
    # 方案1(temperature)+方案2(数据视角)+方案3(风控深度prompt)+方案4(人格化)
    _ROLE_CONFIG = {
        "tech_analyst": {
            "view": "tech_view",
            "model": "deepseek-v4-flash",
            "temperature": 0.7,
            "expertise": "专精K线形态、均线、MACD、RSI等技术指标。",
            "personality": "你每天看200张K线图，练就了从图表中嗅到机会的本能。你相信价格包容一切。",
        },
        "factor_analyst": {
            "view": "factor_view",
            "model": "deepseek-v4-flash",
            "temperature": 0.5,
            "expertise": "专精量化因子评估（动量、波动、价值、质量等），基于IC/IR判断。",
            "personality": "你是量化出身，只信数据不信故事。IC好就做，IC差就撤，不纠结。",
        },
        "sentiment_analyst": {
            "view": "sentiment_view",
            "model": "deepseek-v4-flash",
            "temperature": 0.6,
            "expertise": "专精市场情绪分析（新闻、社交媒体、资金流向）。",
            "personality": "你曾经因为忽视市场情绪错过了一次大跌，此后再也不敢小看新闻的力量。",
        },
        # 风控三角色（节点内用）
        "risk_aggressive": {
            "personality": "你相信风险是收益的代价，回撤10%很正常。只要逻辑成立就值得试。",
        },
        "risk_conservative": {
            "personality": "你在2008年入行，亲眼见过客户爆仓跳楼。你坚信活下来才有机会，宁可错过不能做错。",
        },
        "risk_neutral": {
            "personality": "你是量化风控出身，只看数字说话。夏普比率、最大回撤、Calmar比率——数字不撒谎。",
        },
    }

    def _node_analyst(self, node_name: str):
        """生成分析师节点函数（数据视角+模型+temperature+人格）"""
        cfg = self._ROLE_CONFIG[node_name]
        role_map = {
            "tech_analyst": "技术分析师",
            "factor_analyst": "因子分析师",
            "sentiment_analyst": "情绪分析师",
        }
        role = role_map[node_name]
        view_key = cfg["view"]
        model = cfg["model"]
        temperature = cfg["temperature"]

        def node_fn(state: AgentState) -> AgentState:
            view = state.get(view_key, json.dumps(state.get("snapshot", {}), ensure_ascii=False))
            system = (
                f"你是{role}。{cfg['personality']}\n{cfg['expertise']}\n"
                f"你只能根据提供给你的专属数据做判断，不要臆测你没有的数据。"
            )
            user = (
                f"{view}\n\n"
                f"请用JSON回复: "
                f'{{"bias": "bullish/bearish/neutral", "score": 0-100, '
                f'"key_points": ["你的判断依据1","你的判断依据2"], "confidence": 0.0-1.0}}'
            )
            resp = _call_llm(system, user, max_tokens=300, model=model, temperature=temperature)
            parsed = _parse_json(resp, {"bias": "neutral", "score": 50, "key_points": [], "confidence": 0.5})
            parsed["role"] = role
            state["analyst_reports"] = [parsed]
            logger.debug(f"{role}: bias={parsed.get('bias')}, score={parsed.get('score')} (model={model}, T={temperature})")
            return state
        return node_fn

    def _build_graph(self):
        """构建 LangGraph 状态图"""
        from langgraph.graph import StateGraph, END
        from backtest.analysis.agent_schemas import AgentState

        builder = StateGraph(AgentState)

        # 分析师节点（不同数据视角 + 不同模型 + 不同temperature + 人格）
        builder.add_node("tech_analyst", self._node_analyst("tech_analyst"))
        builder.add_node("factor_analyst", self._node_analyst("factor_analyst"))
        builder.add_node("sentiment_analyst", self._node_analyst("sentiment_analyst"))

        builder.add_node("debate", self._node_debate)
        builder.add_node("trader", self._node_trader)
        builder.add_node("risk_mgmt", self._node_risk_mgmt)
        builder.add_node("portfolio_mgr", self._node_portfolio_mgr)

        # 边（分析师顺序执行，确保各自输出后进入辩论）
        builder.set_entry_point("tech_analyst")
        builder.add_edge("tech_analyst", "factor_analyst")
        builder.add_edge("factor_analyst", "sentiment_analyst")
        builder.add_edge("sentiment_analyst", "debate")
        builder.add_edge("debate", "trader")
        builder.add_edge("trader", "risk_mgmt")
        builder.add_conditional_edges("risk_mgmt", self._after_risk, {
            "approved": "portfolio_mgr",
            "rejected": END,
        })
        builder.add_edge("portfolio_mgr", END)

        return builder.compile()

    def _node_debate(self, state: AgentState) -> AgentState:
        """研究员多空辩论（pro模型 + 深度思考 + 人格）"""
        reports = state.get("analyst_reports", [])
        analyses_text = "\n".join(
            f"- {r.get('role','?')}: bias={r.get('bias','?')}, score={r.get('score','?')}"
            for r in reports
        )
        system = (
            "你是资深量化研究员，有20年买方研究经验，经历过三轮牛熊。"
            "看多和看空都能说得头头是道，你的价值在于不被单一视角绑架。"
            "综合三位分析师的报告，自己做多空辩论后做出判断。"
        )
        user = (
            f"分析师报告:\n{analyses_text}\n\n"
            f"请列多方论据(bull_args)、空方论据(bear_args)，然后综合判断(synthesis)。"
            f"返回JSON: "
            f'{{"decision": "buy/sell/hold", "confidence": 0.0-1.0, '
            f'"bull_args": ["多方论据1"], "bear_args": ["空方论据1"], '
            f'"synthesis": "综合判断(50字)"}}'
        )
        resp = _call_llm(system, user, max_tokens=500, model="deepseek-v4-pro", temperature=0.6)
        parsed = _parse_json(resp, {"decision": "hold", "confidence": 0.5,
                                    "bull_args": [], "bear_args": [], "synthesis": ""})
        state["debate_result"] = parsed
        logger.debug(f"辩论结果: {parsed.get('decision')}, conf={parsed.get('confidence')}")
        return state

    def _node_trader(self, state: AgentState) -> AgentState:
        """交易员下单（flash模型 + 低temperature + 人格）"""
        debate = state.get("debate_result", {})
        system = (
            "你是执行力极强的交易员。见过太多人在犹豫中错过机会，"
            "一旦决策就果断执行。但你从不追高，不为情绪买单。"
        )
        user = (
            f"研究结论: {debate.get('decision','hold')}, "
            f"置信度: {debate.get('confidence',0.5)}, "
            f"理由: {debate.get('synthesis','')}\n\n"
            f"请给出交易计划JSON: "
            f'{{"action": "buy/sell/hold", "position_pct": 0.0-0.3, '
            f'"limit_price": null, "reason": "下单理由(30字)"}}'
        )
        resp = _call_llm(system, user, max_tokens=250, model="deepseek-v4-flash", temperature=0.3)
        parsed = _parse_json(resp, {"action": "hold", "position_pct": 0,
                                    "limit_price": None, "reason": ""})
        state["trade_plan"] = parsed
        logger.debug(f"交易计划: {parsed.get('action')}, pos={parsed.get('position_pct')}")
        return state

    def _node_risk_mgmt(self, state: AgentState) -> AgentState:
        """风控审核 — 三角色不同人格+不同temperature+深度风控规则清单"""
        trade = state.get("trade_plan", {})

        risk_roles = [
            {
                "name": "激进风控官",
                "personality": self._ROLE_CONFIG["risk_aggressive"]["personality"],
                "temperature": 0.5,
                "rules": (
                    "检查以下硬性规则（违反任一条则拒绝）：\n"
                    "1. 仓位不超过总资金 20%\n"
                    "2. 单笔亏损不超过 2%\n"
                    "其他情况可弹性处理。"
                ),
            },
            {
                "name": "保守风控官",
                "personality": self._ROLE_CONFIG["risk_conservative"]["personality"],
                "temperature": 0.1,
                "rules": (
                    "逐条检查以下风控规则，违反任一条必须拒绝：\n"
                    "1. 单仓位 ≤ 总资金 20%\n"
                    "2. 单笔亏损 ≤ 2%\n"
                    "3. 连续亏损 5 笔必须暂停\n"
                    "4. 波动率超 40% → 仓位减半\n"
                    "5. 不追涨停、不接飞刀\n"
                    "6. 20日跌幅≥15% 禁止抄底\n"
                    "7. 量比>3 且无明确利好 → 疑似出货 → 拒绝\n"
                    "8. 盘中临时加仓 ≤ 原计划 1.5倍\n"
                    "对以上8条逐条回复'符合'或'违反'，任一违反则拒绝。"
                ),
            },
            {
                "name": "中性风控官",
                "personality": self._ROLE_CONFIG["risk_neutral"]["personality"],
                "temperature": 0.3,
                "rules": (
                    "基于量化指标审核：\n"
                    "1. 仓位不超过 20%\n"
                    "2. 单笔亏损不超过 2%\n"
                    "3. 波动率超 40% 时减仓\n"
                    "客观评估，数字说话。"
                ),
            },
        ]

        votes = []
        for rc in risk_roles:
            system = (
                f"你是{rc['name']}。{rc['personality']}\n\n{rc['rules']}"
            )
            user = (
                f"待审核交易: action={trade.get('action')}, "
                f"仓位={trade.get('position_pct',0)}, "
                f"理由={trade.get('reason','')}\n\n"
                f"审核后返回JSON: "
                f'{{"approved": true/false, "adjusted_position": 0.0-0.3, '
                f'"risk_flags": ["具体风险"], "note": "审核意见(30字)"}}'
            )
            resp = _call_llm(system, user, max_tokens=300,
                           model="deepseek-v4-flash", temperature=rc["temperature"])
            parsed = _parse_json(resp, {"approved": True,
                                        "adjusted_position": trade.get("position_pct", 0),
                                        "risk_flags": [], "note": ""})
            votes.append(parsed)

        approved_count = sum(1 for v in votes if v.get("approved", False))
        approved = approved_count >= 2
        approved_positions = [v.get("adjusted_position", 0) for v in votes if v.get("approved", False)]
        avg_position = sum(approved_positions) / len(approved_positions) if approved_positions else 0

        all_flags = []
        for v in votes:
            all_flags.extend(v.get("risk_flags", []))

        risk_review = {
            "approved": approved,
            "adjusted_position": round(avg_position, 4),
            "risk_flags": list(set(all_flags)),
            "risk_level": "high" if not approved else ("medium" if len(all_flags) > 2 else "low"),
            "note": f"风控表决: {approved_count}/3通过",
        }
        state["risk_review"] = risk_review
        logger.debug(f"风控审核: approved={approved}, flags={len(all_flags)}")
        return state

    def _after_risk(self, state: AgentState) -> str:
        review = state.get("risk_review", {})
        if review.get("approved", False):
            return "approved"
        return "rejected"

    def _node_portfolio_mgr(self, state: AgentState) -> AgentState:
        """PM 最终审批"""
        trade = state.get("trade_plan", {})
        review = state.get("risk_review", {})
        debate = state.get("debate_result", {})

        action = "hold"
        position = 0
        if review.get("approved", False) and debate.get("confidence", 0) >= 0.5:
            action = trade.get("action", "hold")
            position = review.get("adjusted_position", trade.get("position_pct", 0))

        state["action"] = action
        state["confidence"] = debate.get("confidence", 0.5)
        state["position"] = position
        state["reasoning"] = debate.get("synthesis", "")
        state["mode"] = "llm_langgraph"
        state["error"] = None
        return state

    def _decide_langgraph(self) -> dict:
        """通过 LangGraph 状态图运行决策"""
        initial_state = {
            "symbol": "",
            "snapshot": self.snapshot,
            "factor_signals": self.factor_signals,
            "sentiment_text": self.sentiment_text,
            "tech_view": self._build_tech_view(),
            "factor_view": self._build_factor_view(),
            "sentiment_view": self._build_sentiment_view(),
            "analyst_reports": [],
            "debate_result": None,
            "trade_plan": None,
            "risk_review": None,
            "action": "hold",
            "confidence": 0.5,
            "position": 0.0,
            "reasoning": "",
            "mode": "llm_langgraph",
            "error": None,
        }

        try:
            graph = self._build_graph()
            final_state = graph.invoke(initial_state)

            action = final_state.get("action", "hold")
            confidence = final_state.get("confidence", 0.5)
            position = final_state.get("position", 0.0)

            # 如果风控拒绝了，用 debate 结果给出保守建议
            if action == "hold" and final_state.get("risk_review"):
                review = final_state["risk_review"]
                if not review.get("approved", False):
                    final_state["reasoning"] = (
                        f"风控拒绝: {review.get('note','')}。"
                        f"风险标记: {', '.join(review.get('risk_flags',[]))}"
                    )

            result = {
                "action": action,
                "confidence": confidence,
                "position": position,
                "reasoning": final_state.get("reasoning", ""),
                "risk_note": final_state.get("risk_review", {}).get("note", ""),
                "risk_flags": final_state.get("risk_review", {}).get("risk_flags", []),
                "mode": "llm_langgraph",
                "timestamp": datetime.now().isoformat(),
                "agents": {
                    "analysts": final_state.get("analyst_reports", []),
                    "debate": final_state.get("debate_result", {}),
                    "trade": final_state.get("trade_plan", {}),
                    "risk_officer": final_state.get("risk_review", {}),
                },
            }
            logger.info(f"Agent决策(LangGraph): {action}, 置信度={confidence:.2f}, 仓位={position:.2%}")
            return result

        except Exception as e:
            logger.warning(f"LangGraph运行失败: {e}，回退到顺序LLM调用")
            return self._decide_sequential()

    # ═══════════════════════════════════════════════════════════
    # 顺序 LLM 模式（LangGraph 不可用时的回退）
    # ═══════════════════════════════════════════════════════════

    def _decide_sequential(self) -> dict:
        """顺序调用 LLM（无 LangGraph 时的回退）。同样使用数据视角+模型+temperature+人格。"""
        cfg = self._ROLE_CONFIG

        # 分析师配置：（角色名, node_name, 视角方法）
        analyst_configs = [
            ("技术分析师", "tech_analyst", self._build_tech_view),
            ("因子分析师", "factor_analyst", self._build_factor_view),
            ("情绪分析师", "sentiment_analyst", self._build_sentiment_view),
        ]

        analyses = []
        for role, node_name, view_fn in analyst_configs:
            c = cfg[node_name]
            view = view_fn()
            system = (
                f"你是{role}。{c['personality']}\n{c['expertise']}\n"
                f"你只能根据提供给你的专属数据做判断，不要臆测你没有的数据。"
            )
            user = (
                f"{view}\n\n"
                f"请用JSON回复: "
                f'{{"bias": "bullish/bearish/neutral", "score": 0-100, '
                f'"key_point": "你的判断依据(30字)"}}'
            )
            resp = _call_llm(system, user, max_tokens=300, model=c["model"], temperature=c["temperature"])
            parsed = _parse_json(resp, {"bias": "neutral", "score": 50, "key_point": ""})
            parsed["role"] = role
            analyses.append(parsed)

        # 辩论（pro + 人格 + T=0.6）
        analyses_text = "\n".join(
            f"- {a.get('role','?')}: bias={a.get('bias','?')}, score={a.get('score','?')}"
            for a in analyses
        )
        debate_resp = _call_llm(
            "你是资深量化研究员，有20年买方研究经验，经历过三轮牛熊。"
            "看多和看空都能说得头头是道，你的价值在于不被单一视角绑架。",
            f"分析师报告:\n{analyses_text}\n\n"
            f"返回JSON: {{\"decision\": \"buy/sell/hold\", \"confidence\": 0.0-1.0, \"debate_summary\": \"...(50字)\"}}",
            max_tokens=500, model="deepseek-v4-pro", temperature=0.6,
        )
        debate = _parse_json(debate_resp, {"decision": "hold", "confidence": 0.5, "debate_summary": ""})

        # 交易员（flash + T=0.3）
        trade_resp = _call_llm(
            "你是执行力极强的交易员。见过太多人在犹豫中错过机会，"
            "一旦决策就果断执行。但你从不追高，不为情绪买单。",
            f"研究结论: {debate.get('decision','hold')}, 置信度: {debate.get('confidence',0.5)}, "
            f"理由: {debate.get('debate_summary','')}\n"
            f"返回JSON: {{\"action\": \"buy/sell/hold\", \"position_pct\": 0.0-0.3, \"reason\": \"...(30字)\"}}",
            max_tokens=250, model="deepseek-v4-flash", temperature=0.3,
        )
        trade = _parse_json(trade_resp, {"action": "hold", "position_pct": 0, "reason": ""})

        # 风控（保守人格 + 深度规则 + T=0.1）
        risk_resp = _call_llm(
            "你是保守风控官。2008年入行，亲眼见过客户爆仓跳楼。"
            "你坚信活下来才有机会，宁可错过不能做错。\n\n"
            "逐条检查以下规则，违反任一条必须拒绝：\n"
            "1. 单仓位 ≤ 总资金 20%\n2. 单笔亏损 ≤ 2%\n3. 连续亏损 5 笔暂停\n"
            "4. 波动率>40%→仓位减半\n5. 不追涨停不接飞刀\n"
            "6. 20日跌幅≥15%禁止抄底\n7. 量比>3且无利好→拒绝\n"
            "8. 盘中加仓≤原计划1.5倍",
            f"交易计划: action={trade.get('action')}, 仓位={trade.get('position_pct',0)}, "
            f"理由={trade.get('reason','')}\n"
            f"返回JSON: {{\"approved\": true/false, \"adjusted_position\": 0.0-0.3, \"risk_note\": \"...(30字)\"}}",
            max_tokens=300, model="deepseek-v4-flash", temperature=0.1,
        )
        risk = _parse_json(risk_resp, {"approved": True, "adjusted_position": trade.get("position_pct", 0), "risk_note": ""})

        action = "hold"
        position = 0
        if risk.get("approved", False) and debate.get("confidence", 0) >= 0.5:
            action = trade.get("action", "hold")
            position = risk.get("adjusted_position", trade.get("position_pct", 0))

        result = {
            "action": action,
            "confidence": debate.get("confidence", 0.5),
            "position": position,
            "reasoning": debate.get("debate_summary", ""),
            "risk_note": risk.get("risk_note", ""),
            "risk_flags": [],
            "mode": "llm_sequential",
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "tech_analyst": analyses[0] if len(analyses) > 0 else {},
                "factor_analyst": analyses[1] if len(analyses) > 1 else {},
                "sentiment_analyst": analyses[2] if len(analyses) > 2 else {},
                "debate": debate,
                "trade": trade,
                "risk_officer": risk,
            },
        }

        logger.info(f"Agent决策(Sequential): {action}, 置信度={result['confidence']:.2f}")
        return result

    # ═══════════════════════════════════════════════════════════
    # 主决策入口
    # ═══════════════════════════════════════════════════════════

    def decide(self, use_llm: bool = False) -> dict:
        """
        运行多 Agent 决策流程。

        参数
        ----------
        use_llm : bool
            True = 使用 LLM API（优先 LangGraph 状态图，不可用则顺序调用）
            False = 使用规则引擎回退

        返回
        -------
        dict: action, confidence, position, reasoning, agents, mode
        """
        if not use_llm:
            return self._rule_based_decision()

        if _HAS_LANGGRAPH:
            return self._decide_langgraph()
        else:
            logger.info("LangGraph未安装，使用顺序LLM调用")
            return self._decide_sequential()

    def backtest_quality(self, decisions: list[dict], returns: pd.Series) -> dict:
        """
        评估历史决策质量。

        参数
        ----------
        decisions : list[dict]
            decide() 返回的决策列表
        returns : Series
            同期日收益率（index 为 date）

        返回
        -------
        dict: accuracy, mean_return, sharpe, hit_rate, n_decisions
        """
        if not decisions:
            return {"accuracy": 0, "mean_return": 0, "sharpe": 0, "n_decisions": 0}

        hits = 0
        decision_returns = []
        for d in decisions:
            date = d.get("date", "")
            if date not in returns.index:
                continue
            ret = returns.loc[date]
            action = d.get("action", "hold")
            if (action == "buy" and ret > 0) or (action == "sell" and ret < 0):
                hits += 1
            decision_returns.append(ret if action == "buy" else -ret if action == "sell" else 0)

        n = len(decision_returns)
        if n == 0:
            return {"accuracy": 0, "mean_return": 0, "sharpe": 0, "n_decisions": 0}

        arr = np.array(decision_returns)
        sr = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0

        return {
            "accuracy": round(hits / n, 4),
            "mean_return": round(float(arr.mean()), 6),
            "sharpe": round(sr, 4),
            "n_decisions": n,
        }
