"""
Agent决策层结构化输出模型 — Pydantic schemas。

用于 LangGraph 状态图中各节点的输入/输出类型约束。
"""

from typing import Optional, TypedDict, Annotated
import operator


class AnalystReport(TypedDict, total=False):
    """单个分析师的输出报告"""
    role: str                       # 角色名（技术分析师/因子分析师/情绪分析师）
    bias: str                       # bullish / bearish / neutral
    score: int                      # 0-100，>50偏多 <50偏空
    key_points: list[str]           # 关键判断依据
    confidence: float               # 0-1，分析师自评置信度


class DebateResult(TypedDict, total=False):
    """研究员多空辩论结论"""
    decision: str                   # buy / sell / hold
    confidence: float               # 0-1
    bull_args: list[str]            # 多方论据
    bear_args: list[str]            # 空方论据
    synthesis: str                  # 综合判断（50字）


class TradePlan(TypedDict, total=False):
    """交易员下单计划"""
    action: str                     # buy / sell / hold
    position_pct: float             # 建议仓位比例（0-0.3）
    limit_price: Optional[float]    # 限价（None=市价）
    reason: str                     # 下单理由（30字）


class RiskReview(TypedDict, total=False):
    """风控审核结果"""
    approved: bool                  # 是否通过审核
    adjusted_position: float        # 调整后仓位
    risk_flags: list[str]           # 风险标记
    risk_level: str                 # low / medium / high
    note: str                       # 审核意见（30字）


class AgentState(TypedDict):
    """
    LangGraph 全局状态。

    各节点读取并更新此状态，实现多Agent协作。
    """
    # 输入
    symbol: str
    snapshot: dict                  # 市场快照（通用/旧接口）
    factor_signals: dict            # 因子信号（通用/旧接口）
    sentiment_text: str             # 外部情绪文本（通用/旧接口）
    # 角色专属视角数据
    tech_view: str                  # 技术分析师专属：K线+指标
    factor_view: str                # 因子分析师专属：IC排名表
    sentiment_view: str             # 情绪分析师专属：新闻+涨跌方向

    # 中间结果
    analyst_reports: Annotated[list[AnalystReport], operator.add]
    debate_result: Optional[DebateResult]
    trade_plan: Optional[TradePlan]
    risk_review: Optional[RiskReview]

    # 最终输出
    action: str                     # buy / sell / hold
    confidence: float
    position: float
    reasoning: str
    mode: str                       # llm / rule_based
    error: Optional[str]
