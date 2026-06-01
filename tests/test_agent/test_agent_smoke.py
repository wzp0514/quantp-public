"""
Agent决策层烟雾测试 — 不调用LLM API，只验证规则引擎回退和结构完整性。

LLM模式需网络+API Key，CI环境不可用时跳过。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_agent_df():
    """生成带趋势的测试行情（非纯随机，用于规则引擎有意义测试）"""
    n = 252
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # 生成带趋势+噪声的价格（前半段上涨，后半段震荡）
    np.random.seed(42)
    trend = np.concatenate([
        np.linspace(0, 0.3, n // 2),   # 上涨 30%
        np.sin(np.linspace(0, np.pi * 3, n - n // 2)) * 0.1,  # 震荡
    ])
    noise = np.random.randn(n) * 0.015
    returns = trend / (n // 2) + noise
    price = 100 * np.cumprod(1 + returns)

    return pd.DataFrame({
        "date": dates,
        "open": price * 0.998,
        "high": price * 1.012,
        "low": price * 0.988,
        "close": price,
        "volume": np.random.randint(1000, 50000, n).astype(float) * 100,
    })


class TestAgentDecisionRuleBased:
    def test_import_and_create(self, sample_agent_df):
        """AgentDecision 可导入并实例化"""
        from backtest.analysis.agent_decision import AgentDecision
        ad = AgentDecision(sample_agent_df)
        assert ad.snapshot is not None
        assert "price" in ad.snapshot

    def test_rule_based_decision(self, sample_agent_df):
        """规则引擎模式产出完整结构"""
        from backtest.analysis.agent_decision import AgentDecision
        ad = AgentDecision(sample_agent_df)
        result = ad.decide(use_llm=False)
        assert result["action"] in ("buy", "sell", "hold")
        assert 0 <= result["confidence"] <= 1
        assert result["mode"] == "rule_based"
        assert "agents" in result
        assert "reasoning" in result

    def test_rule_based_with_factor_signals(self, sample_agent_df):
        """带因子信号的规则引擎决策"""
        from backtest.analysis.agent_decision import AgentDecision
        factor_signals = {"strong_count": 7, "top_factors": [
            {"factor": "mom_20", "ic": 0.08},
            {"factor": "vol_60", "ic": -0.05},
        ]}
        ad = AgentDecision(sample_agent_df, factor_signals=factor_signals)
        result = ad.decide(use_llm=False)
        assert result["action"] in ("buy", "sell", "hold")

    def test_snapshot_values_sane(self, sample_agent_df):
        """市场快照值在合理范围"""
        from backtest.analysis.agent_decision import AgentDecision
        ad = AgentDecision(sample_agent_df)
        snap = ad.snapshot
        assert snap["price"] > 0
        assert -100 < snap["change_5d"] < 100
        assert snap["volatility_20d"] >= 0
        assert snap["volume_ratio"] > 0

    def test_backtest_quality(self, sample_agent_df):
        """backtest_quality 评估函数"""
        from backtest.analysis.agent_decision import AgentDecision
        ad = AgentDecision(sample_agent_df)
        decisions = [
            {"date": "2024-06-15", "action": "buy"},
            {"date": "2024-06-16", "action": "sell"},
            {"date": "2024-06-17", "action": "hold"},
        ]
        ret_index = pd.to_datetime(["2024-06-15", "2024-06-16", "2024-06-17"])
        returns = pd.Series([0.01, -0.02, 0.005], index=ret_index)
        quality = ad.backtest_quality(decisions, returns)
        assert "accuracy" in quality
        assert quality["n_decisions"] > 0


class TestAgentStore:
    def test_init_and_save(self, tmp_path):
        """AgentStore 建表+保存+查询"""
        from backtest.analysis.agent_store import AgentStore
        db_path = str(tmp_path / "test_agent.db")
        store = AgentStore(db_path=db_path)

        rid = store.save(
            symbol="沪深300", action="buy", confidence=0.85,
            position_pct=0.2, reasoning="测试决策", mode="rule_based",
        )
        assert rid > 0

        results = store.query(days=1)
        assert len(results) > 0
        assert results[0]["action"] == "buy"

    def test_save_from_result(self, tmp_path):
        """从 decide() 结果保存"""
        from backtest.analysis.agent_store import AgentStore
        db_path = str(tmp_path / "test_agent2.db")
        store = AgentStore(db_path=db_path)

        result = {
            "action": "sell", "confidence": 0.7, "position": 0.15,
            "reasoning": "趋势转弱", "mode": "rule_based",
            "risk_flags": ["高波动"], "agents": {"tech_analyst": {"bias": "bearish"}},
            "timestamp": "2024-06-15T10:00:00",
        }
        store.save_from_result(result, symbol="中证500")

        results = store.query(days=30)
        assert len(results) == 1
        assert results[0]["symbol"] == "中证500"

    def test_stats(self, tmp_path):
        """决策统计"""
        from backtest.analysis.agent_store import AgentStore
        db_path = str(tmp_path / "test_agent3.db")
        store = AgentStore(db_path=db_path)

        store.save(symbol="test", action="buy", confidence=0.8, position_pct=0.2)
        store.save(symbol="test", action="sell", confidence=0.6, position_pct=0.1)
        store.save(symbol="test", action="hold", confidence=0.5)

        stats = store.stats(days=1)
        assert stats["total_decisions"] == 3
        assert "buy" in stats["action_distribution"]


class TestLLMProvider:
    def test_detect_providers(self):
        """_detect_providers 返回有效结构"""
        from backtest.analysis.llm_provider import _detect_providers
        providers = _detect_providers()
        assert isinstance(providers, dict)
        for p in ("anthropic", "openai", "deepseek"):
            assert p in providers, f"缺少 provider: {p}"
            assert "sdk" in providers[p]


class TestAgentSchemas:
    def test_schemas_importable(self):
        """所有 schema 可导入"""
        from backtest.analysis.agent_schemas import (
            AnalystReport, DebateResult, TradePlan, RiskReview, AgentState,
        )
        for cls in (AnalystReport, DebateResult, TradePlan, RiskReview, AgentState):
            assert cls is not None


class TestAgentLLMMock:
    """Mock _call_llm 测试完整LLM链路，不调用真实API"""

    @staticmethod
    def _mock_llm_response(system_prompt: str, user_prompt: str,
                           max_tokens: int = 500, model: str = "",
                           temperature: float = 0.3) -> str:
        """根据角色返回合适的mock JSON，验证system prompt→parse→assemble链路"""
        import json
        if "技术分析师" in system_prompt:
            return json.dumps({"bias": "bullish", "score": 72, "key_points": ["动量向上", "放量突破"], "confidence": 0.75})
        if "因子分析师" in system_prompt:
            return json.dumps({"bias": "bullish", "score": 68, "key_points": ["IC正向", "IR稳定"], "confidence": 0.70})
        if "情绪分析师" in system_prompt:
            return json.dumps({"bias": "bullish", "score": 65, "key_points": ["市场情绪回暖"], "confidence": 0.65})
        if "资深量化研究员" in system_prompt or "辩论" in user_prompt.lower():
            return json.dumps({"decision": "buy", "confidence": 0.72, "bull_args": ["动量+因子共振"], "bear_args": ["波动偏高"], "synthesis": "多方占优，建议买入"})
        if "执行力极强" in system_prompt:
            return json.dumps({"action": "buy", "position_pct": 0.18, "limit_price": None, "reason": "信号共振，果断建仓"})
        if "风控官" in system_prompt or "风控" in system_prompt:
            return json.dumps({"approved": True, "adjusted_position": 0.15, "risk_flags": [], "note": "合规通过"})
        return "{}"

    def test_sequential_path_with_mock(self, sample_agent_df):
        """mock后走通LLM调用完整链路（含分析师→辩论→交易员→风控）"""
        from unittest.mock import patch
        from backtest.analysis.agent_decision import AgentDecision

        with patch("backtest.analysis.agent_decision._call_llm",
                   side_effect=self._mock_llm_response):
            ad = AgentDecision(sample_agent_df)
            result = ad.decide(use_llm=True)

        assert result["action"] in ("buy", "sell", "hold")
        assert 0 <= result["confidence"] <= 1
        assert result["mode"] in ("llm_sequential", "llm_langgraph")
        assert "reasoning" in result
        # 验证所有角色都参与了
        agents = result.get("agents", {})
        if result["mode"] == "llm_langgraph":
            # LangGraph: analysts是列表, debate/trade/risk_officer是顶层key
            assert "analysts" in agents, "缺少 analysts"
            assert "debate" in agents, "缺少 debate"
            assert "trade" in agents, "缺少 trade"
            assert "risk_officer" in agents, "缺少 risk_officer"
            analyst_roles = {a.get("role", "") for a in agents["analysts"]}
            for role in ("技术分析师", "因子分析师", "情绪分析师"):
                assert role in analyst_roles, f"缺少分析师角色: {role} ({analyst_roles})"
        else:
            # Sequential: 各角色是顶级key
            for role in ("tech_analyst", "factor_analyst", "sentiment_analyst", "debate", "trade", "risk_officer"):
                assert role in agents, f"缺少角色: {role}"

    def test_mock_handles_failure_gracefully(self, sample_agent_df):
        """LLM返回空/非法JSON时不崩溃，回退到默认值"""
        from unittest.mock import patch
        from backtest.analysis.agent_decision import AgentDecision

        with patch("backtest.analysis.agent_decision._call_llm",
                   return_value="invalid {{ json"):
            ad = AgentDecision(sample_agent_df)
            result = ad.decide(use_llm=True)

        # 应优雅降级，不崩溃
        assert result["action"] in ("buy", "sell", "hold")
        assert "agents" in result

    def test_single_node_mock(self, sample_agent_df):
        """单独mock技术分析师节点，验证数据视角+人格注入正确"""
        from unittest.mock import patch
        from backtest.analysis.agent_decision import AgentDecision, _call_llm

        captured_system = []
        def capture_and_return(system, user, **kwargs):
            captured_system.append(system)
            return '{"bias":"bullish","score":75,"key_points":["均线金叉"],"confidence":0.8}'

        with patch("backtest.analysis.agent_decision._call_llm",
                   side_effect=capture_and_return):
            ad = AgentDecision(sample_agent_df)
            # 只测技术分析师视角构建
            view = ad._build_tech_view()
            assert "RSI" in view
            assert "MACD" in view
            assert "布林带" in view
            # 验证数据视角隔离：技术视图不含IC值/因子排名/情绪文本
            assert "IC" not in view
            assert "external sentiment" not in view.lower()
