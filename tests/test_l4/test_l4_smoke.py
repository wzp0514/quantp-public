"""
L4信号链路烟雾测试 — 真实数据，不调用外部API（RL/LLM）。

验证: 导入→实例化→三个子组件运行→run()产出合理结果
"""

import pytest


class TestL4SignalChain:
    def test_import_and_create(self, csi300_2023):
        """L4SignalChain 可导入并实例化"""
        from backtest.analysis.l4_integration import L4SignalChain
        l4 = L4SignalChain(csi300_2023, use_rl=False)
        assert l4.df is not None
        assert len(l4.factor_names) > 0

    def test_alt_data_score(self, csi300_2023):
        """_alt_data_score 返回 [0,1] 范围"""
        from backtest.analysis.l4_integration import L4SignalChain
        l4 = L4SignalChain(csi300_2023, use_rl=False)
        score = l4._alt_data_score()
        assert 0.0 <= score <= 1.0, f"alt_score={score} 不在[0,1]"

    def test_factor_ml_score(self, csi300_2023):
        """_factor_ml_score 返回 [0,1] 范围"""
        from backtest.analysis.l4_integration import L4SignalChain
        l4 = L4SignalChain(csi300_2023, use_rl=False)
        score = l4._factor_ml_score()
        assert 0.0 <= score <= 1.0, f"ml_score={score} 不在[0,1]"

    def test_run_produces_valid_output(self, csi300_2023):
        """run() 产出完整结构"""
        from backtest.analysis.l4_integration import L4SignalChain
        l4 = L4SignalChain(csi300_2023, use_rl=False)
        result = l4.run()
        assert "signal" in result
        assert "action" in result
        assert "confidence" in result
        assert "components" in result
        assert result["action"] in ("buy", "sell", "hold")
        for key in ("alt", "rl", "ml"):
            assert key in result["components"], f"缺少component: {key}"

    def test_generate_backward_compat(self, csi300_2023):
        """旧接口 generate() 兼容"""
        from backtest.analysis.l4_integration import L4SignalChain
        l4 = L4SignalChain(csi300_2023, use_rl=False)
        result = l4.generate()
        assert result["action"] in ("buy", "sell", "hold")

    def test_validate_script_importable(self):
        """验证脚本可导入"""
        from backtest.analysis.l4_validate import validate
        assert callable(validate)
