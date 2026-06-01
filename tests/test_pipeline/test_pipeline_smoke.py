"""
管线全流程测试（T5: 单Track 5阶段走通+auto_recover）

运行：pytest tests/test_pipeline/ -v          # 全部
      pytest tests/test_pipeline/ -v -m "not slow"  # 仅快速
"""

import pytest
from core.pipeline import Pipeline, Stage, Track


class TestPipelineBasics:
    """快速测试：创建/导入/基本操作（不含 run）"""

    def test_pipeline_import(self):
        """Pipeline/Stage/Track 可导入"""
        assert Stage.FACTOR.value == 1
        assert len(Stage) >= 5

    def test_pipeline_create_with_real_data(self, csi300_2023):
        """使用真实沪深300数据创建Pipeline"""
        pl = Pipeline(csi300_2023, cash=100000)
        assert len(pl.df) == 242

    def test_add_track_and_status(self, csi300_2023):
        """添加Track+状态"""
        pl = Pipeline(csi300_2023)
        t = pl.add_track("test_track", factors=["ret_3m", "ret_6m"])
        assert len(pl.tracks) == 1
        assert t.stage == Stage.FACTOR
        assert "FACTOR" in t.status() or "Factor" in t.status()

    def test_feed_signals_no_crash(self, csi300_2023):
        """feed_signals() 不崩溃"""
        pl = Pipeline(csi300_2023, cash=100000)
        assert pl.feed_signals([]) == 0


@pytest.mark.slow
class TestPipelineFullFlow:
    """慢测试：完整管线流程（真实数据 + 因子挖掘 + LLM + 回测，每个约 3-8 min）"""

    def test_stage1_to_3_flow(self, csi300_2023):
        """单Track走通 Stage1→2→3：因子挖掘→回测→验证（门控验证）"""
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        pl = Pipeline(csi300_2023, cash=100000)
        pl.add_track("MA_5_20", strategy_class=MaCrossStrategy,
                     strategy_params={"fast": 5, "slow": 20})
        pl.run(until_stage=Stage.VALIDATE)

        t = pl.tracks[0]
        assert t.stage != Stage.FACTOR, "应该离开FACTOR阶段"
        # 门控在工作：要么晋级到 VALIDATE+，要么归档
        assert t.stage in (Stage.BACKTEST, Stage.VALIDATE, Stage.PAPER,
                           Stage.READY, Stage.ARCHIVED)
        if t.stage == Stage.ARCHIVED:
            assert t.archived_reason != ""
        if t.stage.value >= Stage.VALIDATE.value:
            assert t.backtest_result is not None
            assert "wfa_passed" in t.validation_result
            assert "dsr_significant" in t.validation_result

    def test_stage4_paper_trading(self, csi300_2023):
        """Stage4 纸上交易：PaperTrader + StrategyGuard + 风控 + 偏差（手动进PAPER验证机制可用）"""
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        pl = Pipeline(csi300_2023, cash=100000)
        t = pl.add_track("paper_test", strategy_class=MaCrossStrategy,
                         strategy_params={"fast": 5, "slow": 20})
        # 手动置为 PAPER 阶段（绕过 S2/S3 门控，直接测 S4 机制）
        t.stage = Stage.PAPER
        t.backtest_result = {"sharpe": 1.5, "total_trades": 20}

        pl.run(until_stage=Stage.PAPER)
        # PaperTrader 运行不崩溃，要么晋级 READY 要么合理归档
        assert t.stage in (Stage.PAPER, Stage.READY, Stage.ARCHIVED)
        if t.stage == Stage.ARCHIVED:
            # 归档原因应为 PaperTrader 门控（非报错）
            assert "error" not in t.archived_reason.lower()

    def test_auto_recover_mechanism(self, csi300_2015_2023):
        """auto_recover 机制验证：_recover_track 接线正确，不崩溃，返回 list"""
        from backtest.strategy_miner import ENTRY_RULES, EXIT_RULES

        pl = Pipeline(csi300_2015_2023, cash=100000)
        t = pl.add_track("recover_test")
        t.entry_type = list(ENTRY_RULES.keys())[0]   # ma_cross_up
        t.exit_type = list(EXIT_RULES.keys())[0]      # ma_cross_down

        # _recover_track 返回 list（Miner 严格过滤可能导致空，但不应崩溃）
        new_tracks = pl._recover_track(t)
        assert isinstance(new_tracks, list), \
            f"_recover_track 应返回 list，actual={type(new_tracks)}"
        for nt in new_tracks:
            assert "_v" in nt.name
            assert nt.stage == Stage.BACKTEST

    def test_stage5_ready_promotion(self, csi300_2023):
        """Stage 5 晋级验证：直接注入 READY track 验证完整 5 阶段链路"""
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        pl = Pipeline(csi300_2023, cash=100000)
        t = pl.add_track("ready_test", strategy_class=MaCrossStrategy,
                         strategy_params={"fast": 5, "slow": 20})
        # 手动将所有阶段产物注入，模拟已通过全部验证的策略
        t.stage = Stage.PAPER
        t.backtest_result = {"sharpe": 1.2, "total_trades": 25}
        t.validation_result = {"wfa_passed": True, "wfa_winrate": 0.6,
                               "wfa_wfe": 0.55, "dsr_significant": True,
                               "deflated_sr": 1.0}
        t.final_score = 1.2
        # 跑 Paper → READY
        pl.run(until_stage=Stage.PAPER)

        # Stage 5 READY: 从 Paper 晋级后即为 READY
        ready = pl.ready_strategies()
        # 如果 PaperTrader 通过，track 变为 READY
        if t.stage == Stage.READY:
            assert len(ready) >= 1
            assert ready[0].name == "ready_test"
        # 否则至少验证了 PaperTrader 门控工作正常
        assert t.stage in (Stage.PAPER, Stage.READY, Stage.ARCHIVED)

    def test_report_and_ready(self, csi300_2023):
        """管线报告生成 + 候选实盘列表"""
        from backtest.strategies.builtin.ma_cross import MaCrossStrategy

        pl = Pipeline(csi300_2023, cash=100000)
        pl.add_track("MA_5_20", strategy_class=MaCrossStrategy,
                     strategy_params={"fast": 5, "slow": 20})
        pl.run(until_stage=Stage.VALIDATE)

        report = pl.report()
        assert "PIPELINE REPORT" in report
        ready = pl.ready_strategies()
        assert isinstance(ready, list)
