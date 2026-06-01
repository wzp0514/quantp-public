"""
TaskRunner 单元测试 — 测试 TaskSpec/TaskResult/TaskRunner 基础逻辑
"""
import pytest
from core.runner import TaskSpec, TaskResult, TaskRunner


class TestTaskSpec:
    def test_defaults(self):
        spec = TaskSpec(kind="backtest")
        assert spec.kind == "backtest"
        assert spec.symbol == "沪深300"
        assert spec.cash == 100000.0

    def test_custom_fields(self):
        spec = TaskSpec(kind="mine", symbol="中证500",
                        start="20200101", cash=50000.0,
                        max_combos=100, min_trades=20)
        assert spec.max_combos == 100
        assert spec.min_trades == 20
        assert spec.cash == 50000.0

    def test_extra_params_default(self):
        spec = TaskSpec(kind="backtest")
        assert spec.extra_params == {}

    def test_replay_fields(self):
        spec = TaskSpec(kind="replay", replay_days=120,
                        replay_indicators=["ma20", "ma60"])
        assert spec.replay_days == 120
        assert len(spec.replay_indicators) == 2


class TestTaskResult:
    def test_defaults(self):
        result = TaskResult()
        assert result.success is False
        assert result.kind == ""
        assert result.data is None
        assert result.error == ""

    def test_success_result(self):
        result = TaskResult(success=True, kind="backtest",
                           data={"sharpe": 1.5}, elapsed=2.3,
                           summary="All good")
        assert result.success is True
        assert result.data["sharpe"] == 1.5

    def test_error_result(self):
        result = TaskResult(success=False, kind="backtest",
                           error="数据拉取失败", elapsed=1.2)
        assert result.success is False
        assert "数据拉取失败" in result.error


class TestTaskRunner:
    def test_init_registers_defaults(self, runner):
        assert "backtest" in runner._handlers
        assert "shootout" in runner._handlers
        assert "mine" in runner._handlers
        assert "paper" in runner._handlers

    def test_unknown_kind_returns_error(self, runner):
        spec = TaskSpec(kind="nonexistent_kind")
        result = runner.run(spec)
        assert result.success is False
        assert "Unknown" in result.error

    def test_register_custom_handler(self, runner):
        def my_handler(spec):
            return TaskResult(success=True, kind="custom",
                             summary="Custom done")
        runner.register("custom", my_handler)
        assert "custom" in runner._handlers
        result = runner.run(TaskSpec(kind="custom"))
        assert result.success is True
        assert result.summary == "Custom done"

    def test_handler_exception_caught(self, runner):
        def bad_handler(spec):
            raise RuntimeError("Boom")
        runner.register("bad", bad_handler)
        result = runner.run(TaskSpec(kind="bad"))
        assert result.success is False
        assert "Boom" in result.error
        assert result.elapsed >= 0


@pytest.fixture
def runner():
    return TaskRunner()
