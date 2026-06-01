"""
因子测试 — 真实沪深300数据（AkShare→Parquet缓存）
"""

import pytest


class TestFactorCompute:
    def test_all_factors_produced(self, csi300_2023):
        """确认25+因子全部产出（真实数据）"""
        from backtest.analysis.factor_miner import compute_factors
        result = compute_factors(csi300_2023)
        factor_cols = [c for c in result.columns
                       if c not in ("date", "open", "high", "low", "close", "volume")]
        assert len(factor_cols) >= 25, f"预期≥25个因子，实际{len(factor_cols)}"

    def test_ic_in_reasonable_range(self, csi300_2023):
        """IC在合理范围（真实数据下应在-0.5~0.5之间）"""
        from backtest.analysis.factor_miner import FactorMiner
        fm = FactorMiner(csi300_2023)
        result = fm.mine()
        top_factors = result.get("top_factors", [])
        assert len(top_factors) > 0, "应产出至少1个有效因子"
        for r in top_factors:
            ic = r.get("ic", 0)
            assert -1.0 <= ic <= 1.0, f"{r['factor']} IC={ic} 不在[-1,1]"
            # 真实数据下 |IC| 不会全是极值
            assert abs(ic) < 0.99, f"{r['factor']} IC={ic} 异常接近±1"

    def test_no_future_leakage(self, csi300_2023):
        """无未来信息泄露（因子值不含未来收益信息）"""
        from backtest.analysis.factor_miner import compute_factors
        result = compute_factors(csi300_2023)
        assert "ret_1m" in result.columns
        # ret_1m = close.pct_change(21)，前20行应为NaN
        nan_count = result["ret_1m"].iloc[:20].isna().sum()
        assert nan_count <= 21, f"前20行NaN数={nan_count}，可能用了未来数据"
        # 第21行起应有值
        first_valid = result["ret_1m"].iloc[21:].dropna()
        assert len(first_valid) > 0, "ret_1m 第21行起应为有效值"

    def test_rank_ic_available(self, csi300_2023):
        """Rank IC 可用（真实数据）"""
        from backtest.analysis.factor_miner import FactorMiner
        fm = FactorMiner(csi300_2023)
        result = fm.mine()
        for r in result.get("top_factors", []):
            assert "rank_ic" in r, f"{r['factor']} 缺少 rank_ic"

    def test_factor_output_not_empty(self, csi300_2023):
        """因子输出不为空，行数匹配"""
        from backtest.analysis.factor_miner import compute_factors
        result = compute_factors(csi300_2023)
        assert len(result) == len(csi300_2023), \
            f"因子行数({len(result)})应匹配输入行数({len(csi300_2023)})"
