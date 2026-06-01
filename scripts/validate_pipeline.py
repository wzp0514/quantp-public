#!/usr/bin/env python
"""
CI/CD 验证脚本（C18）— 真实数据全流程自动验证

拉取真实沪深300数据 → Pipeline全流程 → 验证链路完整性。
每条链路失败时，报告具体的失败节点和原因。

用法
--------
  python scripts/validate_pipeline.py              # 默认配置
  python scripts/validate_pipeline.py --quick      # 快速模式（减数据量）
  python scripts/validate_pipeline.py --full       # 完整模式（多标的+面板）
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, ".")


def validate_data_fetch() -> dict:
    """验证点 1: 数据获取链路"""
    print("\n[1/5] 验证数据获取...")
    from data.fetchers.fallback import fetch_index_daily_safe
    try:
        df = fetch_index_daily_safe("沪深300", "20240101", "20250601")
        passed = len(df) > 200
        return {
            "step": "数据获取",
            "passed": passed,
            "detail": f"沪深300: {len(df)}条{' → OK' if passed else ' → 数据不足'}",
        }
    except Exception as e:
        return {"step": "数据获取", "passed": False, "detail": str(e)}


def validate_factor_compute(df) -> dict:
    """验证点 2: 因子计算链路"""
    print("[2/5] 验证因子计算...")
    from backtest.analysis.factor_miner import compute_factors, FactorMiner
    try:
        fdf = compute_factors(df)
        fm = FactorMiner(df)
        result = fm.mine()
        n_factors = len([c for c in fdf.columns if c not in
                         ("date", "open", "high", "low", "close", "volume")])
        n_strong = result.get("strong_count", 0)
        passed = n_factors >= 30 and n_strong >= 1
        return {
            "step": "因子计算",
            "passed": passed,
            "detail": f"{n_factors}因子, {n_strong}强预测 → {'OK' if passed else '不足'}",
        }
    except Exception as e:
        return {"step": "因子计算", "passed": False, "detail": str(e)}


def validate_backtest(df) -> dict:
    """验证点 3: 回测引擎链路"""
    print("[3/5] 验证回测引擎...")
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    from backtest.engine.bt_runner import run_backtest
    try:
        r = run_backtest(MaCrossStrategy, df, initial_cash=100000, fast=5, slow=20)
        passed = r.get("total_trades", 0) >= 1
        return {
            "step": "回测引擎",
            "passed": passed,
            "detail": (f"年化={r.get('annual_return', 0):.2%}, "
                       f"{r.get('total_trades', 0)}笔交易 → {'OK' if passed else '无交易'}"),
        }
    except Exception as e:
        return {"step": "回测引擎", "passed": False, "detail": str(e)}


def validate_validation(df) -> dict:
    """验证点 4: 专业验证链路"""
    print("[4/5] 验证专业验证...")
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    from backtest.analysis.validate import out_of_sample_test, param_robustness_test
    try:
        oos = out_of_sample_test(MaCrossStrategy, df, split_date="2024-07-01", fast=5, slow=20)
        train_df = df[df["date"] < "2024-07-01"]
        param = param_robustness_test(MaCrossStrategy, train_df, param_name="slow", base_value=20, fast=5)
        passed = oos.get("passed", False) or param.get("passed", False)
        return {
            "step": "专业验证",
            "passed": passed,
            "detail": (f"样本外={oos.get('passed')}, 稳健性={param.get('passed')}"),
        }
    except Exception as e:
        return {"step": "专业验证", "passed": False, "detail": str(e)}


def validate_pipeline(df) -> dict:
    """验证点 5: Pipeline 管线链路"""
    print("[5/5] 验证 Pipeline 管线...")
    try:
        from core.pipeline import Pipeline, Stage
        from backtest.analysis.factor_miner import compute_factors
        pl = Pipeline(df, cash=100000)
        # 添加一条基线 Track（动量因子 + 双均线策略）
        fdf = compute_factors(df)
        scores = fdf.get("momentum_smooth", None)
        if scores is not None:
            pl.add_track(
                "ci_validate",
                factors=["momentum_smooth"],
            )
        pl.run(until_stage=Stage.BACKTEST, auto_recover=False)
        report = pl.report()
        passed = len(pl.tracks) >= 1
        return {
            "step": "Pipeline管线",
            "passed": passed,
            "detail": f"{len(pl.tracks)}条Track → {'OK' if passed else '无Track'}",
        }
    except Exception as e:
        return {"step": "Pipeline管线", "passed": False, "detail": str(e)}


def main():
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("  QuantP CI/CD 全链路验证")
    print("=" * 60)
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模式: {'快速' if quick else '标准'}")
    print()

    start_time = time.time()
    results = []

    # 1. 数据获取
    r1 = validate_data_fetch()
    results.append(r1)
    print(f"  → {r1['detail']}")
    if not r1["passed"]:
        print("数据获取失败，终止后续验证。")
        print_summary(results, start_time)
        return

    # 需要真实 df
    from data.fetchers.fallback import fetch_index_daily_safe
    df = fetch_index_daily_safe("沪深300", "20240101", "20250601")

    # 2-5
    for validate_fn in [validate_factor_compute, validate_backtest,
                         validate_validation, validate_pipeline]:
        r = validate_fn(df)
        results.append(r)
        print(f"  → {r['detail']}")

    print_summary(results, start_time)


def print_summary(results, start_time):
    elapsed = time.time() - start_time
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    overall = "PASS" if passed == total else "FAIL"

    print("\n" + "=" * 60)
    print("  验证结果汇总")
    print("=" * 60)
    for r in results:
        flag = "[OK]" if r["passed"] else "[!!]"
        print(f"  {flag} {r['step']}: {r['detail']}")
    print(f"\n  通过: {passed}/{total} | 耗时: {elapsed:.1f}s | 结论: {overall}")
    print("=" * 60)

    if overall == "PASS":
        print("\n全链路验证通过 — 数据获取→因子→回测→验证→Pipeline 均正常。")
    else:
        failed = [r for r in results if not r["passed"]]
        print(f"\n{len(failed)} 条链路失败，请排查对应模块。")


if __name__ == "__main__":
    main()
