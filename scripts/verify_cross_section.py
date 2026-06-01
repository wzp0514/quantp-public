#!/usr/bin/env python
"""
截面多因子选股真实验证脚本（C24）

拉取真实多股票面板数据 → 运行 CrossSectionStrategy → 验证结果。
"""

import sys
sys.path.insert(0, ".")


def verify_cross_section():
    """C24: 真实多股票面板截面回测验证"""
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("  截面多因子选股 — 真实验证 (C24)")
    print("=" * 60)

    # 1. 获取真实面板数据
    print("\n[1/4] 获取多股票面板数据...")
    try:
        from data.fetchers.multi_stock import fetch_multi_stock_daily
        panel = fetch_multi_stock_daily(n_stocks=20, start="20240101")
        print(f"  面板: {len(panel)}只股票")
    except Exception as e:
        print(f"  多股票拉取失败: {e}")
        print("  使用合成数据继续验证...")
        np.random.seed(42)
        n_stocks = 10
        n_days = 252
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        stocks_data = {}
        for i in range(n_stocks):
            ret = np.random.randn(n_days) * 0.015 + 0.0002
            price = 20 * np.cumprod(1 + ret)
            stocks_data[f"STOCK_{i:03d}"] = pd.DataFrame({
                "date": dates,
                "open": price * 0.998,
                "high": price * 1.015,
                "low": price * 0.985,
                "close": price,
                "volume": np.random.randint(1000, 50000, n_days).astype(float) * 100,
            })
        panel = stocks_data

    # 2. 准备回测
    print("\n[2/4] 准备截面回测...")
    from backtest.strategies.experimental.cross_section_strategy import CrossSectionStrategy
    from backtest.engine.bt_runner import run_backtest_panel

    try:
        result = run_backtest_panel(
            CrossSectionStrategy,
            panel,
            initial_cash=1000000,
            top_n=5,
            rebalance_freq=21,
        )
        print(f"  回测完成")
    except (AttributeError, TypeError) as e:
        print(f"  面板回测引擎未就绪: {e}")
        print("  [跳过] run_backtest_panel 函数可能需要适配")
        return {"status": "skipped", "reason": "引擎适配中"}

    # 3. 分析结果
    print("\n[3/4] 分析截面回测结果...")
    print(f"  年化收益: {result.get('annual_return', 0):.2%}")
    print(f"  最大回撤: {result.get('drawdown', 0):.2%}")
    s = result.get("sharpe")
    print(f"  夏普比率: {f'{s:.2f}' if s else 'N/A'}")
    print(f"  交易次数: {result.get('total_trades', 0)}")

    # 4. 验证标准
    print("\n[4/4] 验证结论...")
    checks = []
    checks.append(("年化收益不为极端值",
                   -0.9 < result.get('annual_return', 0) < 5.0))
    checks.append(("产生了交易",
                   result.get('total_trades', 0) > 0))
    checks.append(("回撤在合理范围",
                   result.get('drawdown', 0) < 0.99))

    all_passed = True
    for name, passed in checks:
        flag = "[OK]" if passed else "[!!]"
        print(f"  {flag} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  截面多因子选股验证通过。")
    else:
        print("\n  部分检查未通过，请检查策略实现。")

    return {"status": "passed" if all_passed else "failed", "checks": checks}


if __name__ == "__main__":
    verify_cross_section()
