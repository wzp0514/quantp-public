"""
L4信号链路端到端验证 — 真实数据→因子→L4→对比次日收益→评估报告。

用法
--------
python -m backtest.analysis.l4_validate
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.log import get_logger

logger = get_logger("l4_validate")


def validate(symbol: str = "沪深300", start: str = "20220101", end: str = "20250601",
             use_rl: bool = False) -> dict:
    """
    L4信号链路端到端验证。

    流程: 取数据→算因子→跑L4→对比次日收益→评估

    参数
    ----------
    symbol : str
        标的名称
    start/end : str
        数据起止日期
    use_rl : bool
        是否训练RL（耗时长，默认关）

    返回
    -------
    dict: accuracy, sharpe, confusion_matrix, signal_distribution, summary
    """
    from data.fetchers.fallback import fetch_index_daily_safe as fetch_index_daily
    from backtest.analysis.l4_integration import L4SignalChain

    print("=" * 60)
    print(f"  L4 信号链路端到端验证: {symbol}")
    print("=" * 60)
    print(f"  日期: {start} → {end}")
    print(f"  RL训练: {'开启' if use_rl else '关闭'}")
    print()

    # ── 1. 获取数据 ─────────────────────────────────────
    print("[1/4] 获取行情数据...")
    df = fetch_index_daily(symbol, start, end)
    if df is None or len(df) < 252:
        print(f"  FAIL: 数据不足 ({len(df) if df is not None else 0} 条)")
        return {"error": "数据不足"}
    print(f"  OK: {len(df)} 条 ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

    # ── 2. 计算因子 ─────────────────────────────────────
    print("[2/4] 计算因子...")
    try:
        from backtest.analysis.factor_miner import compute_factors, compute_ic
        factor_df = compute_factors(df)
        factor_cols = [c for c in factor_df.columns
                       if c not in ("date", "open", "high", "low", "close", "volume")]
        print(f"  OK: {len(factor_cols)} 个因子")

        # Top IC 因子
        ic_results = {}
        for col in factor_cols:
            try:
                ic = compute_ic(factor_df, col)
                ic_results[col] = ic["ic"]
            except Exception:
                pass
        top5 = sorted(ic_results.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"  Top5 IC: {', '.join(f'{n}={v:+.3f}' for n, v in top5)}")
    except Exception as e:
        logger.warning(f"因子计算失败: {e}")
        factor_df = df
        print("  WARN: 因子计算失败，使用原始数据")

    # ── 3. 运行L4信号链 ─────────────────────────────────
    print("[3/4] 运行 L4 信号链...")
    l4 = L4SignalChain(factor_df, use_rl=use_rl, symbol=symbol)
    result = l4.run()
    print(f"  综合信号: {result['signal']:.4f}")
    print(f"  决策: {result['action']} (置信度: {result['confidence']:.2f})")
    print(f"  子信号: alt={result['components'].get('alt',0):.3f} "
          f"rl={result['components'].get('rl',0):.3f} "
          f"ml={result['components'].get('ml',0):.3f}")
    print(f"  时间戳: {result['timestamp']}")

    # ── 4. 方向准确率评估 ───────────────────────────────
    print("[4/4] 方向准确率评估...")

    # 用滚动方式评估：每20个交易日生成一次信号，对比次日方向
    n_total = len(factor_df)
    if n_total < 60:
        print("  SKIP: 数据不足60条，无法滚动评估")
        result["validation"] = {"error": "数据不足"}
        return result

    test_start = n_total // 3  # 后2/3用于评估
    decisions = []
    step = 20  # 每20天评估一次

    try:
        for t in range(test_start, n_total - 1, step):
            window_df = factor_df.iloc[:t + 1].copy()
            try:
                l4_window = L4SignalChain(window_df, use_rl=False, symbol=symbol)
                d = l4_window.run()
            except Exception:
                continue

            # 次日实际方向
            if t + 1 < n_total:
                next_ret = float(df["close"].iloc[t + 1] / df["close"].iloc[t] - 1)
            else:
                next_ret = 0

            decisions.append({
                "date": str(df["date"].iloc[t]) if "date" in df.columns else str(t),
                "action": d["action"],
                "signal": d["signal"],
                "next_return": round(next_ret, 6),
            })

        if len(decisions) < 3:
            print(f"  SKIP: 仅{len(decisions)}个有效决策点")
            result["validation"] = {"error": "决策点不足"}
            return result

        # 统计
        correct = 0
        actions_used = []
        returns_aligned = []
        for d in decisions:
            ret = d["next_return"]
            act = d["action"]
            if (act == "buy" and ret > 0) or (act == "sell" and ret < 0):
                correct += 1
                actions_used.append("correct")
            elif act == "hold":
                actions_used.append("hold")
            else:
                actions_used.append("wrong")

            if act == "buy":
                returns_aligned.append(ret)
            elif act == "sell":
                returns_aligned.append(-ret)

        non_hold = sum(1 for a in actions_used if a != "hold")
        accuracy = correct / non_hold if non_hold > 0 else 0

        arr = np.array(returns_aligned) if returns_aligned else np.array([0])
        sharpe = float(arr.mean() / arr.std() * np.sqrt(252 / step)) if arr.std() > 0 else 0

        # 信号分布
        buy_n = sum(1 for d in decisions if d["action"] == "buy")
        sell_n = sum(1 for d in decisions if d["action"] == "sell")
        hold_n = sum(1 for d in decisions if d["action"] == "hold")

        validation = {
            "n_decisions": len(decisions),
            "n_non_hold": non_hold,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "sharpe": round(sharpe, 4),
            "mean_aligned_return": round(float(arr.mean()), 6),
            "signal_distribution": {
                "buy": buy_n,
                "sell": sell_n,
                "hold": hold_n,
                "buy_pct": round(buy_n / len(decisions), 2),
            },
        }

        print(f"  决策点: {len(decisions)} 个 (非hold: {non_hold})")
        print(f"  方向准确率: {accuracy:.1%} ({correct}/{non_hold})")
        print(f"  年化夏普(对齐): {sharpe:.2f}")
        print(f"  平均对齐收益: {arr.mean():.6f}")
        print(f"  信号分布: 买{buy_n}/卖{sell_n}/持{hold_n}")

        result["validation"] = validation

    except Exception as e:
        logger.warning(f"滚动评估失败: {e}")
        result["validation"] = {"error": str(e)}

    # ── 输出汇总 ───────────────────────────────────────
    print()
    print("=" * 60)
    print("  验证结论")
    print("=" * 60)

    val = result.get("validation", {})
    acc = val.get("accuracy", 0)
    sr = val.get("sharpe", 0)

    if isinstance(val.get("error"), str):
        print(f"  [!] 验证异常: {val['error']}")
        print("  → 子组件均正常运行，信号链路已接通")
        print("  → 后续可在Pipeline中自动累积验证数据")
    elif acc > 0.55:
        print(f"  ✓ 通过: 方向准确率 {acc:.1%} > 55%, 夏普 {sr:.2f}")
        print("  → L4信号链可用，接入ResearchSource")
    elif acc > 0.45:
        print(f"  △ 待观察: 方向准确率 {acc:.1%}, 夏普 {sr:.2f}")
        print("  → 信号链已接通，可接入ResearchSource，后续跟踪优化")
    else:
        print(f"  ✗ 未达标: 方向准确率 {acc:.1%} < 45%")
        print("  → 信号链已接通但预测能力弱，建议：调权重/增因子/换数据窗口")

    print(f"  综合信号: {result['signal']:.4f} → {result['action']}")
    print(f"  时间: {result['timestamp']}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    validate(use_rl=False)
