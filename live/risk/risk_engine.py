"""
风控引擎 — 交易的安全带和安全气囊

核心作用：
  当亏损达到设定值时，自动暂停或停止交易，防止你"上头"后越亏越多。

所有参数从 config/settings.yaml 读取（可手动调整）。
修改后重启程序即生效，无需改代码。

=========== 手动调整指南 ===========
如果你风险承受能力很低，可以在 config/settings.yaml 中把参数改小：

┌──────────────────────┬──────────┬──────────────┬──────────────────────────────┐
│ 参数                  │ 标准值    │ 保守值        │ 含义                          │
├──────────────────────┼──────────┼──────────────┼──────────────────────────────┤
│ max_position_pct     │ 20%      │ 10%          │ 单策略最多占总资金百分比         │
│ max_single_loss_pct  │ 2%       │ 1%           │ 单笔亏损超此值强制卖出           │
│ max_daily_loss_pct   │ 5%       │ 2%           │ 当天总亏损超此值停止交易          │
│ max_drawdown_pct     │ 15%      │ 10%          │ 策略总回撤超此值永久停用          │
│ stop_after_n_losses  │ 5        │ 3            │ 连续亏损 N 笔后暂停              │
│ max_orders_per_hour  │ 20       │ 10           │ 每小时最大报单次数（防空哥事件）    │
│ martingale_multiplier│ 1.5      │ 1.3          │ 亏损后仓位扩大倍数检测上限         │
│ no_leverage          │ true     │ true（不要改）│ 是否使用杠杆（个人强烈建议不开）    │
└──────────────────────┴──────────┴──────────────┴──────────────────────────────┘

配置位置: config/settings.yaml → risk 段
修改后重启程序即生效。
===================================
"""

import logging

from config.loader import get_config, reload_config
# ============================================================
# 硬编码兜底值（当配置文件不存在或值非法时使用）
# ============================================================
# 注意：兜底值采用保守值，宁可过度保守也不能暴露风险

from config.log import get_logger

logger = get_logger("risk_engine")

_FALLBACK = {
    "max_position_pct": 0.10,
    "max_single_loss_pct": 0.01,
    "max_daily_loss_pct": 0.02,
    "max_drawdown_pct": 0.10,
    "no_leverage": True,
    "stop_after_n_losses": 3,
    "max_orders_per_hour": 20,
    "martingale_position_multiplier": 1.5,
    "max_session_minutes": 240,
}

# ============================================================
# 从配置文件加载风控参数
# ============================================================

_params: dict = {}


def _load_risk_params() -> dict:
    """从 config.loader 加载风控参数，缺失时使用保守兜底值。"""
    params = dict(_FALLBACK)
    cfg = get_config().get("risk", {})
    params.update(cfg)
    return params


def reload() -> dict:
    """重新加载风控参数（配置修改后无需重启）。"""
    global _params
    reload_config()
    _params = _load_risk_params()
    logger.info(f"风控参数已加载: {_params}")
    return _params


# ============================================================
# 风控参数访问
# ============================================================

def _ensure_loaded():
    """确保参数已加载（懒加载，第一次访问时才读文件）"""
    if not _params:
        reload()


def get_params() -> dict:
    """获取当前所有风控参数"""
    _ensure_loaded()
    return dict(_params)


# 为了向后兼容，保留模块级别的常量访问方式
# 但实际值从配置文件读取

def _get(key: str):
    _ensure_loaded()
    return _params.get(key, _FALLBACK.get(key))


# ============================================================
# 风控检查函数
# ============================================================

def check_position_limit(position_pct: float) -> tuple[bool, str]:
    """
    检查单仓位是否超限

    参数
    ----------
    position_pct : float
        当前仓位占总资金的百分比，如 0.15 表示 15%

    返回
    -------
    (是否通过, 说明文字)

    示例
    --------
    >>> ok, msg = check_position_limit(0.08)
    >>> print(ok, msg)  # True, "仓位 8.0% ≤ 上限 20.0%，通过"
    """
    limit = _get("max_position_pct")
    ok = position_pct <= limit
    if ok:
        msg = f"仓位 {position_pct * 100:.1f}% ≤ 上限 {limit * 100:.1f}%，通过"
    else:
        msg = f"仓位 {position_pct * 100:.1f}% > 上限 {limit * 100:.1f}%，超限！必须减仓"
        logger.warning(msg)
    return ok, msg


def check_single_loss(loss_pct: float) -> tuple[bool, str]:
    """
    检查单笔亏损是否触及止损

    示例
    --------
    >>> ok, msg = check_single_loss(0.015)
    >>> print(ok, msg)  # 标准值下: False, "亏损 1.5% > 上限 2.0%（标准），触发止损！"
    """
    limit = _get("max_single_loss_pct")
    ok = loss_pct <= limit
    if ok:
        msg = f"亏损 {loss_pct * 100:.2f}% ≤ 上限 {limit * 100:.1f}%，未触发止损"
    else:
        msg = f"亏损 {loss_pct * 100:.2f}% > 上限 {limit * 100:.1f}%，触发止损！必须卖出"
        logger.error(msg)
    return ok, msg


def check_daily_loss(daily_loss_pct: float) -> tuple[bool, str]:
    """
    检查当日累计亏损是否触及熔断

    示例
    --------
    >>> ok, msg = check_daily_loss(0.04)
    >>> print(ok, msg)  # 标准值下: False, "当日亏损 4.0% > 上限 5.0%（标准），熔断！"
    """
    limit = _get("max_daily_loss_pct")
    ok = daily_loss_pct <= limit
    if ok:
        msg = f"当日亏损 {daily_loss_pct * 100:.2f}% ≤ 上限 {limit * 100:.1f}%，未熔断"
    else:
        msg = f"当日亏损 {daily_loss_pct * 100:.2f}% > 上限 {limit * 100:.1f}%，触发熔断！停止当日所有交易"
        logger.error(msg)
    return ok, msg


def check_drawdown(drawdown_pct: float) -> tuple[bool, str]:
    """
    检查总回撤是否超限（超了策略永久停用）

    示例
    --------
    >>> ok, msg = check_drawdown(0.18)
    >>> print(ok, msg)  # 标准值下: False, "回撤 18.0% > 上限 15.0%，策略永久停用！"
    """
    limit = _get("max_drawdown_pct")
    ok = drawdown_pct <= limit
    if ok:
        msg = f"回撤 {drawdown_pct * 100:.1f}% ≤ 上限 {limit * 100:.1f}%，通过"
    else:
        msg = f"回撤 {drawdown_pct * 100:.1f}% > 上限 {limit * 100:.1f}%，策略永久停用！"
        logger.error(msg)
    return ok, msg


def check_consecutive_losses(count: int) -> tuple[bool, str]:
    """
    检查连续亏损笔数

    示例
    --------
    >>> ok, msg = check_consecutive_losses(6)
    >>> print(ok, msg)  # False, "连续亏损 6 笔 ≥ 上限 5，暂停交易！"
    """
    limit = _get("stop_after_n_losses")
    ok = count < limit
    if ok:
        msg = f"连续亏损 {count} 笔 < 上限 {limit}，继续交易"
    else:
        msg = f"连续亏损 {count} 笔 ≥ 上限 {limit}，自动暂停交易！"
        logger.error(msg)
    return ok, msg


def check_martingale(trade_history: list = None, position_pct: float = 0.0, prev_position_pct: float = 0.0) -> tuple[bool, str]:
    """
    检测马丁格尔模式：亏损后加倍仓位。

    当连续亏损且仓位扩大超过阈值时告警。
    """
    if trade_history is None:
        trade_history = []
    limit = _get("martingale_position_multiplier")
    # Check if position increased after a loss
    if len(trade_history) >= 2:
        last = trade_history[-1]
        prev = trade_history[-2]
        if last.get("pnl", 0) < 0 and prev.get("pnl", 0) < 0:
            if position_pct > 0 and prev_position_pct > 0:
                ratio = position_pct / prev_position_pct
                if ratio > limit:
                    msg = f"马丁格尔嫌疑: 连续亏损后仓位扩大 {ratio:.1f}x > 上限 {limit:.1f}x"
                    logger.error(msg)
                    return False, msg
    return True, "无马丁格尔嫌疑"


def check_order_frequency(count: int, time_minutes: int = 60) -> tuple[bool, str]:
    """
    检查报撤单频率是否超限（防止空哥事件：6万次报撤单400万申报费）。
    """
    limit = _get("max_orders_per_hour")
    hourly_rate = count / max(time_minutes, 1) * 60
    ok = hourly_rate <= limit
    if ok:
        return True, f"报单频率 {hourly_rate:.0f}/h ≤ 上限 {limit}/h"
    else:
        msg = f"报单频率 {hourly_rate:.0f}/h > 上限 {limit}/h！触发频率熔断"
        logger.error(msg)
        return False, msg


def check_vol_adaptive_position(
    recent_volatility: float,
    base_position_pct: float = 0.0,
    historical_volatility: float = None,
) -> tuple[bool, str, float]:
    """
    波动率自适应仓位：高波动自动缩仓、低波动扩仓。

    参数
    ----------
    recent_volatility : float
        近期波动率（如 20 日年化波动率，小数表示，0.25=25%）
    base_position_pct : float
        基础仓位比例（小数，0.20=20%）
    historical_volatility : float, optional
        历史参考波动率。未提供时使用 recent_volatility 自身。

    返回
    -------
    (是否通过, 说明文字, 调整后仓位比例)

    规则
    ----
    - 高波动（≥30%年化）→ 缩仓至 60%
    - 低波动（≤10%年化）→ 扩仓至 120%
    - 正常范围 → 保持基础仓位
    - 参数可配置（config/settings.yaml → risk → vol_adaptive_*）
    """
    _ensure_loaded()
    enabled = _params.get("vol_adaptive_position", True)
    if not enabled:
        return True, "波动率自适应仓位未启用", base_position_pct

    high_vol_threshold = _params.get("vol_high_threshold", 0.30)
    low_vol_threshold = _params.get("vol_low_threshold", 0.10)
    high_vol_scale = _params.get("vol_high_scale", 0.60)
    low_vol_scale = _params.get("vol_low_scale", 1.20)

    if historical_volatility is None:
        historical_volatility = recent_volatility

    adjusted = base_position_pct

    if recent_volatility >= high_vol_threshold:
        adjusted = base_position_pct * high_vol_scale
        msg = (
            f"高波动({recent_volatility:.1%} ≥ {high_vol_threshold:.0%}) → "
            f"仓位 {base_position_pct:.1%} → {adjusted:.1%} (缩至{high_vol_scale:.0%})"
        )
        logger.warning(msg)
        return True, msg, adjusted
    elif recent_volatility <= low_vol_threshold:
        adjusted = base_position_pct * low_vol_scale
        msg = (
            f"低波动({recent_volatility:.1%} ≤ {low_vol_threshold:.0%}) → "
            f"仓位 {base_position_pct:.1%} → {adjusted:.1%} (扩至{low_vol_scale:.0%})"
        )
        logger.info(msg)
        return True, msg, adjusted
    else:
        msg = (
            f"波动率正常({recent_volatility:.1%}) → "
            f"仓位保持 {base_position_pct:.1%}"
        )
        return True, msg, adjusted


def check_leverage(requested_leverage: float = 0.0) -> tuple[bool, str]:
    """
    检查是否使用杠杆（主动拦截，不只是被动配置）。
    """
    no_lev = _get("no_leverage")
    if no_lev and requested_leverage > 0:
        msg = f"杠杆已被禁止 (no_leverage=True)。请求杠杆 {requested_leverage}x 被拒绝。"
        logger.error(msg)
        return False, msg
    return True, "未使用杠杆"


def run_all_checks(
    position_pct: float = 0.0,
    single_loss_pct: float = 0.0,
    daily_loss_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    consecutive_losses: int = 0,
    order_count: int = 0,
    order_time_minutes: int = 60,
    requested_leverage: float = 0.0,
    trade_history: list = None,
    martingale_position_pct: float = 0.0,
    recent_volatility: float = None,
    historical_volatility: float = None,
) -> list[tuple[bool, str]]:
    """
    一次性跑完所有风控检查

    返回
    -------
    list of (是否通过, 说明文字)

    示例
    --------
    >>> results = run_all_checks(position_pct=0.05, single_loss_pct=0.03)
    >>> for ok, msg in results:
    ...     print(f"{'通过的' if ok else '未通过的'}: {msg}")
    """
    results = [
        check_position_limit(position_pct),
        check_single_loss(single_loss_pct),
        check_daily_loss(daily_loss_pct),
        check_drawdown(drawdown_pct),
        check_consecutive_losses(consecutive_losses),
        check_order_frequency(order_count, order_time_minutes),
        check_leverage(requested_leverage),
    ]
    if martingale_position_pct > 0:
        results.append(check_martingale(
            trade_history, position_pct, martingale_position_pct
        ))
    if recent_volatility is not None and position_pct > 0:
        ok, msg, _ = check_vol_adaptive_position(
            recent_volatility, position_pct, historical_volatility
        )
        results.append((ok, msg))
    return results


# ============================================================
# 命令行测试
# ============================================================
# python live/risk/risk_engine.py

if __name__ == "__main__":
    params = get_params()
    print("当前风控参数:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v * 100:.1f}%")
        else:
            print(f"  {k}: {v}")
    print()

    # 模拟一次风控检查
    print("模拟检查（单笔亏损 1.5%，连续亏损 3 笔）:")
    print("-" * 50)
    for ok, msg in run_all_checks(
        position_pct=0.05,
        single_loss_pct=0.015,
        daily_loss_pct=0.03,
        drawdown_pct=0.08,
        consecutive_losses=3,
    ):
        status = "通过的" if ok else "未通过的"
        print(f"  [{status}] {msg}")
