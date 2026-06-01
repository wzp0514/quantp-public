"""
A股交易约束 — 回测精度最关键的两个要素

如果不建模以下约束，回测收益会虚高，实盘根本执行不了：

1. T+1 制度
   - 今天买入的股票，明天（下一个交易日）才能卖出
   - 回测中如果忽略了 → 当日低买高卖被抓到，但实盘做不到
   处理：记录买入日期，只有在次日及之后才允许卖出

2. 涨跌停板（Price Limit）
   - 主板（60xxxx/00xxxx）：±10%
   - 创业板（30xxxx）：±20%
   - 科创板（688xxx）：±20%
   - 北交所（8xxxxx）：±30%
   - ST股票（含*ST）：±5%
   处理：价格触及涨停时不能买入（买不到），跌停时不能卖出（卖不掉）

3. ST股票（Special Treatment）
   - 有退市风险的股票，涨跌幅限制更严（5%）
   - 机构投资者通常不允许持有ST股
   处理：策略可配置是否允许交易ST股

用法
--------
>>> from data.cleaner.a_share_constraints import AShareConstraints
>>> asc = AShareConstraints()
>>> asc.can_buy(code="600519", price=2000.0, prev_close=1800.0)  # 涨停？→ False
>>> asc.can_sell(buy_date=date(2025,1,6), today=date(2025,1,6))  # T+1？→ False
"""

import logging
from datetime import date, timedelta
from typing import Optional

from config.log import get_logger
logger = get_logger("a_share_constraints")
def get_price_limit_pct(code: str) -> float:
    """
    根据股票代码返回涨跌停幅度。

    规则：
      300xxx/301xxx → 创业板 ±20%
      688xxx        → 科创板 ±20%
      8xxxxx        → 北交所 ±30%
      ST/*ST         → ±5%
      其他(60xxxx/00xxxx) → 主板 ±10%
    """
    code = str(code).strip()
    if code.startswith(("300", "301")):
        return 0.20
    elif code.startswith("688"):
        return 0.20
    elif code.startswith("8"):
        return 0.30
    else:
        return 0.10


def is_st_stock(code_or_name: str) -> bool:
    """判断是否为 ST 股票"""
    return "ST" in str(code_or_name).upper()


def limit_up_price(prev_close: float, limit_pct: float = 0.10) -> float:
    """涨停价"""
    return round(prev_close * (1 + limit_pct), 2)


def limit_down_price(prev_close: float, limit_pct: float = 0.10) -> float:
    """跌停价"""
    return round(prev_close * (1 - limit_pct), 2)


class AShareConstraints:
    """
    A股交易约束检查器。

    应该在每次买入/卖出前调用，不满足约束时拒绝交易。
    """

    def __init__(
        self,
        enable_t1: bool = True,
        enable_price_limit: bool = True,
        allow_st: bool = False,
        log_rejections: bool = True,
        asset_type: str = "stock",
    ):
        self.enable_t1 = enable_t1
        self.enable_price_limit = enable_price_limit
        self.allow_st = allow_st
        self.log_rejections = log_rejections
        self.asset_type = asset_type

    @staticmethod
    def for_asset_type(asset_type: str = "stock") -> "AShareConstraints":
        """从配置读取品种参数，创建约束检查器"""
        try:
            from config.loader import get_config
            cfg = get_config().get("asset_types", {}).get(asset_type, {})
        except Exception:
            cfg = {}
        t_plus = cfg.get("t_plus", 1)
        price_limit = cfg.get("price_limit", 0.10)
        return AShareConstraints(
            enable_t1=(t_plus > 0),
            enable_price_limit=(price_limit > 0),
            asset_type=asset_type,
        )

    # ── 品种专用 ──

    def _get_price_limit(self, code: str) -> float:
        """根据品种和代码获取涨跌停幅度"""
        if self.asset_type == "convertible_bond":
            return 0.0  # 无涨跌停
        if self.asset_type == "etf":
            code_str = str(code)
            if code_str.startswith(("159", "588")):
                return 0.20  # 科创/创业板ETF ±20%
            return 0.10  # 主板ETF ±10%
        # stock: 按代码判断
        return 0.05 if is_st_stock(code) else get_price_limit_pct(code)

    # ============================================================
    # 买入检查
    # ============================================================

    def can_buy(
        self,
        code: str,
        price: float,
        prev_close: float,
        today: date = None,
    ) -> tuple[bool, str]:
        """
        检查是否可以买入。

        返回 (是否允许, 原因)。

        拒绝原因可能包括：
          - 涨停（买不到）
          - ST股票且配置为不允许
        """
        # 1. ST 检查
        if not self.allow_st and is_st_stock(code):
            msg = f"ST股票不允许买入: {code}"
            if self.log_rejections:
                logger.info(msg)
            return False, msg

        # 2. 涨停检查
        if self.enable_price_limit and prev_close > 0:
            limit_pct = self._get_price_limit(code)
            if limit_pct > 0:
                upper = limit_up_price(prev_close, limit_pct)
                if price >= upper * 0.995:  # 接近涨停价（千分之五内）就拒绝
                    msg = f"涨停拒绝买入: {code} @ {price:.2f} ≥ 涨停价 {upper:.2f}"
                    if self.log_rejections:
                        logger.info(msg)
                    return False, msg

        return True, ""

    # ============================================================
    # 卖出检查
    # ============================================================

    def can_sell(
        self,
        code: str,
        price: float,
        prev_close: float,
        buy_date: date,
        today: date,
    ) -> tuple[bool, str]:
        """
        检查是否可以卖出。

        拒绝原因可能包括：
          - T+1（今天买的今天不能卖）
          - 跌停（卖不掉）
        """
        # 1. T+1 检查
        if self.enable_t1:
            if today <= buy_date:
                msg = f"T+1拒绝卖出: {code} 买入于 {buy_date}, 今日 {today} 不可卖出"
                if self.log_rejections:
                    logger.info(msg)
                return False, msg

        # 2. 跌停检查
        if self.enable_price_limit and prev_close > 0:
            limit_pct = self._get_price_limit(code)
            if limit_pct <= 0:
                pass
            else:
                lower = limit_down_price(prev_close, limit_pct)
                if price <= lower * 1.005:
                    msg = f"跌停拒绝卖出: {code} @ {price:.2f} <= 跌停价 {lower:.2f}"
                    if self.log_rejections:
                        logger.info(msg)
                    return False, msg

        return True, ""

    # ============================================================
    # 综合 — 一次检查所有约束
    # ============================================================

    def check_buy(
        self,
        code: str,
        price: float,
        prev_close: float,
        today: date = None,
    ) -> tuple[bool, str]:
        """买入前检查所有A股约束（等同于 can_buy）"""
        return self.can_buy(code, price, prev_close, today)

    def check_sell(
        self,
        code: str,
        price: float,
        prev_close: float,
        buy_date: date,
        today: date,
    ) -> tuple[bool, str]:
        """卖出前检查所有A股约束"""
        return self.can_sell(code, price, prev_close, buy_date, today)

    # ============================================================
    # 便捷 — 用行情数据行来检查
    # ============================================================

    def check_buy_row(self, code: str, row) -> tuple[bool, str]:
        """
        用 DataFrame 的一行数据检查买入。

        row 需要包含: close/open/pre_close（任一），date 列可选
        """
        price = row.get("close") or row.get("open", 0)
        prev_close = row.get("pre_close", row.get("open", price))
        today = row.get("date")
        if hasattr(today, "date"):
            today = today.date()
        return self.check_buy(str(code), float(price), float(prev_close), today)


# ============================================================
# 命令行测试
# ============================================================
# python data/cleaner/a_share_constraints.py

if __name__ == "__main__":
    asc = AShareConstraints()

    print("=" * 60)
    print("A股约束测试")
    print("=" * 60)

    # 测试涨停
    print("\n1. 涨停买入被拒:")
    ok, msg = asc.can_buy("600519", price=110.0, prev_close=100.0)
    print(f"   600519 @ 110.0 (昨收100, 涨停110): {ok} — {msg}")

    # 测试跌停
    print("\n2. 跌停卖出被拒:")
    ok, msg = asc.can_sell("600519", 90.0, 100.0,
                           buy_date=date(2025, 1, 5), today=date(2025, 1, 6))
    print(f"   600519 @ 90.0 (昨收100, 跌停90): {ok} — {msg}")

    # 测试T+1
    print("\n3. T+1卖出被拒:")
    ok, msg = asc.can_sell("000001", 15.0, 14.5,
                           buy_date=date(2025, 1, 6), today=date(2025, 1, 6))
    print(f"   000001 今天买今天卖: {ok} — {msg}")

    # 测试正常
    print("\n4. 正常交易:")
    ok, msg = asc.can_buy("600519", price=105.0, prev_close=100.0)
    print(f"   600519 @ 105.0: {ok}")
    ok, msg = asc.can_sell("600519", 105.0, 100.0,
                           buy_date=date(2025, 1, 5), today=date(2025, 1, 8))
    print(f"   600519 1月5买 1月8卖: {ok}")

    # 创业板20%涨停
    print("\n5. 创业板±20%:")
    print(f"   300xxx 涨跌停幅度: {get_price_limit_pct('300750'):.0%}")
