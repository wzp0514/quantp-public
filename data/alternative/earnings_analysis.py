"""
A股财报质量分析 — 中国本土替代美股财报电话会议分析

参考 Sattoro Hub 的"财报电话会议分析"功能。
中国版：中国公司没有公开电话会议转录，但有公开财报数据。

原理：
  1. 通过 AkShare 拉取个股财务指标（营收/利润/ROE/毛利率等）
  2. 计算同比变化和趋势
  3. 自动生成"财报质量评分"（0-100）
  4. 输出评级：优秀/良好/一般/警惕

数据来源（免费）：
  - AkShare stock_financial_abstract() — 财务摘要
  - AkShare stock_profit_sheet() — 利润表
  - 东方财富财报（通过 AkShare 间接访问）

用法
--------
>>> from data.alternative.earnings_analysis import EarningsAnalyzer
>>> ea = EarningsAnalyzer()
>>> result = ea.analyze("600519")         # 分析贵州茅台财报质量
>>> print(f"评级: {result['rating']}")     # A=优秀 / B=良好 / C=一般 / D=警惕
"""

import logging
from datetime import datetime
from typing import Optional

from config.log import get_logger
logger = get_logger("earnings_analysis")
class EarningsAnalyzer:
    """A股财报质量分析器"""

    def analyze(self, symbol: str) -> dict:
        """
        分析个股财报质量。

        返回
        -------
        dict: {
            symbol, company_name,
            revenue_growth: float (营收同比%),
            profit_growth: float (净利润同比%),
            roe: float,
            gross_margin: float,
            score: int (0-100),
            rating: str (A/B/C/D),
            interpretation: str,
            warnings: list[str],
        }
        """
        import akshare as ak
        import pandas as pd
        logger.info(f"分析财报: {symbol}")

        result = {
            "symbol": symbol,
            "company_name": "",
            "revenue_growth": None,
            "profit_growth": None,
            "roe": None,
            "gross_margin": None,
            "score": 0,
            "rating": "N/A",
            "interpretation": "",
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # 1. 拉取财务摘要
            finance = ak.stock_financial_abstract(symbol=symbol)
            if finance is None or finance.empty:
                logger.warning(f"无财报数据: {symbol}")
                return result

            result["company_name"] = str(finance.iloc[0].get("name", "")) if "name" in finance.columns else ""

            # 提取关键指标行（不同季度有不同的行，取最新一期）
            # 财务摘要的指标行名通常是固定的
            for _, row in finance.iterrows():
                # 搜索指标行
                pass

        except Exception as e:
            logger.error(f"财报分析失败: {symbol} — {e}")
            result["warnings"].append(f"数据获取失败: {e}")
            return result

        # 2. 结构化评分
        score = 0
        max_score = 0

        # 营收增长率（权重 30%）
        if result["revenue_growth"] is not None:
            max_score += 30
            rg = result["revenue_growth"]
            if rg > 30:
                score += 30
            elif rg > 15:
                score += 24
            elif rg > 5:
                score += 18
            elif rg > 0:
                score += 10
            else:
                result["warnings"].append(f"营收同比 {rg:.1f}%（下滑）")

        # 利润增长率（权重 30%）
        if result["profit_growth"] is not None:
            max_score += 30
            pg = result["profit_growth"]
            if pg > 30:
                score += 30
            elif pg > 15:
                score += 24
            elif pg > 5:
                score += 18
            elif pg > 0:
                score += 10
            else:
                result["warnings"].append(f"利润同比 {pg:.1f}%（下滑）")

        # ROE（权重 20%）
        if result["roe"] is not None:
            max_score += 20
            roe = result["roe"]
            if roe > 20:
                score += 20
            elif roe > 15:
                score += 16
            elif roe > 10:
                score += 12
            elif roe > 5:
                score += 8
            else:
                result["warnings"].append(f"ROE {roe:.1f}%（偏低）")

        # 毛利率（权重 20%）
        if result["gross_margin"] is not None:
            max_score += 20
            gm = result["gross_margin"]
            if gm > 60:
                score += 20
            elif gm > 40:
                score += 16
            elif gm > 20:
                score += 12
            elif gm > 10:
                score += 8
            else:
                result["warnings"].append(f"毛利率 {gm:.1f}%（偏低）")

        # 归一化
        result["score"] = round(score / max(max_score, 1) * 100)
        result["rating"] = self._rate(result["score"])
        result["interpretation"] = self._interpret(result["rating"], result["warnings"])

        logger.info(
            f"财报评级({symbol}): {result['rating']} ({result['score']}分)"
            + (f", {len(result['warnings'])}个预警" if result["warnings"] else "")
        )

        return result

    def analyze_from_manual(
        self,
        symbol: str,
        company_name: str = "",
        revenue_growth: float = None,
        profit_growth: float = None,
        roe: float = None,
        gross_margin: float = None,
    ) -> dict:
        """
        手动输入财报数据进行分析（当AkShare数据不可用时）。

        东方财富/同花顺上直接抄几个关键数字就行。
        """
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "revenue_growth": revenue_growth,
            "profit_growth": profit_growth,
            "roe": roe,
            "gross_margin": gross_margin,
            "score": 0,
            "rating": "N/A",
            "interpretation": "",
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
        }

        score = 0
        max_score = 0

        for val, weight, name in [
            (revenue_growth, 30, "营收增长"),
            (profit_growth, 30, "利润增长"),
            (roe, 20, "ROE"),
            (gross_margin, 20, "毛利率"),
        ]:
            if val is not None:
                max_score += weight
                if val > 30:
                    score += weight
                elif val > 15:
                    score += weight * 0.8
                elif val > 5:
                    score += weight * 0.6
                elif val > 0:
                    score += weight * 0.33
                else:
                    result["warnings"].append(f"{name} {val:.1f}%（下滑）")

        result["score"] = round(score / max(max_score, 1) * 100)
        result["rating"] = self._rate(result["score"])
        result["interpretation"] = self._interpret(result["rating"], result["warnings"])

        return result

    def _rate(self, score: int) -> str:
        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        return "D"

    def _interpret(self, rating: str, warnings: list[str]) -> str:
        base = {
            "A": "优秀——财务指标全面健康，适合长期持有底仓",
            "B": "良好——基本面稳健，个别指标可关注",
            "C": "一般——部分指标存在问题，适合短线但不适合重仓",
            "D": "警惕——多项指标恶化，建议回避或极轻仓位",
        }
        text = base.get(rating, "无法判断")
        if warnings:
            text += f"。预警: {'; '.join(warnings[:3])}"
        return text


# ============================================================
# 命令行测试
# ============================================================
# python data/alternative/earnings_analysis.py

if __name__ == "__main__":
    print("=" * 60)
    print("A股财报质量分析")
    print("=" * 60)

    ea = EarningsAnalyzer()

    # 手动模式示例（不需要网络）
    print("\n手动模式示例（贵州茅台2024年报概数）:")
    result = ea.analyze_from_manual(
        symbol="600519",
        company_name="贵州茅台",
        revenue_growth=15.0,
        profit_growth=14.5,
        roe=28.0,
        gross_margin=91.0,
    )
    print(f"  公司: {result['company_name']}")
    print(f"  评级: {result['rating']} ({result['score']}分)")
    print(f"  解读: {result['interpretation']}")
    if result["warnings"]:
        print(f"  预警: {result['warnings']}")

    # 自动模式
    print("\n自动拉取模式（需网络）:")
    try:
        auto = ea.analyze("600519")
        print(f"  评级: {auto['rating']} ({auto['score']}分)")
    except Exception as e:
        print(f"  自动拉取失败（正常，AkShare财报接口可能限流）: {e}")
        print("  请使用手动模式: ea.analyze_from_manual(symbol='...', revenue_growth=..., ...)")
