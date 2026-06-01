"""
地缘政治监控 — 参考 Sattoro Hub 地缘政治监控

免费方案：国际新闻 RSS + 地缘关键词过滤 → 风险等级 → 预警通知。

原理：
  1. 从国际新闻 RSS 源拉取标题
  2. 用地缘关键词库匹配（战争/制裁/关税/冲突等）
  3. 统计匹配数量 → 风险评分（0-100）
  4. 超过阈值 → 触发预警（可在策略中减仓或暂停交易）

用法
--------
>>> from data.alternative.geopolitical import GeopoliticalMonitor
>>> gm = GeopoliticalMonitor()
>>> result = gm.scan()                    # 扫描当前地缘风险
>>> print(f"风险等级: {result['level']}")  # low/medium/high/critical
>>> if result['alerts']:
...     print("需要关注的预警:")
...     for a in result['alerts']:
...         print(f"  - {a}")
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional
import urllib.request
import xml.etree.ElementTree as ET

from config.log import get_logger

logger = get_logger("geopolitical")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# ============================================================
# 国际新闻 RSS 源
# ============================================================

RSS_SOURCES = {
    "Reuters World": "https://rsshub.app/reuters/world",
    "BBC World": "https://rsshub.app/bbc/world",
    "新华社国际": "https://rsshub.app/xinhua/news/world",
}

# ============================================================
# 地缘政治关键词库（分等级）
# ============================================================

# 严重关键词（每条 +4 分）— 直接影响市场，需要立即关注
CRITICAL_KEYWORDS = [
    "宣战", "军事冲突", "导弹", "空袭", "核武器",
    "全面制裁", "金融制裁", "资产冻结", "swift",
    "政权更迭", "军事政变", "紧急状态",
    "war", "invasion", "missile", "nuclear", "military coup",
    "sanctions imposed", "asset freeze",
]

# 重大关键词（每条 +2 分）— 可能影响市场，需要关注
MAJOR_KEYWORDS = [
    "关税", "贸易战", "供应链中断", "能源危机",
    "地缘", "台海", "南海", "朝鲜", "俄罗斯", "北约",
    "制裁", "出口管制", "芯片禁令",
    "tariff", "trade war", "geopolitical", "taiwan strait",
    "south china sea", "north korea", "nato", "sanction",
    "export control", "chip ban",
]

# 注意关键词（每条 +1 分）— 值得留意
MINOR_KEYWORDS = [
    "大选", "选举", "公投", "脱欧", "抗议", "示威",
    "外交", "谈判破裂", "召回大使",
    "election", "referendum", "brexit", "protest",
    "diplomatic", "ambassador",
]


def _fetch_titles(url: str, timeout: int = 15) -> list[str]:
    """从 RSS 获取标题"""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        root = ET.fromstring(content)

        titles = []
        for item in root.iter("item"):
            t = item.find("title")
            if t is not None and t.text:
                titles.append(t.text.strip())
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            t = entry.find("{http://www.w3.org/2005/Atom}title")
            if t is not None and t.text:
                titles.append(t.text.strip())
        return titles
    except Exception:
        return []


def _score_geopolitical(titles: list[str]) -> dict:
    """对标题列表进行地缘风险评分"""
    score = 0
    alerts = []

    for title in titles:
        title_lower = title.lower()

        for kw in CRITICAL_KEYWORDS:
            if kw.lower() in title_lower:
                score += 4
                alerts.append(f"[严重] {title}")
                break

        for kw in MAJOR_KEYWORDS:
            if kw.lower() in title_lower:
                score += 2
                if not any(title in a for a in alerts):
                    alerts.append(f"[重大] {title}")
                break

        for kw in MINOR_KEYWORDS:
            if kw.lower() in title_lower:
                score += 1
                break

    return {"score": score, "alerts": alerts}


def _risk_level(score: int) -> str:
    if score >= 20:
        return "critical"    # 极高风险——建议暂停所有交易
    elif score >= 10:
        return "high"         # 高风险——建议减仓
    elif score >= 5:
        return "medium"       # 中等风险——正常交易但密切监控
    else:
        return "low"          # 低风险——正常交易


class GeopoliticalMonitor:
    """地缘政治风险监控器"""

    def __init__(self):
        self.last_scan: Optional[dict] = None

    def scan(self, sources: list[str] = None) -> dict:
        """
        扫描当前地缘政治风险。

        返回
        -------
        dict: {score, level, alerts, interpretation, timestamp}
        """
        src_names = sources or list(RSS_SOURCES.keys())
        all_titles = []

        for name in src_names:
            url = RSS_SOURCES.get(name)
            if not url:
                continue
            titles = _fetch_titles(url)
            all_titles.extend(titles)
            time.sleep(0.5)

        result = _score_geopolitical(all_titles)
        level = _risk_level(result["score"])

        interpretations = {
            "low": "低风险——地缘政治环境平静，正常交易",
            "medium": "中等风险——存在地缘事件，注意仓位控制",
            "high": "高风险——重大地缘事件，建议减仓至 50% 以下",
            "critical": "极高风险——严重地缘冲突，建议暂停所有交易",
        }

        # 置信度：有数据起 0.3，按文章数加至 1.0
        has_data = len(all_titles) > 0
        confidence = round(0.3 + min(1.0, len(all_titles) / 10) * 0.7, 2) if has_data else 0.1

        output = {
            "score": result["score"],
            "level": level,
            "alerts": result["alerts"][:10],
            "interpretation": interpretations[level],
            "articles_scanned": len(all_titles),
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
        }

        self.last_scan = output

        if result["alerts"]:
            logger.warning(f"地缘风险: {level} (评分 {result['score']})")
            for a in result["alerts"][:3]:
                logger.warning(f"  {a}")

        return output

    def should_reduce_position(self) -> tuple[bool, float, str]:
        """
        判断是否应减仓。

        返回 (是否减仓, 建议仓位比例, 原因)
        """
        if self.last_scan is None:
            self.scan()

        level = self.last_scan["level"]
        if level == "critical":
            return True, 0.0, "极高地缘风险，建议空仓"
        elif level == "high":
            return True, 0.5, "高地缘风险，建议减仓至 50%"
        elif level == "medium":
            return False, 0.8, "中等地缘风险，可满仓但警惕"
        else:
            return False, 1.0, "低地缘风险，正常交易"


# ============================================================
# 命令行测试
# ============================================================
# python data/alternative/geopolitical.py

if __name__ == "__main__":
    print("=" * 60)
    print("地缘政治风险监控")
    print("=" * 60)

    gm = GeopoliticalMonitor()
    result = gm.scan()

    print(f"\n风险评分: {result['score']}/100")
    print(f"风险等级: {result['level']}")
    print(f"解读: {result['interpretation']}")
    print(f"扫描文章: {result['articles_scanned']}")

    if result["alerts"]:
        print(f"\n预警 ({len(result['alerts'])} 条):")
        for a in result["alerts"]:
            print(f"  {a}")

    reduce, ratio, reason = gm.should_reduce_position()
    print(f"\n仓位建议: {'减仓' if reduce else '不减仓'} → {ratio:.0%} — {reason}")
