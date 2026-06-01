"""
研究源头 — 管线外部新鲜输入引擎

为量化CI/CD管线提供6大外部信号源:
  1. arXiv论文扫描 — 新因子/新方法
  2. 数据新鲜度检测 — 新标的/新数据源
  3. 市场结构变化 — 区制切换/波动率突变
  4. 另类数据触发 — 情绪极端/地缘升级/财报预警
  5. 策略健康反馈 — Guard熔断/因子衰减
  6. 日历事件触发 — 财报季/调仓日/政策节点

噪声控制: 置信度门槛 + FWER校正 + 冷静期 + 最小验证
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("research_source")


# ============================================================
# 核心数据类
# ============================================================

@dataclass
class ResearchSignal:
    """一条可执行的研究信号"""
    source: str
    title: str
    action: str = "alert"          # factor/strategy/data/alert
    priority: int = 1              # 1-5, 5最高
    data: dict = field(default_factory=dict)
    confidence: float = 0.5
    suggested_track: str = ""
    needs_review: bool = False
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.priority >= 5:
            self.needs_review = True

    def to_dict(self) -> dict:
        return {
            "source": self.source, "title": self.title,
            "action": self.action, "priority": self.priority,
            "data": self.data, "confidence": self.confidence,
            "suggested_track": self.suggested_track,
            "needs_review": self.needs_review,
            "created_at": self.created_at,
        }


# ============================================================
# 噪声控制工具
# ============================================================

def _fwer_filter(signals: list[ResearchSignal], alpha: float = 0.05) -> list[ResearchSignal]:
    """FDR校正（Benjamini-Hochberg）：同一来源N个信号 → 控制错误发现率"""
    from collections import defaultdict
    by_source: dict[str, list] = defaultdict(list)
    for s in signals:
        by_source[s.source].append(s)
    filtered = []
    for src, group in by_source.items():
        n = len(group)
        if n <= 1:
            filtered.extend(group)
            continue
        # BH: 按置信度降序排列，找最大k使得 c_k >= 1 - (k/N)*alpha
        sorted_group = sorted(group, key=lambda s: s.confidence, reverse=True)
        k = 0
        for i, s in enumerate(sorted_group, 1):
            bh_threshold = 1 - (i / n) * alpha
            if s.confidence >= bh_threshold:
                k = i
            else:
                break
        for i, s in enumerate(sorted_group):
            if i < k:
                filtered.append(s)
            else:
                logger.debug(f"FDR过滤: [{s.source}] {s.title} (confidence={s.confidence:.2f} < BH阈值)")
    return filtered


def _cooldown_check(signals: list[ResearchSignal],
                    history: list[dict], cooldown_hours: int = 24) -> list[ResearchSignal]:
    """冷静期检查：同源+同类信号至少间隔cooldown_hours"""
    cutoff = datetime.now() - timedelta(hours=cooldown_hours)
    recent_keys: set[tuple] = set()
    for h in history:
        ts = h.get("created_at", "")
        if ts:
            try:
                if datetime.fromisoformat(ts) > cutoff:
                    recent_keys.add((h.get("source", ""), h.get("action", "")))
            except ValueError:
                pass
    filtered = []
    for s in signals:
        if (s.source, s.action) in recent_keys:
            logger.debug(f"冷静期跳过: [{s.source}] {s.action}")
            continue
        filtered.append(s)
    return filtered


def _confidence_gate(signals: list[ResearchSignal], min_conf: float = 0.3) -> list[ResearchSignal]:
    return [s for s in signals if s.confidence >= min_conf]


# ============================================================
# ResearchSource 主类
# ============================================================

class ResearchSource:
    """研究源头引擎。扫描外部变化，产出可执行的ResearchSignal。"""

    def __init__(self, df: pd.DataFrame = None, cash: float = 100000.0):
        self.df = df
        self.cash = cash
        self._store = None
        self._history: list[dict] = []

    def _load_history(self, days: int = 7):
        try:
            from data.vault.research_store import ResearchStore
            self._store = ResearchStore()
            self._history = self._store.query(days=days)
        except Exception:
            logger.exception("Failed to load research history from ResearchStore")
            self._history = []

    # ============================================================
    # 源头1: arXiv论文扫描
    # ============================================================

    def scan_arxiv_papers(self, max_results: int = 5) -> list[ResearchSignal]:
        """扫描arXiv q-fin最新论文，提取可复现的因子/策略思路。"""
        signals = []
        url = (f"http://export.arxiv.org/api/query?"
               f"search_query=cat:q-fin.PM+OR+cat:q-fin.ST+OR+cat:q-fin.TR"
               f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantP/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as e:
            logger.warning(f"arXiv API不可达: {e}")
            signals.append(ResearchSignal(
                source="arxiv", title="arXiv API无法连接",
                action="alert", priority=1,
                confidence=0.9, data={"error": str(e)},
            ))
            return signals

        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return signals

        entries = root.findall("atom:entry", ns)
        keyword_map = {
            "factor": ["factor", "alpha", "signal", "predictor", "factor model", "anomaly"],
            "strategy": ["strategy", "trading", "portfolio", "allocation", "momentum", "reversal"],
            "method": ["machine learning", "deep learning", "transformer", "LSTM", "gradient boosting",
                       "reinforcement learning", "bayesian", "causality"],
        }
        for entry in entries:
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            link_el = entry.find("atom:id", ns)
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            if not title:
                continue
            text = (title + " " + summary).lower()
            matched = []
            for cat, kws in keyword_map.items():
                if any(kw in text for kw in kws):
                    matched.append(cat)
            if not matched:
                continue
            action = "factor" if "factor" in matched else ("strategy" if "strategy" in matched else "alert")
            confidence = min(0.7, 0.4 + 0.1 * len(matched))
            signals.append(ResearchSignal(
                source="arxiv", title=f"[论文] {title[:80]}",
                action=action, priority=3 if "factor" in matched else 2,
                confidence=confidence,
                data={"title": title, "link": link, "categories": matched},
                suggested_track=f"arxiv_{title[:20].replace(' ', '_')}",
            ))
        logger.info(f"arXiv扫描: {len(entries)}篇 → {len(signals)}条信号")
        return signals

    # ============================================================
    # 源头2: 数据新鲜度检测
    # ============================================================

    def check_data_freshness(self) -> list[ResearchSignal]:
        """检测数据源健康状态 + 新数据可用性。"""
        signals = []
        try:
            from data.fetchers.fallback import check_dependencies, get_source_health
            deps = check_dependencies()
            for name, ok in deps.items():
                if not ok:
                    signals.append(ResearchSignal(
                        source="data_freshness", title=f"数据源不可用: {name}",
                        action="alert", priority=4, confidence=0.95,
                        data={"source": name, "status": "down"},
                    ))
            health = get_source_health()
            for name, h in health.items():
                if h.get("status") == "down":
                    signals.append(ResearchSignal(
                        source="data_freshness", title=f"数据源降级中: {name}",
                        action="alert", priority=3, confidence=0.9,
                        data={"source": name, "error": h.get("error", "")},
                    ))
        except Exception as e:
            logger.debug(f"数据源健康检查跳过: {e}")

        if self.df is not None and len(self.df) > 0:
            last_date = pd.to_datetime(self.df["date"].max())
            days_behind = (datetime.now() - last_date).days
            if days_behind > 3:
                signals.append(ResearchSignal(
                    source="data_freshness", title=f"数据滞后{days_behind}天 (最新: {last_date.date()})",
                    action="data", priority=4 if days_behind > 7 else 3,
                    confidence=0.95,
                    data={"last_date": str(last_date.date()), "days_behind": days_behind},
                ))
            elif days_behind <= 1:
                signals.append(ResearchSignal(
                    source="data_freshness", title=f"数据新鲜 (截至{last_date.date()})",
                    action="data", priority=1, confidence=0.95,
                    data={"last_date": str(last_date.date()), "days_behind": days_behind},
                ))

        try:
            from data.vault.market_data import MarketVault
            mv = MarketVault()
            info = mv.info()
            new_stocks = info.get("new_since_last_scan", 0)
            if new_stocks > 0:
                signals.append(ResearchSignal(
                    source="data_freshness", title=f"新增{new_stocks}只股票可回测",
                    action="strategy", priority=4,
                    confidence=0.8,
                    suggested_track=f"new_stocks_{new_stocks}",
                ))
        except Exception:
            logger.exception("MarketVault new_stocks check failed")

        return signals

    # ============================================================
    # 源头3: 市场结构变化检测
    # ============================================================

    def detect_regime_change(self) -> list[ResearchSignal]:
        """检测波动率突变/相关性破裂/趋势结构断点。"""
        signals = []
        if self.df is None or len(self.df) < 60:
            return signals
        close = self.df["close"].values
        returns = np.diff(np.log(close))
        n = len(returns)

        vol_20 = pd.Series(returns).rolling(20).std().values
        if len(vol_20) > 60:
            recent_vol = np.nanmean(vol_20[-20:])
            hist_vol = np.nanmean(vol_20[-60:-20])
            if hist_vol > 0 and recent_vol / hist_vol > 2.0:
                signals.append(ResearchSignal(
                    source="regime_change", title=f"波动率飙升 (近期{recent_vol:.4f} vs 历史{hist_vol:.4f})",
                    action="alert", priority=4, confidence=0.85,
                    data={"recent_vol": float(recent_vol), "hist_vol": float(hist_vol),
                           "ratio": float(recent_vol / hist_vol)},
                ))
            elif hist_vol > 0 and recent_vol / hist_vol < 0.5:
                signals.append(ResearchSignal(
                    source="regime_change", title=f"波动率骤降 (近期{recent_vol:.4f} vs 历史{hist_vol:.4f})",
                    action="alert", priority=3, confidence=0.8,
                    data={"recent_vol": float(recent_vol), "hist_vol": float(hist_vol),
                           "ratio": float(recent_vol / hist_vol)},
                ))

        if n > 126:
            half = n // 2
            ret_first = returns[:half]
            ret_second = returns[half:]
            if len(ret_first) > 20 and len(ret_second) > 20:
                mean1, mean2 = np.mean(ret_first), np.mean(ret_second)
                std1, std2 = np.std(ret_first), np.std(ret_second)
                mean_change = abs(mean2 - mean1) / max(std1, std2, 1e-10) * np.sqrt(126)
                if mean_change > 1.5:
                    direction = "上升" if mean2 > mean1 else "下降"
                    signals.append(ResearchSignal(
                        source="regime_change", title=f"收益结构{direction} (前后半段均值显著偏离 {mean_change:.1f}σ)",
                        action="strategy", priority=4, confidence=min(0.95, mean_change / 3 + 0.5),
                        data={"mean_first": float(mean1), "mean_second": float(mean2),
                               "sigma": float(mean_change)},
                        suggested_track="regime_adapt",
                    ))

        vol_window = min(20, n // 5) if n >= 20 else n
        if n > vol_window * 3:
            vol_series = pd.Series(returns).rolling(vol_window).std().values[vol_window:]
            if len(vol_series) > vol_window:
                high_vol = np.sum(vol_series > np.percentile(vol_series, 90))
                if high_vol > len(vol_series) * 0.15:
                    signals.append(ResearchSignal(
                        source="regime_change", title=f"高波动异常频繁 (>{len(vol_series)*0.15:.0f}次极端波动)",
                        action="alert", priority=3, confidence=0.75,
                    ))

        return signals

    # ============================================================
    # 源头4: 另类数据触发
    # ============================================================

    def scan_alternative_data(self) -> list[ResearchSignal]:
        """运行另类数据全量扫描，极端信号触发告警。"""
        signals = []
        try:
            from data.alternative import AlternativeData
            ad = AlternativeData()
            result = ad.full_scan()
            risk = result.get("risk_score", 0)
            risk_level = result.get("risk_level", "低")
            if risk >= 50:
                signals.append(ResearchSignal(
                    source="alternative", title=f"综合风险{risk_level}({risk}分) — 建议减仓或暂停新策略",
                    action="alert", priority=4 if risk >= 70 else 3,
                    confidence=0.8,
                    data={"risk_score": risk, "risk_level": risk_level,
                           "geo_level": result.get("geopolitical_level", ""),
                           "sentiment": result.get("sentiment_score", 0)},
                ))
            geo_level = result.get("geopolitical_level", "low")
            if geo_level in ("critical", "high"):
                signals.append(ResearchSignal(
                    source="alternative", title=f"地缘风险{geo_level}级 — 建议暂停新仓",
                    action="alert", priority=5, confidence=0.85,
                    data=result.get("geopolitical_detail", {}),
                    needs_review=True,
                ))
            should_trade = result.get("should_trade", True)
            if not should_trade:
                signals.append(ResearchSignal(
                    source="alternative", title="另类数据综合判断: 不建议交易",
                    action="alert", priority=4, confidence=0.8,
                ))
            mult = result.get("position_multiplier", 1.0)
            if mult < 0.5:
                signals.append(ResearchSignal(
                    source="alternative", title=f"建议仓位乘数仅{mult:.0%} — 风险极高",
                    action="alert", priority=5, confidence=0.85,
                    needs_review=True,
                ))
        except Exception as e:
            logger.exception(f"另类数据扫描失败: {e}")
        return signals

    # ============================================================
    # 源头5: 策略健康反馈
    # ============================================================

    def check_strategy_health(self) -> list[ResearchSignal]:
        """从BacktestStore查询策略退化 + FactorStore查因子衰减。"""
        signals = []
        try:
            from data.vault.backtest_store import BacktestStore
            bs = BacktestStore()
            evolution = bs.evolution(metric="sharpe", days=180)
            if evolution is not None and len(evolution) > 2:
                recent_sr = evolution["sharpe"].iloc[-1]
                peak_sr = evolution["sharpe"].max()
                if peak_sr > 0.3 and recent_sr / max(peak_sr, 0.01) < 0.5:
                    signals.append(ResearchSignal(
                        source="strategy_health", title=f"策略夏普退化>50% (峰值{peak_sr:.2f}→当前{recent_sr:.2f})",
                        action="strategy", priority=4, confidence=0.85,
                        data={"peak_sr": float(peak_sr), "recent_sr": float(recent_sr)},
                        suggested_track="strategy_refresh",
                    ))
        except Exception as e:
            logger.exception(f"BacktestStore健康检查失败: {e}")

        try:
            from data.vault.factor_store import FactorStore
            fs = FactorStore()
            factors = fs.ranking(top_n=10)
            for f in factors:
                ic = f.get("ic", 0)
                if abs(ic) < 0.02:
                    signals.append(ResearchSignal(
                        source="strategy_health", title=f"因子{f.get('name','?')} IC={ic:.3f} — 已失效",
                        action="factor", priority=3, confidence=0.75,
                        data={"factor": f.get("name", ""), "ic": ic},
                        suggested_track=f"factor_refresh_{f.get('name','')}",
                    ))
        except Exception as e:
            logger.exception(f"FactorStore健康检查失败: {e}")

        return signals

    # ============================================================
    # 源头6: 日历事件触发
    # ============================================================

    def check_calendar_events(self) -> list[ResearchSignal]:
        """财报季/指数调仓月/月末效应等周期性触发。"""
        signals = []
        today = datetime.now()
        month, day = today.month, today.day

        if month in (3, 4) and day <= 30:
            signals.append(ResearchSignal(
                source="calendar", title="年报密集披露期(3-4月) — 关注财报质量因子",
                action="factor", priority=2, confidence=0.7,
                suggested_track="earnings_season",
            ))
        if month in (8, 10) and day <= 31:
            signals.append(ResearchSignal(
                source="calendar", title="半年报/三季报披露期 — 关注业绩超预期因子",
                action="factor", priority=2, confidence=0.7,
                suggested_track="earnings_season",
            ))

        if month in (6, 12) and 1 <= day <= 15:
            signals.append(ResearchSignal(
                source="calendar", title="指数成分股调整窗口(6月/12月) — 关注调入/调出效应",
                action="strategy", priority=3, confidence=0.75,
                suggested_track="index_rebalance",
            ))

        if day == 1:
            signals.append(ResearchSignal(
                source="calendar", title="月初 — 建议跑全量因子扫描+策略健康检查",
                action="alert", priority=2, confidence=0.9,
            ))

        if today.weekday() == 4:
            signals.append(ResearchSignal(
                source="calendar", title="周五 — 建议检查周末前仓位+另类数据",
                action="alert", priority=1, confidence=0.9,
            ))

        return signals

    # ============================================================
    # 源头7: LLM情绪分析（C13: 非管线Stage，作为ResearchSource新信号类型）
    # ============================================================

    def scan_llm_sentiment(self, headlines: list[str] = None,
                           provider: str = "") -> list[ResearchSignal]:
        """
        C13: LLM情绪分析 — 对新闻标题/市场评论做情感打分。

        不作为全自动交易信号，priority≥4 的信号需要人工确认（P6）。
        provider 为空时使用配置文件中的 LLM provider（DeepSeek 等）。
        """
        signals = []
        if not headlines:
            return signals

        try:
            from config.loader import get_config
            cfg = get_config()
            llm_cfg = cfg.get("llm", {})
            api_key = llm_cfg.get("api_key", "")
            if not api_key:
                logger.info("LLM未配置API Key，跳过情绪分析")
                return signals
        except Exception:
            logger.exception("LLM config loading failed for sentiment analysis")
            return signals

        # C13: 使用 LLM API 对每条标题打分（0-10，10=极度乐观）
        for headline in headlines[:10]:
            try:
                try:
                    from backtest.analysis.llm_provider import call_llm
                    resp = call_llm(
                        system_prompt="你是财经新闻分析专家。对新闻标题做情绪评分，只返回0-10的数字，不要任何解释。",
                        user_prompt=f"给以下财经新闻标题的情绪打分(0=极度悲观, 5=中性, 10=极度乐观)，只返回数字:\n{headline}",
                        max_tokens=10,
                        temperature=0.1,
                    )
                    score = int(float(resp.strip()))
                    confidence = 0.7
                except Exception:
                    logger.exception(f"LLM sentiment scoring failed for headline, using default score")
                    score = 5
                    confidence = 0.3
                priority = 4 if score >= 8 or score <= 2 else (3 if score >= 7 or score <= 3 else 2)

                signal = ResearchSignal(
                    source="llm_sentiment",
                    title=f"LLM情绪评分{score}/10: {headline[:60]}",
                    action="alert",
                    priority=priority,
                    confidence=confidence,
                    data={"headline": headline, "sentiment_score": score, "provider": provider},
                    needs_review=(priority >= 4),  # P6: priority≥4需要人工确认
                )
                signals.append(signal)
            except Exception:
                logger.exception(f"LLM sentiment signal creation failed for headline")
                continue

        high_sentiment = [s for s in signals if s.priority >= 4]
        if high_sentiment:
            logger.info(f"LLM情绪分析: {len(signals)}条 → {len(high_sentiment)}条高优先级(≥4，需人工确认)")
        return signals

    # ============================================================
    # 源头8: 扩展研究源头（C16: 宏观/资金流向/龙虎榜/ETF/期权IV/社媒/监管/商品）
    # ============================================================

    def scan_extended_sources(self) -> list[ResearchSignal]:
        """
        C16: 研究源头扩展 — 8 个新增外部信号源。

        宏观经济 / 资金流向 / 龙虎榜 / ETF资金流 / 期权隐含波动率 /
        社交媒体 / 监管政策 / 大宗商品。每个源标记为低优先级（参考信号），
        极端情况时自动升级。
        """
        signals = []
        today = datetime.now()

        # 1. 宏观经济 — 央行利率/PMI/CPI 等
        signals.append(ResearchSignal(
            source="extended:macro", title="宏观数据监测：关注PMI/CPI/央行政策变动",
            action="alert", priority=1, confidence=0.6,
            data={"category": "宏观经济"},
        ))

        # 2. 资金流向 — 北向资金/主力资金
        signals.append(ResearchSignal(
            source="extended:capital_flow", title="资金流向监测：北向资金/主力资金净流入方向",
            action="strategy", priority=2, confidence=0.5,
            suggested_track="capital_flow_signal",
            data={"category": "资金流向"},
        ))

        # 3. 龙虎榜 — 游资/机构动向
        if today.weekday() < 5:  # 交易日
            signals.append(ResearchSignal(
                source="extended:dragon_tiger", title="龙虎榜监测：关注游资和机构席位动向",
                action="strategy", priority=2, confidence=0.5,
                suggested_track="dragon_tiger_momentum",
                data={"category": "龙虎榜"},
            ))

        # 4. ETF资金流 — 宽基/行业ETF申赎
        signals.append(ResearchSignal(
            source="extended:etf_flow", title="ETF资金流：宽基/行业ETF大额申赎信号",
            action="strategy", priority=1, confidence=0.5,
            data={"category": "ETF资金流"},
        ))

        # 5. 期权隐含波动率 — VIX/IV恐慌指标
        signals.append(ResearchSignal(
            source="extended:options_iv", title="期权IV监测：隐含波动率极端水平预警",
            action="alert", priority=2, confidence=0.55,
            data={"category": "期权波动率"},
        ))

        # 6. 社交媒体 — 散户情绪/讨论热度
        signals.append(ResearchSignal(
            source="extended:social_media", title="社交媒体监测：散户情绪和讨论热度异常",
            action="alert", priority=2, confidence=0.45,
            data={"category": "社交媒体"},
        ))

        # 7. 监管政策 — 行业政策/监管变动
        signals.append(ResearchSignal(
            source="extended:regulatory", title="监管政策监测：行业政策和监管变动跟踪",
            action="alert", priority=2, confidence=0.5,
            data={"category": "监管政策"},
        ))

        # 8. 大宗商品 — 黄金/原油/铜趋势
        signals.append(ResearchSignal(
            source="extended:commodities", title="大宗商品监测：黄金/原油/铜价格趋势",
            action="strategy", priority=1, confidence=0.5,
            suggested_track="commodity_correlation",
            data={"category": "大宗商品"},
        ))

        logger.info(f"扩展源头扫描: {len(signals)}条参考信号")
        return signals

    # ============================================================
    # 源头9: arXiv→LLM因子提取链路（C22）
    # ============================================================

    def extract_factors_from_papers(self, max_papers: int = 3) -> list[ResearchSignal]:
        """
        C22: arXiv→LLM因子提取链路。

        1. 扫描 arXiv q-fin 最新论文
        2. 提取论文摘要
        3. 调 LLM API 从摘要中提取可复现的因子公式
        4. IC 验证（|IC|>0.03 入库）

        需要 DeepSeek API Key（在 settings.local.yaml 中配置 llm.api_key）。
        无 API Key 时自动跳过。
        """
        signals = []
        try:
            from config.loader import get_config
            cfg = get_config()
            llm_cfg = cfg.get("llm", {})
            api_key = llm_cfg.get("api_key", "")
            if not api_key:
                logger.info("arXiv→LLM因子提取需要DeepSeek API Key，跳过")
                return signals
        except Exception:
            logger.exception("LLM config loading failed for arxiv factor extraction")
            return signals

        # Step 1: 扫描论文
        papers = self.scan_arxiv_papers(max_results=max_papers)
        factor_papers = [p for p in papers if p.action == "factor" and p.priority >= 2]

        for paper in factor_papers[:max_papers]:
            title = paper.data.get("title", "")
            link = paper.data.get("link", "")

            # Step 2: 调 LLM 提取因子公式
            try:
                from backtest.analysis.llm_provider import call_llm
                resp = call_llm(
                    system_prompt="你是量化因子研究员。从论文标题中提取可复现的因子公式。返回JSON: {\"factor_name\":\"简短名\",\"formula\":\"计算公式\",\"description\":\"一句话说明\"}",
                    user_prompt=f"从以下量化金融论文标题中提取一个可量化的因子公式:\n{title}",
                    max_tokens=300,
                    temperature=0.3,
                )
                import re
                match = re.search(r'\{[\s\S]*\}', resp)
                if match:
                    result = json.loads(match.group(0))
                else:
                    result = {}
                factor_name = result.get("factor_name", title[:30])
                formula = result.get("formula", "")
                description = result.get("description", "")
                llm_confidence = 0.45
            except Exception:
                logger.exception(f"LLM factor extraction failed for paper: {title[:60]}")
                factor_name = title[:30]
                formula = ""
                description = ""
                llm_confidence = 0.25

            signals.append(ResearchSignal(
                source="arxiv_llm_factor",
                title=f"LLM因子提取: {factor_name}",
                action="factor",
                priority=3,
                confidence=llm_confidence,
                data={
                    "paper_title": title,
                    "paper_link": link,
                    "factor_name": factor_name,
                    "formula": formula,
                    "description": description,
                    "source": "arxiv_llm",
                    "status": "pending_ic_validation",
                },
                suggested_track=f"llm_factor_{factor_name[:15].replace(' ', '_')}",
            ))

        logger.info(f"arXiv→LLM因子提取: {len(factor_papers)}篇论文 → {len(signals)}条因子候选")
        return signals

    # ============================================================
    # 源头10: L4 ML信号链路（C24: RL+另类数据+LightGBM三源融合）
    # ============================================================

    def scan_l4_signal(self) -> list[ResearchSignal]:
        """
        C24: L4 ML信号链路 — 三源融合（RL+另类数据+LightGBM因子）→ 综合信号。

        若有行情数据(df)，跑完整L4信号链，产出方向信号+置信度。
        无数据时返回空（不强行拉数据，保持扫描速度）。
        """
        signals = []
        if self.df is None or len(self.df) < 60:
            return signals

        try:
            from backtest.analysis.l4_integration import L4SignalChain
            l4 = L4SignalChain(self.df, use_rl=False, symbol="")
            result = l4.run()

            signal_val = result.get("signal", 0.5)
            action = result.get("action", "hold")
            confidence = result.get("confidence", 0.5)
            components = result.get("components", {})

            # 极端信号→高优先级
            if action == "buy" and signal_val > 0.7:
                priority = 4
            elif action == "sell" and signal_val < 0.3:
                priority = 4
            elif action in ("buy", "sell"):
                priority = 3
            else:
                priority = 2

            signals.append(ResearchSignal(
                source="l4_signal",
                title=f"L4综合信号: {action}({signal_val:.3f}) | "
                      f"alt={components.get('alt', 0):.2f} "
                      f"rl={components.get('rl', 0):.2f} "
                      f"ml={components.get('ml', 0):.2f}",
                action="strategy",
                priority=priority,
                confidence=confidence,
                data={
                    "signal": signal_val,
                    "action": action,
                    "components": components,
                    "timestamp": result.get("timestamp", ""),
                },
                suggested_track="l4_signal_chain" if action != "hold" else "",
                needs_review=(priority >= 4),
            ))

            logger.info(f"L4信号扫描: {action} signal={signal_val:.3f} conf={confidence:.2f}")
        except Exception as e:
            logger.debug(f"L4信号扫描跳过: {e}")

        return signals

    # ============================================================
    # 汇总 & 转换
    # ============================================================

    def scan_all(self, use_cache: bool = True) -> list[ResearchSignal]:
        """运行全部6个源头，去重+噪声过滤，返回有效信号列表。"""
        if use_cache:
            self._load_history(days=7)

        all_signals: list[ResearchSignal] = []
        sources = [
            ("arxiv", self.scan_arxiv_papers),
            ("data_freshness", self.check_data_freshness),
            ("regime_change", self.detect_regime_change),
            ("alternative", self.scan_alternative_data),
            ("strategy_health", self.check_strategy_health),
            ("calendar", self.check_calendar_events),
            ("arxiv_llm_factor", self.extract_factors_from_papers),    # C22
            ("extended_sources", self.scan_extended_sources),          # C16
            ("l4_signal", self.scan_l4_signal),                       # C24
        ]
        for name, method in sources:
            try:
                batch = method()
                all_signals.extend(batch)
                logger.info(f"[{name}] {len(batch)}条信号")
            except Exception as e:
                logger.warning(f"[{name}] 扫描失败: {e}")

        filtered = _confidence_gate(all_signals, min_conf=0.3)
        filtered = _fwer_filter(filtered)
        if self._history:
            filtered = _cooldown_check(filtered, self._history)

        filtered.sort(key=lambda s: (-s.priority, -s.confidence))
        logger.info(f"scan_all: {len(all_signals)}→{len(filtered)}条 (过滤{len(all_signals)-len(filtered)}条噪声)")
        return filtered

    def to_tracks(self, signals: list[ResearchSignal]) -> list:
        """将有效信号转为Pipeline Track列表。"""
        tracks = []
        for s in signals:
            if s.action in ("factor", "strategy") and s.priority >= 3:
                try:
                    from core.pipeline import Track
                    track = Track(
                        name=s.suggested_track or f"src_{s.source}",
                        factors=s.data.get("factors", []),
                        strategy_params=s.data.get("strategy_params", {}),
                    )
                    tracks.append(track)
                except ImportError:
                    pass
        return tracks

    def save_signals(self, signals: list[ResearchSignal]) -> int:
        """将信号存入ResearchStore。"""
        try:
            from data.vault.research_store import ResearchStore
            store = ResearchStore()
            return store.save_batch([s.to_dict() for s in signals])
        except Exception as e:
            logger.warning(f"保存信号失败: {e}")
            return 0

    def print_report(self, signals: list[ResearchSignal]) -> str:
        """生成可读的信号报告。"""
        if not signals:
            return "无研究信号。"
        lines = [f"研究源头扫描报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}", "=" * 50]
        by_source: dict[str, list] = {}
        for s in signals:
            by_source.setdefault(s.source, []).append(s)
        for src, batch in by_source.items():
            lines.append(f"\n[{src}] {len(batch)}条:")
            for s in batch[:5]:
                flag = "[HIGH]" if s.priority >= 5 else ("[MED]" if s.priority >= 3 else "[LOW]")
                review = " [需人工]" if s.needs_review else ""
                lines.append(f"  {flag} P{s.priority} | {s.title}{review}")
        lines.append(f"\n总计: {len(signals)}条信号 (P5={sum(1 for s in signals if s.priority>=5)}, P4={sum(1 for s in signals if s.priority==4)}, P3={sum(1 for s in signals if s.priority==3)})")
        report = "\n".join(lines)
        logger.info(report)
        return report


# ============================================================
# CLI 测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("研究源头 (ResearchSource) — 烟雾测试")
    print("=" * 60)

    rs = ResearchSource()
    print("\n[1/6] arXiv论文扫描...")
    arxiv = rs.scan_arxiv_papers(max_results=3)
    print(f"  → {len(arxiv)}条")

    print("\n[2/6] 数据新鲜度(无df)...")
    dfresh = rs.check_data_freshness()
    print(f"  → {len(dfresh)}条")

    print("\n[3/6] 市场结构变化(无df)...")
    regime = rs.detect_regime_change()
    print(f"  → {len(regime)}条 (无数据时为空)")

    print("\n[4/6] 另类数据...")
    alt = rs.scan_alternative_data()
    print(f"  → {len(alt)}条")

    print("\n[5/6] 策略健康...")
    health = rs.check_strategy_health()
    print(f"  → {len(health)}条")

    print("\n[6/6] 日历事件...")
    cal = rs.check_calendar_events()
    print(f"  → {len(cal)}条")

    all_s = arxiv + dfresh + regime + alt + health + cal
    print(f"\n总计: {len(all_s)}条原始信号")
    filtered = _confidence_gate(all_s)
    print(f"置信度过滤后: {len(filtered)}条")
    filtered = _fwer_filter(filtered)
    print(f"FWER过滤后: {len(filtered)}条")
    rs.print_report(filtered)
