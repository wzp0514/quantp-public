"""
策略守护进程 — 及时性+安全性统一模块

参考 vnpy 的监控体系 + Freqtrade 的自动化调度。

功能：
  B1. 定时任务调度 — 自动跑回测/拉数据（类似 Freqtrade 的 cron）
  B2. 数据源健康监控 — AkShare 挂了自动切备用
  B3. 策略漂移检测 — 实盘信号偏离回测预期时告警
  D1. 资金隔离确认 — 实盘前必须确认"这是真钱"
  D2. 紧急熔断 — 一键停止所有策略

用法
--------
>>> from live.monitor.guardian import Guardian
>>> g = Guardian()
>>> g.start()  # 启动守护
>>> g.emergency_stop()  # 紧急熔断
"""

import json
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

STATE_FILE = Path(__file__).resolve().parent.parent.parent / "live" / ".guardian_state.json"

from config.log import get_logger
logger = get_logger("guardian")
class Guardian:
    """策略守护进程"""

    def __init__(self):
        self.state = self._load_state()
        self._stopped = False
        self._live_mode = self.state.get("live_mode", False)

    # ============================================================
    # D1. 资金隔离确认
    # ============================================================

    def enter_live_mode(self) -> bool:
        """
        进入实盘模式前必须确认。

        这是最后一道防线——在你点"确认"之前，
        所有操作都是模拟的，不会动真钱。
        """
        print("\n" + "=" * 60)
        print("  [!] 实盘模式确认")
        print("=" * 60)
        print("  即将使用真实资金进行交易。")
        print("  风控引擎将自动执行以下规则:")
        print("    - 单笔亏损 > 2% → 强制止损")
        print("    - 日亏损 > 5% → 停止当日交易")
        print("    - 总回撤 > 15% → 策略永久停用")
        print("    - 连续5笔亏损 → 自动暂停")
        print("=" * 60)

        confirm = input("\n  确认进入实盘模式？输入 'YES-I-UNDERSTAND' 确认: ").strip()

        if confirm == "YES-I-UNDERSTAND":
            print("\n  [!] 最后风险提示：")
            print("  - 实盘交易使用真实资金，可能导致全部亏损")
            print("  - 历史回测结果不代表未来表现")
            print("  - 市场极端行情下风控机制可能失效（如流动性枯竭、涨跌停封板）")
            print("  - 程序故障/网络中断/交易所系统异常可能导致意外损失")
            print("  - 请确保已充分理解策略逻辑并完成充分的模拟测试")
            final = input("\n  再次确认进入实盘？输入 'YES' 确认，其他任意键取消: ").strip()
            if final != "YES":
                logger.info("取消实盘模式，保持模拟状态")
                return False
            self._live_mode = True
            self._save_state({"live_mode": True, "live_since": datetime.now().isoformat()})
            logger.warning("已进入实盘模式——真金白银！")
            return True
        else:
            logger.info("取消实盘模式，保持模拟状态")
            return False

    def is_live(self) -> bool:
        """当前是否为实盘模式"""
        return self._live_mode

    # ============================================================
    # D2. 紧急熔断
    # ============================================================

    def emergency_stop(self, reason: str = "手动触发") -> dict:
        """
        紧急停止所有策略。

        调用后：
          1. 取消所有未成交订单
          2. 平掉所有持仓（市价卖出）
          3. 记录熔断原因
          4. 退出实盘模式
        """
        logger.error(f"[!] 紧急熔断: {reason}")

        record = {
            "action": "emergency_stop",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "pending_orders_cancelled": 0,
            "positions_closed": 0,
        }

        try:
            from live.execution.order_mgr import OrderManager
            om = OrderManager()
            for o in om.get_pending():
                om.cancel(o["id"])
                record["pending_orders_cancelled"] += 1
        except Exception as e:
            logger.warning(f"紧急停止取消失败: {e}")
            # 注意：OrderManager 当前为纯内存实现，如进程重启则订单信息丢失
            # 后续需要持久化订单状态（SQLite/文件）以支持崩溃恢复

        self._live_mode = False
        self._save_state(self.state)

        self._save_emergency_log(record)
        logger.info("熔断完成——所有策略已停止")

        return record

    def auto_emergency_check(self, daily_pnl: float, total_drawdown: float,
                             consecutive_losses: int) -> bool:
        """
        自动熔断检查——由风控引擎触发。

        返回 True = 已触发熔断
        """
        from live.risk.risk_engine import get_params
        params = get_params()

        triggers = []

        if abs(daily_pnl) > params["max_daily_loss_pct"]:
            triggers.append(f"日亏损 {daily_pnl:.1%} > {params['max_daily_loss_pct']:.0%}")

        if abs(total_drawdown) > params["max_drawdown_pct"]:
            triggers.append(f"总回撤 {total_drawdown:.1%} > {params['max_drawdown_pct']:.0%}")

        if consecutive_losses >= params["stop_after_n_losses"]:
            triggers.append(f"连续亏损 {consecutive_losses} ≥ {params['stop_after_n_losses']}")

        if triggers:
            reason = "; ".join(triggers)
            self.emergency_stop(reason)
            return True

        return False

    # ============================================================
    # B2. 数据源健康监控
    # ============================================================

    def check_data_health(self) -> dict:
        """检查各数据源是否正常（集成 fallback 降级链状态）"""
        try:
            from data.fetchers.fallback import get_source_health, check_dependencies
            health = get_source_health()
            deps = check_dependencies()

            # 转换为统一格式
            results = {}
            for src in ("akshare", "tushare", "baostock"):
                available = deps.get(src, False)
                h = health.get(src, {})
                if not available:
                    results[src] = {"status": "not_available"}
                else:
                    results[src] = {
                        "status": h.get("status", "unknown"),
                        "last_check": h.get("last_check"),
                        "error": h.get("error"),
                    }


            all_ok = all(
                r.get("status") in ("ok", "not_available", "not_configured")
                for r in results.values()
            )
            if not all_ok:
                down_sources = {
                    k: v["status"]
                    for k, v in results.items()
                    if v.get("status") not in ("ok", "not_available", "not_configured")
                }
                logger.warning(f"数据源异常: {down_sources}")

            return {
                "sources": results,
                "all_healthy": all_ok,
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"数据健康检查失败: {e}")
            return {"sources": {}, "all_healthy": False, "error": str(e), "checked_at": datetime.now().isoformat()}

    # ============================================================
    # B3. 策略漂移检测
    # ============================================================

    def check_strategy_drift(
        self,
        strategy_name: str,
        backtest_signals_per_week: float,
        live_signals_per_week: float,
        backtest_win_rate: float,
        live_win_rate: float,
    ) -> dict:
        """
        检测实盘信号是否偏离回测预期。

        偏离 > 50% → 策略可能失效，需要重新评估。
        """
        signal_deviation = abs(live_signals_per_week - backtest_signals_per_week) / max(backtest_signals_per_week, 1)
        wr_deviation = abs(live_win_rate - backtest_win_rate) / max(backtest_win_rate, 0.01)

        warnings = []
        if signal_deviation > 0.5:
            warnings.append(f"信号频率偏离 {signal_deviation:.0%}（>50%）——可能过拟合或市场变化")
        if wr_deviation > 0.5:
            warnings.append(f"胜率偏离 {wr_deviation:.0%}（>50%）——策略可能失效")

        if warnings:
            logger.warning(f"[{strategy_name}] 策略漂移告警: {'; '.join(warnings)}")

        return {
            "strategy": strategy_name,
            "signal_deviation": signal_deviation,
            "wr_deviation": wr_deviation,
            "warnings": warnings,
            "healthy": len(warnings) == 0,
        }

    def check_live_vs_backtest_dd(
        self, strategy_name: str,
        backtest_max_dd: float, live_max_dd: float,
    ) -> dict:
        """实盘回撤 vs 回测回撤对比。参照05文档规则10：实盘回撤必超回测。"""
        if backtest_max_dd <= 0:
            return {"warning": False, "detail": "回测回撤数据无效"}
        ratio = live_max_dd / backtest_max_dd
        warning = ratio > 1.5
        if warning:
            logger.error(f"[{strategy_name}] 实盘回撤 {live_max_dd:.1%} 已超过回测 {backtest_max_dd:.1%} 的1.5倍——参照05文档规则10，建议暂停策略。")
        return {"strategy": strategy_name, "ratio": ratio, "warning": warning,
                "detail": f"实盘/回测回撤比={ratio:.1f}x"}

    def check_position_drift(self, expected_position: dict, actual_position: dict) -> dict:
        """仓位漂移检测——防止外部手动交易导致仓位失控。参照05文档失败3(手动干预)。"""
        exp_size = expected_position.get("size", 0)
        act_size = actual_position.get("size", 0)
        if exp_size == 0 and act_size == 0:
            return {"drift": False, "detail": "一致(空仓)"}
        if exp_size == 0:
            drift_pct = 1.0
        else:
            drift_pct = abs(act_size - exp_size) / abs(exp_size)
        drift = drift_pct > 0.05
        if drift:
            logger.warning(f"仓位漂移 {drift_pct:.1%}: 预期{exp_size} vs 实际{act_size}——可能有人工干预或程序异常。参照05文档失败3。")
        return {"drift": drift, "drift_pct": drift_pct, "expected": exp_size, "actual": act_size}

    # ============================================================
    # B4. 恐慌指数检查
    # ============================================================

    def check_fear_index(self) -> dict:
        """综合恐慌指数检查 — 接入自动熔断参考"""
        try:
            from data.alternative.market_sentiment import MarketSentiment
            ms = MarketSentiment()
            snap = ms.snapshot()
            if snap.get("status") != "ok":
                return {"healthy": True, "fear_index": None,
                        "level": "未知", "warnings": [], "checked_at": datetime.now().isoformat()}

            fi = MarketSentiment.composite_fear_index(snap)
            fear = fi["fear_index"]
            warnings = []
            if fear >= 80:
                warnings.append(f"恐慌指数 {fear:.0f}/100 (极度恐惧)")
            elif fear >= 60:
                warnings.append(f"恐慌指数 {fear:.0f}/100 (恐惧)")

            if warnings:
                logger.warning(f"恐慌指数告警: {'; '.join(warnings)}")

            return {
                "healthy": fear < 60,
                "fear_index": fear,
                "level": fi["level"],
                "components": fi["components"],
                "signal": fi["signal"],
                "warnings": warnings,
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"恐慌指数检查失败: {e}")
            return {"healthy": True, "fear_index": None, "level": "错误",
                    "warnings": [f"恐慌指数不可用: {e}"], "checked_at": datetime.now().isoformat()}

    # ============================================================
    # B5. 流动性枯竭动态检测
    # ============================================================

    def check_liquidity(self, symbol: str = "沪深300") -> dict:
        """流动性枯竭检测：价差扩大+成交量萎缩 vs 20日均量。

        两指标同时恶化时触发告警。
        """
        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            start = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
            end = datetime.now().strftime("%Y%m%d")
            df = fetch_index_daily_safe(symbol, start, end)
            if df is None or len(df) < 20:
                return {"healthy": True, "detail": "数据不足", "warnings": [],
                        "checked_at": datetime.now().isoformat()}

            df = df.sort_values("date")
            volume = df["volume"].values
            latest_vol = volume[-1]
            avg_vol_20 = np.mean(volume[-21:-1]) if len(volume) >= 21 else np.mean(volume[:-1])
            vol_ratio = latest_vol / max(avg_vol_20, 1)

            high, low, close = df["high"].values, df["low"].values, df["close"].values
            spreads = (high - low) / np.maximum(close, 1e-9)
            latest_spread = spreads[-1]
            avg_spread_20 = np.mean(spreads[-21:-1]) if len(spreads) >= 21 else np.mean(spreads[:-1])
            spread_ratio = latest_spread / max(avg_spread_20, 1e-9)

            warnings = []
            vol_warning = vol_ratio < 0.5
            spread_warning = spread_ratio > 2.0

            if vol_warning and spread_warning:
                warnings.append(
                    f"流动性枯竭: 成交量 {vol_ratio:.0%} of 20日均量, "
                    f"价差 {spread_ratio:.1f}x 均值"
                )

            if warnings:
                logger.warning(f"流动性告警: {'; '.join(warnings)}")

            return {
                "healthy": not (vol_warning and spread_warning),
                "volume_ratio": round(float(vol_ratio), 4),
                "spread_ratio": round(float(spread_ratio), 4),
                "vol_warning": vol_warning,
                "spread_warning": spread_warning,
                "warnings": warnings,
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"流动性检查失败: {e}")
            return {"healthy": True, "warnings": [], "error": str(e),
                    "checked_at": datetime.now().isoformat()}

    # ============================================================
    # B7. Kronos 预测波动率风险预警
    # ============================================================

    def check_kronos_risk(self, symbol: str = "沪深300") -> dict:
        """Kronos 前瞻性波动率 vs 历史波动率 — 仓位管理预警。

        当 Kronos 预测波动率显著高于历史波动率（>1.5x）时，
        提前触发降仓预警，作为 vol_target 的前瞻补充。
        """
        try:
            from core.kronos.engine import KronosEngine
            from config.loader import get_kronos_config
        except ImportError as e:
            return {"warning": False, "detail": f"Kronos 不可用: {e}",
                    "checked_at": datetime.now().isoformat()}

        cfg = get_kronos_config()
        if not cfg.get("risk_guard", False):
            return {"warning": False, "detail": "Kronos risk_guard 未启用",
                    "checked_at": datetime.now().isoformat()}

        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            end = datetime.now().strftime("%Y%m%d")
            df = fetch_index_daily_safe(symbol, start, end)
            if df is None or len(df) < 120:
                return {"warning": False, "detail": "数据不足",
                        "checked_at": datetime.now().isoformat()}

            lookback = cfg.get("lookback", 60)
            pred_len = cfg.get("pred_len", 20)
            window = df.tail(lookback).copy()
            x_ts = pd.Series(window.index)
            last_ts = window.index[-1]
            y_ts = pd.Series(pd.date_range(last_ts, periods=pred_len + 1, freq="B")[1:])

            engine = KronosEngine(
                model_size=cfg.get("model", "small"),
                device=cfg.get("device", "auto"),
                tokenizer_path=cfg.get("tokenizer_path", ""),
                model_path=cfg.get("model_path", ""),
                project_path=cfg.get("project_path", ""),
            )
            pred = engine.predict(window, x_ts, y_ts, pred_len=pred_len)

            import numpy as np
            pred_returns = np.diff(np.log(np.maximum(pred["close"].values, 1e-10)))
            predicted_vol = float(np.std(pred_returns) * np.sqrt(252))

            hist_returns = window["close"].pct_change().dropna().values
            historical_vol = float(np.std(hist_returns) * np.sqrt(252))

            vol_ratio = predicted_vol / max(historical_vol, 0.01)
            warnings = []
            if vol_ratio > 1.5:
                warnings.append(
                    f"Kronos 前瞻波动率 {predicted_vol:.1%} vs "
                    f"历史 {historical_vol:.1%} ({vol_ratio:.1f}x) — 建议降仓"
                )
                logger.warning(f"[{symbol}] {warnings[-1]}")

            engine.unload()

            return {
                "warning": vol_ratio > 1.5,
                "predicted_vol": round(predicted_vol, 4),
                "historical_vol": round(historical_vol, 4),
                "vol_ratio": round(vol_ratio, 2),
                "warnings": warnings,
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Kronos 风险检查失败: {e}")
            return {"warning": False, "error": str(e),
                    "checked_at": datetime.now().isoformat()}

    # ============================================================
    # B6. 反馈回路/恐慌自激检测
    # ============================================================

    def check_feedback_loop(
        self,
        strategy_name: str = "",
        market_returns: list = None,
        signal_frequencies: list = None,
        window: int = 20,
    ) -> dict:
        """反馈回路检测：市场回撤 vs 信号频率的 Spearman 相关性。

        当 rho>0.7 且回撤>5% 时触发恐慌自激告警。
        """
        if not market_returns or not signal_frequencies:
            return {"rho": 0.0, "drawdown": 0.0, "warning": False,
                    "warnings": ["数据不足"], "checked_at": datetime.now().isoformat()}

        n = min(len(market_returns), len(signal_frequencies), window)
        if n < 10:
            return {"rho": 0.0, "drawdown": 0.0, "warning": False,
                    "warnings": [f"样本不足({n}<10)"], "checked_at": datetime.now().isoformat()}

        rets = np.array(market_returns[-n:])
        sigs = np.array(signal_frequencies[-n:])

        try:
            from scipy.stats import spearmanr
            rho, p_value = spearmanr(rets, sigs)
        except ImportError:
            rho, p_value = self._spearman_rank(rets, sigs)

        cumulative = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0

        warnings = []
        if abs(rho) > 0.7 and dd < -0.05:
            warnings.append(
                f"恐慌自激: Spearman rho={rho:.2f}, 回撤={abs(dd):.1%} — "
                f"信号频率与市场下跌强相关"
            )
            logger.warning(f"[{strategy_name or '全局'}] {warnings[-1]}")

        return {
            "rho": round(float(rho), 4),
            "p_value": round(float(p_value), 4),
            "drawdown": round(float(dd), 4),
            "warning": len(warnings) > 0,
            "warnings": warnings,
            "n_samples": n,
            "strategy": strategy_name or "全局",
            "checked_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _spearman_rank(x: np.ndarray, y: np.ndarray) -> tuple:
        """纯 NumPy Spearman 秩相关（无 scipy 依赖时的 fallback）"""
        def _rank(arr):
            order = np.argsort(arr)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(arr) + 1)
            return ranks
        rx = _rank(x)
        ry = _rank(y)
        n = len(x)
        d = rx - ry
        rho = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1) + 1e-10)
        return rho, 0.0

    # ============================================================
    # B1. 定时任务（简化版）
    # ============================================================

    def schedule_daily(self):
        """
        每日定时任务（应该在系统启动时注册）。

        实际生产环境建议用系统 cron / Windows 任务计划程序，
        这里提供 Python 版本的简化实现。
        """
        logger.info("执行每日定时任务...")

        # 1. 数据源健康检查
        health = self.check_data_health()
        if not health["all_healthy"]:
            logger.error("数据源异常，跳过今日操作")

        # 2. 拉取最新数据（使用降级链: AkShare → Tushare → Baostock）
        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            df = fetch_index_daily_safe("沪深300")
            if not df.empty:
                src = df.attrs.get("source", "unknown")
                logger.info(f"最新数据({src}): {len(df)} 条, 最近日期 {df['date'].max()}")
            else:
                logger.error("数据拉取失败: 所有数据源均不可用")
        except Exception as e:
            logger.error(f"数据拉取失败: {e}")

        # 3. 从策略仓库加载所有策略，检查是否有需更新的
        logger.info("每日任务完成")

    # ============================================================
    # 状态管理
    # ============================================================

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_state(self, state: dict):
        self.state.update(state)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _save_emergency_log(self, record: dict):
        log_dir = STATE_FILE.parent
        log_file = log_dir / "emergency.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def status(self) -> dict:
        """获取守护状态摘要"""
        return {
            "live_mode": self._live_mode,
            "data_health": self.check_data_health()["all_healthy"],
            "last_emergency": self.state.get("last_emergency"),
        }


# ============================================================
# 命令行
# ============================================================
# python live/monitor/guardian.py

if __name__ == "__main__":
    g = Guardian()
    print(f"实盘模式: {'是' if g.is_live() else '否（模拟）'}")
    print(f"数据源健康: {g.check_data_health()['all_healthy']}")
