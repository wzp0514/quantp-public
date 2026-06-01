"""
定时体检调度器 — 每日/每周/每月自动健康检查

不是 24 小时挖矿，而是在正确的时间跑正确的检查。
报告写入 reports/ 目录，可作为 git 提交或日报素材。

任务:
  每日 (<10s):  数据源健康 + 策略信号漂移检测
  每周 (<2min): 因子 IC 重算 + Walk-Forward 滚动验证
  每月 (<10min): LLM因子挖掘 + 策略批量重筛

用法
--------
>>> from live.monitor.scheduler import HealthScheduler
>>> hs = HealthScheduler()
>>> hs.daily_check()     # 每日
>>> hs.weekly_check()    # 每周
>>> hs.monthly_check()   # 每月
>>> print(hs.last_report)
"""

import os
import json
import time
from datetime import datetime

from config.log import get_logger

logger = get_logger("scheduler")


class HealthScheduler:
    """
    定时体检调度器。

    所有检查结果写入 reports/health/ 目录下的 JSON + TXT 文件。
    """

    def __init__(self, report_dir: str = "reports/health"):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        self.last_report = {}
        self._today = datetime.now().strftime("%Y-%m-%d")

    def _save_report(self, name: str, data: dict):
        """保存报告"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.report_dir, f"{name}_{ts}")
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Report saved: {path}.json")
        return path

    # ═══════════════════════════════════════════
    # 每日检查 (<10s)
    # ═══════════════════════════════════════════

    def daily_check(self) -> dict:
        """
        每日: 数据源健康 + 最新行情检查。
        出问题及时告警，不等到策略跑的时候才发现数据断了。
        """
        logger.info("=== Daily Health Check ===")
        t0 = time.time()
        report = {"date": self._today, "timestamp": datetime.now().isoformat()}

        # 1. 数据源健康
        try:
            from data.fetchers.fallback import check_dependencies, get_source_health, fetch_index_daily_safe
            deps = check_dependencies()
            report["data_sources"] = deps
            available = sum(1 for v in deps.values() if v)
            report["data_ok"] = available >= 1
            if not report["data_ok"]:
                logger.warning("ALL DATA SOURCES DOWN!")
                report["alerts"] = report.get("alerts", []) + ["所有数据源不可用"]

            # 2. 最新行情
            df = fetch_index_daily_safe("沪深300",
                                        start_date=(datetime.now().replace()).strftime("%Y%m%d"),
                                        end_date=datetime.now().strftime("%Y%m%d"))
            if df.empty:
                df = fetch_index_daily_safe("沪深300",
                                            start_date=(datetime.now().replace()).strftime("%Y%m01"),
                                            end_date=datetime.now().strftime("%Y%m%d"))
            report["latest_data"] = {
                "rows": len(df),
                "last_date": str(df["date"].max().date()) if not df.empty and "date" in df.columns else "N/A",
            }
        except Exception as e:
            report["data_error"] = str(e)
            report["data_ok"] = False

        # 3. 策略信号监控（如果在跑实盘/纸上）
        report["strategy_health"] = self._check_running_strategies()

        report["elapsed"] = round(time.time() - t0, 1)
        logger.info(f"Daily check done ({report['elapsed']}s): data_ok={report.get('data_ok')}")
        self.last_report["daily"] = report
        self._save_report("daily", report)
        return report

    # ═══════════════════════════════════════════
    # 每周检查 (<2min)
    # ═══════════════════════════════════════════

    def weekly_check(self) -> dict:
        """
        每周: 因子 IC 重新计算 + Walk-Forward 滚动验证。
        监控已有策略是否退化，因子是否过期。
        """
        logger.info("=== Weekly Check ===")
        t0 = time.time()
        report = {"date": self._today, "timestamp": datetime.now().isoformat()}

        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            df = fetch_index_daily_safe("沪深300", "20150101", datetime.now().strftime("%Y%m%d"))

            # 1. 因子 IC 分析
            from backtest.analysis.factor_miner import FactorMiner
            fm = FactorMiner(df)
            result = fm.mine()
            report["factors"] = {
                "total": len(result["results"]),
                "strong": result["strong_count"],
                "medium": result["medium_count"],
                "top3": [
                    {"name": r["factor"], "ic": r["ic"], "rank_ic": r.get("rank_ic", 0)}
                    for r in result["top_factors"][:3]
                ],
                "redundant_pairs": len(result["redundant_pairs"]),
            }
            logger.info(f"Factors: {result['strong_count']} strong, {result['medium_count']} medium")

            # 2. Walk-Forward 验证（双均线作为基准）
            from backtest.analysis.validate import rolling_window_validate
            from backtest.strategies.builtin.ma_cross import MaCrossStrategy
            wf = rolling_window_validate(MaCrossStrategy, df, is_years=2, oos_months=6, step_months=1,
                                         fast=5, slow=20)
            report["walk_forward"] = {
                "n_windows": wf["metrics"]["n_windows"],
                "wfe": wf["metrics"]["wfe"],
                "win_rate": wf["metrics"]["win_rate"],
                "passed": wf["passed"],
            }
            logger.info(f"WFA: {wf['metrics']['n_windows']} windows, {'PASS' if wf['passed'] else 'FAIL'}")

        except Exception as e:
            report["error"] = str(e)
            logger.error(f"Weekly check failed: {e}")

        report["elapsed"] = round(time.time() - t0, 1)
        self.last_report["weekly"] = report
        self._save_report("weekly", report)
        return report

    # ═══════════════════════════════════════════
    # 每月检查 (<10min)
    # ═══════════════════════════════════════════

    def monthly_check(self) -> dict:
        """
        每月: LLM 因子挖掘 + 策略批量重筛。
        尝试发现新因子，用最新数据重新评估所有策略。
        """
        logger.info("=== Monthly Check ===")
        t0 = time.time()
        report = {"date": self._today, "timestamp": datetime.now().isoformat()}

        # 1. 尝试 LLM 因子挖掘（需要 DeepSeek API key）
        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            df = fetch_index_daily_safe("沪深300", "20180101", datetime.now().strftime("%Y%m%d"))

            from backtest.analysis.llm_factor_miner import LLMFactorMiner
            llm = LLMFactorMiner(df)
            llm_result = llm.mine(n_rounds=3)
            report["llm_factors"] = {
                "rounds": llm_result.get("rounds", 0),
                "new_factors": len(llm_result.get("factors", [])),
                "best_ic": llm_result.get("best_ic", 0),
            }
            logger.info(f"LLM factors: {report['llm_factors']['new_factors']} new")
        except Exception as e:
            report["llm_factors"] = {"error": str(e), "note": "需要 DeepSeek API key"}
            logger.debug(f"LLM mining skipped: {e}")

        # 2. 策略批量重筛
        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            df = fetch_index_daily_safe("沪深300", "20200101", datetime.now().strftime("%Y%m%d"))

            from backtest.engine.batch_runner import BatchRunner, BatchConfig
            from backtest.strategy_market import ALL_STRATEGIES
            strategies = {}
            for name, info in ALL_STRATEGIES.items():
                cls = info.get("class")
                if cls:
                    strategies[name] = (cls, info.get("params", {}))
            runner = BatchRunner(df, BatchConfig(cash=100000))
            batch = runner.run(strategies)
            best = batch.best()
            report["strategy_rescreen"] = {
                "total": batch.total,
                "passed": batch.passed,
                "best_name": best["name"] if best else "N/A",
                "best_sr": best["result"].get("sharpe") if best else 0,
            }
            logger.info(f"Rescreen: {batch.passed}/{batch.total} passed")
        except Exception as e:
            report["strategy_rescreen"] = {"error": str(e)}

        report["elapsed"] = round(time.time() - t0, 1)
        self.last_report["monthly"] = report
        self._save_report("monthly", report)
        return report

    def run_all(self) -> dict:
        """一键跑完日/周/月所有检查"""
        daily = self.daily_check()
        weekly = self.weekly_check()
        monthly = self.monthly_check()
        summary = {
            "date": self._today,
            "daily": f"data_ok={daily.get('data_ok')} ({daily.get('elapsed')}s)",
            "weekly": f"factors={weekly.get('factors',{}).get('strong',0)} strong ({weekly.get('elapsed')}s)",
            "monthly": f"strategies={monthly.get('strategy_rescreen',{}).get('passed',0)} passed ({monthly.get('elapsed')}s)",
        }
        self._save_report("summary", summary)
        logger.info(f"All checks done: {summary}")
        return summary

    def _check_running_strategies(self) -> dict:
        """检查当前运行中的策略状态（占位——需要策略运行时注册）"""
        return {
            "active_count": 0,
            "note": "无活跃策略监控（未连接实盘/纸上交易实例）",
        }


# ============================================================
# 命令行
# ============================================================
# python live/monitor/scheduler.py

if __name__ == "__main__":
    hs = HealthScheduler()

    print("=== Daily Check ===")
    d = hs.daily_check()
    print(f"  data_ok={d.get('data_ok')} ({d.get('elapsed')}s)")

    print()
    print("=== Weekly Check ===")
    w = hs.weekly_check()
    print(f"  factors={w.get('factors',{}).get('strong',0)} strong ({w.get('elapsed')}s)")

    print()
    print("=== Monthly Check ===")
    m = hs.monthly_check()
    print(f"  strategies={m.get('strategy_rescreen',{}).get('passed',0)} passed ({m.get('elapsed')}s)")
