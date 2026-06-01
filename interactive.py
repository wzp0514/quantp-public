#!/usr/bin/env python
"""
量化鲲鹏 (QuantP) 交互式菜单 — 记不住命令？选数字就行。

启动:
    python interactive.py

然后按数字选择你要做的事，不需要记任何参数。

完整使用说明见 readme.md → 「菜单使用说明」章节。
[AI] = 可无人值守自动跑（11项：2,3,4,6,7,8,10,16,17,22 + 数据菜单[3]）
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _pick(choices: list[str], prompt: str = "请选择") -> str:
    """通用选择器"""
    print(f"\n{prompt}:")
    for i, c in enumerate(choices, 1):
        print(f"  [{i}] {c}")
    while True:
        try:
            n = int(input("> "))
            if 1 <= n <= len(choices):
                return choices[n - 1]
        except ValueError:
            pass
        print(f"输入 1-{len(choices)} 之间的数字")


def _input_num(prompt: str, default: float) -> float:
    """输入数字，回车用默认值"""
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _input_str(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default


def _pick_date_preset(default_start="20200101", default_end=None):
    """回测周期预设 — 支持全周期/完整牛熊/通用/近期/自定义"""
    today = datetime.now().strftime("%Y%m%d")
    if default_end is None:
        default_end = today

    presets = [
        ("全周期（2005至今）", "20050101", today),
        ("完整牛熊（2015至今）", "20150101", today),
        ("通用（2020至今）", "20200101", today),
        ("近期（2023至今）", "20230101", today),
        ("自定义", None, None),
    ]

    choice = _pick([p[0] for p in presets], "选择回测周期")
    for label, s, e in presets:
        if choice == label:
            if s is None:
                return _input_str("开始日期", default_start), _input_str("结束日期", default_end)
            print(f"  已选: {s} ~ {e}")
            return s, e
    return default_start, default_end


def menu_quick():
    """5分钟快速体验——无需配置，拿缓存数据直跑回测"""
    print("\n" + "=" * 50)
    print("  快速体验（5分钟）— 零配置，直接看结果")
    print("=" * 50)

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.engine.bt_runner import run_backtest
    from backtest.analysis.report import generate_report, plot_equity_curve, scorecard

    # 1. 尝试从缓存拉取沪深300数据
    print("\n[1/3] 获取数据...")
    df = fetch_index_daily_safe("沪深300", "20200101", "20251231")

    if df is None or df.empty:
        print("数据获取失败，请先运行 [数据] 拉取行情数据")
        return

    print(f"  沪深300: {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 2. 双均线策略回测（量化入门第一课）
    print("\n[2/3] 运行双均线策略回测（5/20均线交叉）...")
    from backtest.strategies.builtin.ma_cross import MaCrossStrategy
    result = run_backtest(MaCrossStrategy, df, fast=5, slow=20,
                          symbol="沪深300", auto_validate=True)

    # 3. 输出结果
    print("\n[3/3] 回测结果:\n")
    print(generate_report(result))
    print(scorecard(result))

    # 4. 尝试画图
    try:
        print("\n生成权益曲线图...")
        plot_equity_curve(result)
    except Exception:
        print("(权益曲线图生成跳过)")

    print("\n---")
    print("以上是量化交易的最简流程: 数据 → 策略 → 回测 → 报告。")
    print("其他菜单可以帮你: 换策略/优化参数/纸上交易/实盘监控。")
    print("详细使用说明见 readme.md")
    print("---")


def menu_data():
    """拉取数据"""
    print("\n" + "=" * 50)
    print("  拉取行情数据")
    print("=" * 50)

    # M8: 添加 [0] 全部 选项
    ALL_SYMBOLS = ["沪深300", "中证500", "创业板指", "上证50", "上证指数"]
    ALL_SYMBOLS_CHOICES = ["[0] 全部"] + ALL_SYMBOLS
    choice = _pick(ALL_SYMBOLS_CHOICES, "选择指数（[0]=全部5个指数）")

    if "[0]" in choice:
        symbols = ALL_SYMBOLS
    else:
        symbols = [choice]

    start, end = _pick_date_preset("20230101")
    save = _pick(["是（存到CSV，Excel可打开）", "否（只打印到屏幕）"], "保存到文件？")

    from data.fetchers.fallback import fetch_index_daily_safe

    # M10: 拉取完成后显示每个标的数据行数汇总
    summary_rows = []
    for symbol in symbols:
        print(f"\n拉取 {symbol} 数据...")
        df = fetch_index_daily_safe(symbol, start, end)
        summary_rows.append((symbol, len(df), str(df['date'].min().date()), str(df['date'].max().date())))
        print(f"完成！共 {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")
        if len(symbols) == 1:
            print(df.tail(5))

        if "是" in save:
            path = f"notebooks/{symbol}_data.csv"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"已保存: {path}")

    # M10: 多标的汇总表
    if len(summary_rows) > 1:
        print("\n── 拉取汇总 ──")
        print(f"  {'标的':<10} {'行数':>6} {'起始':>12} {'截止':>12}")
        print(f"  {'-'*42}")
        for sym, cnt, d1, d2 in summary_rows:
            print(f"  {sym:<10} {cnt:>6} {d1:>12} {d2:>12}")
        total_rows = sum(c for _, c, _, _ in summary_rows)
        print(f"  {'合计':<10} {total_rows:>6}")


def menu_backtest():
    """单策略回测"""
    print("\n" + "=" * 50)
    print("  单策略回测")
    print("=" * 50)

    from backtest.strategy_market import ALL_STRATEGIES
    strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（元）", 100000)

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.engine.bt_runner import run_backtest
    from backtest.analysis.report import generate_report, plot_equity_curve, scorecard

    info = ALL_STRATEGIES[strategy]
    # M14: 面向用户的来源标签（外来策略/挖掘策略/自建策略）
    from backtest.strategy_market import SOURCE_LABELS
    src_label = SOURCE_LABELS.get(info.get("source", ""), info.get("source", "?"))
    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    print(f"运行 {strategy} 回测 (来源: {src_label})...")

    # 支持两种策略格式：Backtrader类 或 fep包
    if "fep_package" in info:
        result = run_backtest(
            df=df, initial_cash=cash,
            use_universal=True, package_dir=info["fep_package"],
            **info.get("params", {}),
        )
    else:
        result = run_backtest(info["class"], df, initial_cash=cash, **info.get("params", {}))

    print("\n" + generate_report(result))
    print(scorecard(result))

    # M12: 跑完回测后自动展示结果摘要 + 提示打开 Streamlit 仪表板
    print("\n── 结果摘要 ──")
    print(f"  策略: {strategy}")
    print(f"  年化收益: {result.get('annual_return', 0):.2%}")
    print(f"  最大回撤: {result.get('drawdown', 0):.2%}")
    s = result.get("sharpe")
    print(f"  夏普比率: {f'{s:.2f}' if s else 'N/A'}")
    print(f"  交易次数: {result.get('total_trades', 0)}")
    print(f"  胜率: {result.get('win_rate', 0):.1%}")
    print(f"\n  [提示] 运行 Streamlit 仪表板可查看多策略对比、权益曲线和归因分析：")
    print(f"  菜单 [15] 启动监控仪表盘 或直接运行 streamlit run dashboard/app.py")

    plot_want = _pick(["是", "否"], "画权益曲线？")
    if "是" in plot_want:
        path = f"notebooks/{strategy}_equity.png"
        plot_equity_curve(result, save_path=path)
        print(f"图表已保存: {path}")


def menu_shootout():
    """策略大比武"""
    print("\n" + "=" * 50)
    print("  策略大比武（全部策略对比排名）")
    print("=" * 50)
    print("用同一份数据跑全部策略，看哪个最好。")

    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（万）", 10) * 10000

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.shootout import run_shootout

    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    print("回测中...")
    result = run_shootout(df, cash=cash)
    print("\n" + result["summary"])


def menu_market():
    """策略市场扫描"""
    print("\n" + "=" * 50)
    print("  策略市场扫描（社区策略逐个跑）")
    print("=" * 50)
    print("从 vnpy/海龟/聚宽 等开源社区导入策略，" + "\n" +
          "加上内置的 5 个，逐个回测看谁最好。")

    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（万）", 10) * 10000

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_market import scan_market

    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    scan_market(df, cash=cash)


def menu_mine():
    """策略挖掘"""
    print("\n" + "=" * 50)
    print("  策略挖掘（自动排列组合找好策略）")
    print("=" * 50)
    print("把策略拆成「入场规则 + 出场规则 + 参数」，像化学元素一样排列组合，" + "\n" +
          "自动回测筛选，好的留下。")

    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（万）", 10) * 10000
    max_combos = int(_input_num("最多测试多少组合（越大越慢，推荐 50-200）", 50))

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_miner import StrategyMiner

    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    miner = StrategyMiner(df, cash=cash)
    results = miner.mine(max_combinations=max_combos, min_trades=3)
    print("\n" + results["summary"])

    if miner.results:
        save = _pick(["是", "否"], "保存 Top10 策略到 JSON？")
        if "是" in save:
            miner.save_top("notebooks/top_strategies.json")
            print("已保存: notebooks/top_strategies.json")


def menu_paper():
    """纸上交易"""
    print("\n" + "=" * 50)
    print("  纸上交易（模拟真实逐日决策）")
    print("=" * 50)

    from backtest.strategy_market import ALL_STRATEGIES
    strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（元）", 100000)

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_market import ALL_STRATEGIES
    from live.paper_trader import run_paper_trading

    info = ALL_STRATEGIES[strategy]
    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    result = run_paper_trading(
        info["class"], df,
        initial_cash=cash,
        **info.get("params", {}),
    )

    print("\n" + result["summary"])
    if result["warnings"]:
        print(f"\n风控拦截 {len(result['warnings'])} 次:")


def menu_alternative():
    """另类数据扫描"""
    from data.alternative import AlternativeData
    print("\n扫描另类数据（新闻情绪+地缘风险）...")
    ad = AlternativeData()
    result = ad.full_scan()
    print("\n" + result["summary"])


def menu_regime():
    """马尔可夫区制检测"""
    print("\n" + "=" * 50)
    print("  市场区制检测（Markov Regime Detection）")
    print("=" * 50)
    print("分析当前市场处于 Bull/Bear/Sideways 哪个区制，")
    print("为策略提供方向确认和尾部风险过滤。")

    symbol = _pick(["沪深300", "中证500", "创业板指", "上证50"], "选择标的")
    start, end = _pick_date_preset("20200101")

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.regime_filter import get_regime_signal

    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)
    close = df.set_index("date")["close"]

    print("计算区制...")
    sig = get_regime_signal(close)

    print("\n" + "=" * 50)
    print(f"  {symbol} 市场区制分析")
    print("=" * 50)
    print(f"  数据范围: {close.index.min().date()} ~ {close.index.max().date()} ({len(close)}条)")
    print(f"\n  当前区制: {sig['current_regime']}")
    print(f"  区制信号: {sig['signal']:+.4f} (Bull概率 - Bear概率)")
    print(f"  下一日Bull概率: {sig['bull_prob']:.1%}")
    print(f"  下一日Bear概率: {sig['bear_prob']:.1%}")
    print(f"\n  区制粘性（自持概率）:")
    print(f"    Bear持留:     {sig['persistence']['bear']:.1%}")
    print(f"    Sideways持留: {sig['persistence']['sideways']:.1%}")
    print(f"    Bull持留:     {sig['persistence']['bull']:.1%}")
    print(f"\n  长期区制分布（平稳分布）:")
    print(f"    Bear:     {sig['stationary']['bear']:.1%}")
    print(f"    Sideways: {sig['stationary']['sideways']:.1%}")
    print(f"    Bull:     {sig['stationary']['bull']:.1%}")
    print(f"\n  交易判断:")
    print(f"    做多允许: {'是' if sig['long_ok'] else '否'} (Bull区或Sideways+正信号)")
    print(f"    做空允许: {'是' if sig['short_ok'] else '否'} (Bear区或Sideways+负信号)")

    # optional: full report
    if input("\n查看完整转移矩阵？[y/N]: ").strip().lower() == "y":
        from backtest.analysis.markov_regime import analyze
        full = analyze(close, source=symbol)
        P = full["transition_matrix"]
        print(f"\n  转移矩阵 (行=from, 列=to):")
        print(f"              {'Bear':>9s} {'Sideways':>9s} {'Bull':>9s}")
        for i, state in enumerate(["Bear", "Sideways", "Bull"]):
            row = "  ".join(f"{P[i][j] * 100:7.2f}%" for j in range(3))
            print(f"    {state:>9s}  {row}")
        wf = full["walk_forward"]
        if wf["sharpe"] and wf["sharpe"] == wf["sharpe"]:  # not NaN
            print(f"\n  区制信号 Walk-Forward 回测:")
            print(f"    夏普: {wf['sharpe']:.3f}  最大回撤: {wf['max_drawdown']:.2%}  交易: {wf['n_trades']}次")


def menu_agent_decision():
    """Agent 多角色辩论 — L4环境研判"""
    print("\n" + "=" * 60)
    print("  Agent 多角色环境研判（分析-辩论-决策）")
    print("=" * 60)
    print("3位分析师（技术面/因子/情绪）并行 → 研究员辩论 →")
    print("交易员下单 → 风控官审核 → PM最终决策")
    print("─" * 60)
    llm = input("使用 LLM 模式？(需DeepSeek API Key) [y/N]: ").strip().lower() == "y"

    symbol = _pick(["沪深300", "中证500", "创业板指", "上证50"], "选择标的")
    start, end = _pick_date_preset("20240101")

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.agent_decision import AgentDecision

    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)
    if df.empty:
        print("数据为空，退出。")
        return

    print(f"运行 Agent 决策（{'LLM' if llm else '规则引擎'}模式）...")
    try:
        ad = AgentDecision(df)
        result = ad.decide(use_llm=llm)
    except Exception as e:
        print(f"Agent 决策失败: {e}")
        return

    action_label = {"buy": "买入", "sell": "卖出", "hold": "观望"}.get(result["action"], result["action"])
    mode_label = {"rule_based": "规则引擎", "langgraph": "LangGraph状态图",
                  "sequential": "LLM顺序调用"}.get(result.get("mode", ""), result.get("mode", ""))

    print("\n" + "=" * 60)
    print(f"  {symbol} Agent 环境研判结果")
    print("=" * 60)
    print(f"  数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({len(df)}条)")
    print(f"  决策模式: {mode_label}")
    print(f"\n  ┌─ 最终决策 ─────────────────────────────┐")
    print(f"  │  方向: {action_label:<10}  置信度: {result['confidence']:.0%}  仓位: {result['position']:.0%}  │")
    print(f"  └────────────────────────────────────────┘")
    print(f"\n  决策理由: {result.get('reasoning', '无')}")
    if result.get("tech_score"):
        print(f"  技术评分: {result['tech_score']}/100")

    agents = result.get("agents", {})
    if agents:
        print(f"\n  ── 分析师详情 ──")
        for name, info in agents.items():
            if isinstance(info, dict):
                parts = []
                for k, v in info.items():
                    if isinstance(v, bool):
                        parts.append(f"{k}={'是' if v else '否'}")
                    elif isinstance(v, (int, float)):
                        parts.append(f"{k}={v}")
                    elif isinstance(v, str) and len(v) < 50:
                        parts.append(f"{k}={v}")
                print(f"  {name}: {' | '.join(parts) if parts else str(info)[:80]}")

    risk_flags = result.get("risk_flags", [])
    if risk_flags:
        print(f"\n  ⚠ 风险标记 ({len(risk_flags)}项):")
        for f in risk_flags:
            print(f"    - {f}")
    else:
        print(f"\n  ✅ 无风险标记")


def menu_risk():
    """查看/修改风控 — P4增强：查看参数+切换预设+查看覆盖规则"""
    from live.risk.risk_engine import get_params, reload as reload_risk
    from config.loader import get_config

    params = get_params()
    cfg = get_config()
    risk_cfg = cfg.get("risk", {})
    active_preset = risk_cfg.get("active_preset", "neutral")
    presets = risk_cfg.get("presets", {})

    while True:
        print("\n" + "=" * 55)
        print(f"  风控参数管理（当前预设: {active_preset}）")
        print("=" * 55)

        # P4: 查看当前参数
        print(f"\n  ── 核心参数 ──")
        print(f"  单策略仓位上限: {params['max_position_pct']*100:.0f}%")
        print(f"  单笔止损线:     {params['max_single_loss_pct']*100:.1f}%")
        print(f"  日亏损熔断:     {params['max_daily_loss_pct']*100:.1f}%")
        print(f"  总回撤报废:     {params['max_drawdown_pct']*100:.1f}%")
        print(f"  连续亏损暂停:   {params['stop_after_n_losses']} 笔")
        print(f"  使用杠杆:       {'否' if params.get('no_leverage', True) else '是'}")

        # P3: 补充参数（如果存在）
        extra_params = ["max_symbols", "max_single_symbol_pct", "max_daily_trades",
                        "vol_adaptive_position", "correlation_limit", "liquidity_min_volume"]
        extra_available = [k for k in extra_params if k in params]
        if extra_available:
            print(f"\n  ── 补充参数 ──")
            for k in extra_available:
                print(f"  {k}: {params[k]}")

        # P5: 全局开关
        print(f"\n  ── 全局开关 ──")
        print(f"  风控引擎:       {'开' if risk_cfg.get('enable_risk_engine', True) else '关'}")
        print(f"  自动熔断:       {'开' if risk_cfg.get('auto_circuit_breaker', True) else '关'}")
        print(f"  熔断通知:       {'开' if risk_cfg.get('notify_on_breach', True) else '关'}")
        print(f"  熔断冷却:       {risk_cfg.get('breach_cooldown_h', 4)}h")

        # 三套预设对比
        if presets:
            print(f"\n  ── 三套预设对比 ──")
            print(f"  {'参数':<22} {'保守':>8} {'中性':>8} {'激进':>8}")
            print(f"  {'-'*46}")
            for key in ["max_position_pct", "max_single_loss_pct", "max_daily_loss_pct",
                         "max_drawdown_pct", "stop_after_n_losses", "max_symbols"]:
                c = presets.get("conservative", {}).get(key, "-")
                n = presets.get("neutral", {}).get(key, "-")
                a = presets.get("aggressive", {}).get(key, "-")
                # 格式化显示
                if isinstance(c, float):
                    cstr = f"{c*100:.0f}%"
                elif isinstance(c, bool):
                    cstr = "是" if c else "否"
                else:
                    cstr = str(c)
                if isinstance(n, float):
                    nstr = f"{n*100:.0f}%"
                elif isinstance(n, bool):
                    nstr = "是" if n else "否"
                else:
                    nstr = str(n)
                if isinstance(a, float):
                    astr = f"{a*100:.0f}%"
                elif isinstance(a, bool):
                    astr = "是" if a else "否"
                else:
                    astr = str(a)
                print(f"  {key:<22} {cstr:>8} {nstr:>8} {astr:>8}")

        # 操作菜单
        print(f"\n  ── 操作 ──")
        print(f"  [1] 切换预设（conservative/neutral/aggressive）")
        print(f"  [2] 查看覆盖规则（settings.local.yaml 优先级）")
        print(f"  [3] 重新加载配置")
        print(f"  [0] 返回")

        c = input("\n> ").strip()
        if c == "0":
            break
        elif c == "1":
            choice = _pick(["conservative（保守）", "neutral（中性）", "aggressive（激进）"],
                          "选择预设")
            preset_name = choice.split("（")[0]
            print(f"\n已选择预设: {preset_name}")
            print("请在 config/settings.yaml 中修改 risk.active_preset 后重启生效。")
            print(f"当前预设覆盖: 编辑 settings.yaml → risk.active_preset: {preset_name}")
        elif c == "2":
            print("\n── 覆盖规则说明 ──")
            print("配置加载顺序: settings.yaml（默认值）→ settings.local.yaml（本地覆盖）")
            print("如果 settings.local.yaml 中定义了 risk 段，会覆盖 settings.yaml 的同名参数。")
            print("预设优先级: active_preset 指定的预设 > 直接写在 risk 下的参数（向后兼容）。")
            print("test/live 环境共用同一 active_preset；demo 环境可通过 settings.local.yaml 独立覆盖。")
        elif c == "3":
            reload_risk()
            params = get_params()
            print("配置已重新加载。")


def menu_repo_mgmt():
    """策略仓库管理（含导入/同步/爬取）"""
    from strategies_repo.repo import StrategyRepo

    while True:
        repo = StrategyRepo()
        stats = repo.stats()
        print("\n" + "=" * 50)
        print(f"  策略仓库管理（共 {stats['total']} 个策略）")
        print("=" * 50)
        actions = [
            ("列出全部策略", lambda: _repo_list(repo)),
            ("搜索策略", lambda: _repo_search(repo)),
            ("策略对比（选几个跑回测排名）", lambda: _repo_compare(repo)),
            ("导入策略（vnpy/Freqtrade/TradingView）", _menu_import),
            ("策略源更新检查", _menu_sync),
        ]
        for i, (desc, _) in enumerate(actions, 1):
            print(f"  [{i}] {desc}")
        print(f"  [0] 返回主菜单")
        try:
            n = int(input("\n> "))
        except ValueError:
            continue
        if n == 0:
            break
        elif 1 <= n <= len(actions):
            try:
                actions[n - 1][1]()
            except Exception as e:
                print(f"出错: {e}")
        else:
            print("超出范围")


def _menu_import():
    """导入子菜单 — 本地路径优先，远程兜底"""
    from strategies_repo.importer import (
        import_from_vnpy, import_from_gitee_vnpy, import_from_freqtrade,
        import_from_local, import_all, _verify_repo_url,
    )

    import os
    WS = "D:/workspace/workspace-python"

    # 检测本地仓库
    LOCAL_REPOS = []
    for label, dirname in [
        ("vnpy_ctastrategy", "vnpy_ctastrategy"),
        ("freqtrade-strategies", "freqtrade-strategies"),
        ("vnpy", "vnpy"),
    ]:
        path = os.path.join(WS, dirname)
        if os.path.isdir(path):
            n = _count_strategy_files(path, deep=True)
            LOCAL_REPOS.append((label, path, n))

    print("\n" + "=" * 40)
    print("  策略导入")
    print("=" * 40)

    if LOCAL_REPOS:
        print("── 本地（已拉好的仓库，秒导）──")
        for i, (label, path, cnt) in enumerate(LOCAL_REPOS, 1):
            print(f"  [{i}] {label:<22} → {path} ({cnt}个策略)")

    from config.loader import is_crypto_enabled
    print("── 远程（自动 git clone）──")
    print("  [A] vnpy GitHub    (需VPN, ~20策略)")
    print("  [B] vnpy Gitee     (国内直连)")
    if is_crypto_enabled():
        print("  [C] Freqtrade 社区 (需VPN)")
    print("  [D] TradingView    (需代理)")
    print("  [V] 验证远程仓库")
    print("  [0] 返回")
    c = input("> ").strip().upper()

    # 本地选项：数字 1-N
    if c.isdigit():
        idx = int(c) - 1
        if 0 <= idx < len(LOCAL_REPOS):
            label, path, cnt = LOCAL_REPOS[idx]
            n = import_from_local(path, label)
            print(f"{label} 导入完成: {n} 个策略")
            return

    if c == "A":
        n = import_from_vnpy()
        print(f"vnpy GitHub 导入完成: {n} 个策略")
    elif c == "B":
        n = import_from_gitee_vnpy()
        print(f"Gitee vnpy 导入完成: {n} 个策略")
    elif c == "C":
        n = import_from_freqtrade()
        print(f"Freqtrade 导入完成: {n} 个策略")
    elif c == "D":
        from strategies_repo.crawler import crawl_tradingview_top
        top_n = input("获取 Top N 个高分策略 [20]: ").strip()
        top_n = int(top_n) if top_n.isdigit() else 20
        n = crawl_tradingview_top(top_n=top_n, min_rating=0)
        print(f"TradingView 导入完成: {n} 个策略")
    elif c == "V":
        print("\n验证远程仓库...")
        repos = [
            ("vnpy (GitHub)", "https://github.com/vnpy/vnpy_ctastrategy.git"),
            ("vnpy (Gitee)", "https://gitee.com/vnpy/vnpy.git"),
            ("freqtrade", "https://github.com/freqtrade/freqtrade-strategies.git"),
        ]
        for name, url in repos:
            ok = _verify_repo_url(url)
            print(f"  {name:<20} {'OK' if ok else 'FAIL'}")
        print("验证完成")


def _count_strategy_files(path: str, deep: bool = False) -> int:
    """统计目录下策略文件数。deep=True时遍历子目录。"""
    import os
    count = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ("venv", "__pycache__", ".git", ".idea")]
        for f in files:
            if f.endswith(".py") and not f.startswith("_") and f != "__init__.py":
                if deep:
                    # 读文件确认含策略类
                    try:
                        with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        if "class " in content and ("Strategy" in content or "IStrategy" in content):
                            count += 1
                    except Exception:
                        pass
                else:
                    count += 1
        if not deep:
            break  # 浅层：只查第一级目录
    return count


def _menu_sync():
    """同步子菜单"""
    from strategies_repo.sync import StrategySync
    sync = StrategySync()
    print(sync.status())
    updates = sync.check_updates()
    if updates:
        print(f"\n{len(updates)} 个源有更新，是否立即同步？[y/N]")
        if input("> ").strip().lower() == "y":
            results = sync.sync_all()
            print(f"同步完成: {results}")
    else:
        print("所有源都是最新的")


def _repo_list(repo):
    """分类展示策略库，高分优先"""
    from strategies_repo.scoring import rank_strategies

    all_s = repo.list()
    stats = repo.stats()
    total = stats["total"]

    # 1. 总览
    print(f"\n  共 {total} 个策略")
    by_cat = stats.get("by_category", {})
    for cat, cnt in sorted(by_cat.items()):
        print(f"    {cat:<10} {cnt:>4} 个")
    by_type = stats.get("by_type", {})
    if by_type:
        types_str = ", ".join(f"{t}:{c}" for t, c in sorted(by_type.items(), key=lambda x: -x[1]))
        print(f"    类型分布: {types_str}")

    if total == 0:
        print("  (空仓库，请先导入策略)")
        return

    # 2. 按评分排名，展示 Top N
    ranked = rank_strategies(all_s)
    # 分两拨：有回测结果的 vs 未回测的
    scored = [s for s in ranked if s.get("metrics", {}).get("annual_return") is not None]
    unscored = [s for s in ranked if s not in scored]

    if scored:
        top_n = min(5, len(scored))
        print(f"\n  ── 已回测 Top {top_n}（按综合评分）──")
        print(f"  {'排名':<4} {'策略':<18} {'来源':<16} {'评分':>6} {'年化':>6}")
        print(f"  {'-' * 52}")
        for i, s in enumerate(scored[:top_n]):
            m = s.get("metrics", {})
            print(f"  {i+1:<4} {s['name'][:17]:<18} {s.get('source','')[:15]:<16} "
                  f"{s.get('_score', 0):>6.3f} {m.get('annual_return', 0):>5.1%}")

    if unscored:
        print(f"\n  ── 待回测: {len(unscored)} 个策略 ──")
        # 展示前3个
        for s in unscored[:3]:
            print(f"  · {s['name'][:30]:<30} [{s.get('source', '?')}]")
        if len(unscored) > 3:
            print(f"  ... 还有 {len(unscored) - 3} 个，用 [2] 搜索或 [3] 回测后见排名")

    if not scored and unscored:
        print(f"\n  提示: 所有策略尚未回测。用 [3] 策略对比来跑回测，出分后这里会按排名展示。")


def _repo_search(repo):
    kw = input("搜索关键词: ").strip()
    if not kw:
        return
    results = repo.search(kw)
    if results:
        print(f"找到 {len(results)} 个:")
        for r in results:
            print(f"  - {r['name']} [{r.get('type', '')}] {r.get('description', '')}")
    else:
        print("未找到")


def _repo_compare(repo):
    from data.fetchers.fallback import fetch_index_daily_safe
    names_input = input("输入策略名（多个用逗号分隔，回车=全部）: ").strip()
    if names_input:
        names = [n.strip() for n in names_input.split(",")]
    else:
        names = [s["name"] for s in repo.list()]

    symbol = _pick(["沪深300", "中证500"], "标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（万）", 10) * 10000

    print(f"拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    print(f"对比 {len(names)} 个策略...")
    results = repo.compare(names, df, cash)

    print(f"\n{'排名':<4} {'策略':<14} {'年化':>6} {'回撤':>6} {'夏普':>6} {'交易':>4}")
    print("-" * 50)
    for i, r in enumerate(results):
        s = r.get("sharpe")
        print(f"{i+1:<4} {r['name']:<14} {r['annual_return']:>5.1%} {r['drawdown']:>5.1%} "
              f"{f'{s:.2f}' if s else 'N/A':>6} {r['trades']:>4}")



def _build_param_space(strategy_name: str, scale: float = 0.5) -> dict:
    """根据策略默认参数构建搜索空间"""
    from backtest.strategy_market import ALL_STRATEGIES
    params = ALL_STRATEGIES[strategy_name].get("params", {})
    space = {}
    for k, v in params.items():
        if isinstance(v, (int, float)):
            if isinstance(v, int) or v == int(v):
                low = max(2, int(v * (1 - scale)))
                high = int(v * (1 + scale)) + 1
                space[k] = (low, high)
            else:
                space[k] = (v * (1 - scale), v * (1 + scale))
    return space


def menu_ga():
    """遗传算法优化（研究/实验用，Bayes通常更优）"""
    from backtest.strategy_market import ALL_STRATEGIES
    print("\n" + "=" * 50)
    print("  遗传算法优化（实验 — 推荐用[4]贝叶斯）")
    print("=" * 50)
    print("GA 适合离散/组合参数空间，Bayes 通常更快更准。")
    print("两者解决同一问题(参数优化)，日常使用推荐 Bayes。")
    strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    pop = int(_input_num("种群大小（推荐20-50）", 30))
    gen = int(_input_num("进化代数（推荐10-30）", 20))

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.genetic_miner import GeneticOptimizer

    info = ALL_STRATEGIES[strategy]
    param_space = _build_param_space(strategy)
    start, end = _pick_date_preset("20180101")
    print("  [注意] 全周期数据量大会导致优化耗时较长")
    df = fetch_index_daily_safe(symbol, start, end)
    go = GeneticOptimizer(df, cash=100000)
    best = go.evolve(info["class"], param_space=param_space, pop_size=pop, generations=gen)
    print(f"\n最优参数: {best['params']}")
    print(f"夏普={best['sharpe']:.2f}  年化={best['annual_return']:.2%}  回撤={best['drawdown']:.2%}")


def menu_bayes():
    """贝叶斯优化（推荐 — 已接入策略挖掘精调）"""
    from backtest.strategy_market import ALL_STRATEGIES
    print("\n" + "=" * 50)
    print("  贝叶斯优化（推荐 — 概率模型搜索最优参数）")
    print("=" * 50)
    print("已接入策略挖掘器：mine()后自动对Top3做Bayes精调。")
    print("比网格搜索更智能，支持剪枝+参数重要性分析。")
    strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    trials = int(_input_num("采样次数（推荐30-100）", 50))

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.bayesian_opt import BayesianOptimizer

    info = ALL_STRATEGIES[strategy]
    param_space = _build_param_space(strategy)
    start, end = _pick_date_preset("20180101")
    print("  [注意] 全周期数据量大会导致优化耗时较长")
    df = fetch_index_daily_safe(symbol, start, end)
    bo = BayesianOptimizer(df, cash=100000)
    best = bo.optimize(info["class"], param_space=param_space, n_trials=trials)
    print(f"\n最优参数: {best['params']}")
    print(f"夏普={best['sharpe']:.2f}  年化={best['annual_return']:.2%}  回撤={best['drawdown']:.2%}")
    imp = bo.get_importance()
    if imp:
        print(f"参数重要性: {imp}")


def menu_rl():
    """强化学习"""
    print("\n" + "=" * 50)
    print("  强化学习训练（PPO智能体）")
    print("=" * 50)
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    steps = int(_input_num("训练步数（推荐30000-100000）", 50000))

    from data.fetchers.fallback import fetch_index_daily_safe
    from experimental.rl_trader import RLTrainer

    start, end = _pick_date_preset("20180101")
    print("  [注意] 全周期数据量大会导致训练耗时较长")
    df = fetch_index_daily_safe(symbol, start, end)
    trainer = RLTrainer(df, cash=100000)
    result = trainer.train(timesteps=steps)
    print(f"\n测试收益: {result['test_return']:.2%} (买入持有: {result['buy_hold_return']:.2%})")
    print(f"夏普: {result['test_sharpe']:.2f}  交易: {result['n_trades']}次")


def menu_health():
    """数据源健康检查"""
    from data.fetchers.fallback import get_source_health, check_dependencies, check_data_completeness
    from data.fetchers.fallback import fetch_index_daily_safe

    print("\n" + "=" * 50)
    print("  数据源健康检查")
    print("=" * 50)

    # 依赖检测
    deps = check_dependencies()
    print("\n── 依赖可用性 ──")
    for src, available in deps.items():
        icon = "OK" if available else "MISSING"
        print(f"  {src:12} {icon}")

    # 实际拉取测试
    print("\n── 拉取测试（自动降级）──")
    df = fetch_index_daily_safe("沪深300",
                                start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                                end_date=datetime.now().strftime("%Y%m%d"))
    if not df.empty:
        src = df.attrs.get("source", "unknown")
        print(f"  数据源: {src}, {len(df)} 条")
        completeness = check_data_completeness(df)
        if completeness["complete"]:
            print(f"  完整性: OK（最新 {completeness['last_date']}）")
        else:
            print(f"  完整性: 告警 — {completeness['warning']}")

    # 历史健康状态
    print("\n── 各数据源状态 ──")
    health = get_source_health()
    for src, h in health.items():
        icon = "OK" if h["status"] == "ok" else ("DOWN" if h["status"] == "down" else "??")
        last = h.get("last_check", "never")
        err = f" ({h['error'][:60]})" if h.get("error") else ""
        print(f"  {src:12} {icon}  last_check: {last}{err}")


def menu_minute_backtest():
    """分钟级回测（需 Tushare Pro Token，¥200/年）"""
    from config.loader import get_tushare_token

    print("\n" + "=" * 50)
    print("  分钟级回测（日内策略）")
    print("=" * 50)

    token = get_tushare_token()
    if not token or token == "你的Tushare Token":
        print("\n[Tushare Token 未配置]")
        print("分钟级数据需要 Tushare Pro（¥200/年），请先:")
        print("  1. 注册 https://tushare.pro → 获取 Token")
        print("  2. 填入 config/settings.local.yaml:")
        print("     data_sources:")
        print('       tushare:')
        print('         token: "你的Token"')
        print("  3. 重新运行本菜单")
        return

    from data.fetchers.tushare_fetch import fetch_minute_kline
    from backtest.strategy_market import ALL_STRATEGIES

    strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
    symbol = _pick(["沪深300", "中证500", "创业板指", "上证50"], "选择标的")
    freq = _pick(["1min", "5min", "15min", "30min", "60min"], "K线周期")
    start = _input_str("开始日期 YYYYMMDD", "20250101")
    end = _input_str("结束日期 YYYYMMDD（Tushare分钟线限最近30天）", "")
    cash = _input_num("初始资金（元）", 100000)

    from backtest.engine.bt_runner import run_backtest
    from backtest.analysis.report import generate_report, plot_equity_curve, scorecard

    ts_code_map = {"沪深300": "000300.SH", "中证500": "000905.SH", "创业板指": "399006.SZ", "上证50": "000016.SH"}
    ts_code = ts_code_map[symbol]

    print(f"\n拉取 {symbol}({ts_code}) {freq} 分钟线...")
    df = fetch_minute_kline(ts_code, freq=freq, start_date=start, end_date=end)

    if df is None or df.empty:
        print("拉取失败。Tushare 分钟线可能原因:")
        print("  - Token 未激活或已过期")
        print("  - 时间范围超出限制（免费版限最近30天）")
        print("  - 网络问题")
        return

    # 列名统一（Tushare分钟线列名: ts_code, trade_time, open, high, low, close, vol, amount）
    if "trade_time" in df.columns:
        df = df.rename(columns={"trade_time": "date", "vol": "volume"})

    print(f"完成！共 {len(df)} 条 {freq} K线, {df['date'].min()} ~ {df['date'].max()}")

    info = ALL_STRATEGIES[strategy]
    print(f"\n运行 {strategy} {freq} 回测...")
    result = run_backtest(info["class"], df, initial_cash=cash, **info.get("params", {}))

    print("\n" + generate_report(result))
    print(scorecard(result))

    plot_want = _pick(["是", "否"], "画权益曲线？")
    if "是" in plot_want:
        path = f"notebooks/{strategy}_{freq}_equity.png"
        plot_equity_curve(result, save_path=path)
        print(f"图表已保存: {path}")


def menu_notify_test():
    """通知渠道测试"""
    from live.gateway.notifier import send, _load_notify_config

    print("\n" + "=" * 50)
    print("  通知渠道测试")
    print("=" * 50)

    cfg = _load_notify_config()
    channels = [
        ("钉钉", cfg.get("dingtalk_webhook")),
        ("飞书", cfg.get("feishu_webhook")),
        ("PushPlus", cfg.get("pushplus_token")),
        ("Server酱", cfg.get("server_chan_sendkey")),
        ("Telegram", cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id")),
        ("企业微信", cfg.get("wecom_webhook")),
    ]

    available = []
    print("\n── 渠道状态 ──")
    for i, (name, configured) in enumerate(channels, 1):
        status = "已配置" if configured else "未配置"
        print(f"  [{i}] {name:12} {status}")
        if configured:
            available.append((i, name))

    if not available:
        print("\n所有渠道未配置。请在 config/settings.local.yaml 中添加 notification 配置。")
        print("推荐: 钉钉群机器人（免费不限量）或 PushPlus（微信推送，200条/天）")
        return

    # M7: 列出渠道 + 允许多选
    print(f"\n可用渠道: {', '.join(n for _, n in available)}")
    print("输入渠道编号（多选用逗号分隔，如 1,3,5），回车 = 全部可用渠道")
    choice = input("> ").strip()

    selected = []
    if choice == "":
        selected = [n for _, n in available]  # 全选
    else:
        available_map = {i: n for i, n in available}
        for part in choice.split(","):
            try:
                idx = int(part.strip())
                if idx in available_map:
                    selected.append(available_map[idx])
            except ValueError:
                pass

    if not selected:
        print("未选择任何可用渠道")
        return

    print(f"\n将发送测试消息到: {', '.join(selected)}")
    test_msg = (
        f"**QuantP 通知测试**\n\n"
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"测试渠道: {', '.join(selected)}\n\n"
        f"你的通知配置正常！"
    )
    for ch_name in selected:
        print(f"  发送到 {ch_name}...")
        ok = send(test_msg, channel="auto", title="QuantP 通知测试")
        print(f"  {ch_name}: {'成功' if ok else '失败（检查网络或配置）'}")


def menu_paper_league():
    """纸上联赛：多策略并行竞赛"""
    print("\n" + "=" * 50)
    print("  纸上联赛（多策略并行竞赛）")
    print("=" * 50)

    from backtest.strategy_market import ALL_STRATEGIES
    names = list(ALL_STRATEGIES.keys())
    chosen = []
    while True:
        remaining = [n for n in names if n not in chosen]
        if not remaining:
            break
        s = _pick(remaining + ["── 选完了 ──"], f"选择参赛策略（已选 {len(chosen)} 个）")
        if "选完" in s:
            break
        chosen.append(s)

    if len(chosen) < 2:
        print("至少需要2个策略才能开赛。")
        return

    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    cash = _input_num("每策略初始资金（元）", 100000)

    from data.fetchers.fallback import fetch_index_daily_safe
    from live.paper_league import PaperLeague, LeagueConfig

    start, end = _pick_date_preset("20240101")
    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)

    strategies = {}
    for name in chosen:
        info = ALL_STRATEGIES[name]
        strategies[name] = (info["class"], info.get("params", {}))

    config = LeagueConfig(
        strategies=strategies,
        symbol=symbol,
        initial_cash_per_strategy=cash,
        data_df=df,
    )
    league = PaperLeague(config)
    league.run_history()
    league.print_leaderboard()


def menu_paper_diff():
    """纸上交易 vs 回测对比"""
    print("\n" + "=" * 50)
    print("  纸上交易 vs 回测 差异分析")
    print("=" * 50)

    from live.execution.paper_store import PaperTradeStore
    from live.analysis.paper_diff import compare_paper_backtest, generate_diff_text

    store = PaperTradeStore()
    instances = store.list_instances(20)
    if not instances:
        print("尚无纸上交易实例。请先在 [10] 纸上交易 中运行一次。")
        return

    print(f"\n已有实例（最近 {len(instances)} 个）:")
    for inst in instances[:10]:
        print(f"  #{inst['id']} {inst['strategy']} [{inst['status']}] "
              f"{inst.get('started_at', '')[:19]}")
    store.close()

    iid = int(_input_num("选择实例ID", 1))

    # 需要用户提供对应回测结果
    print("\n对比需要同策略/同期回测结果。")
    print("请确保已在 [5] 单策略回测 中运行过同一策略。")

    from backtest.strategy_market import ALL_STRATEGIES
    strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")

    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.engine.bt_runner import run_backtest

    start, end = _pick_date_preset("20240101")
    print(f"\n拉取数据并运行回测...")
    df = fetch_index_daily_safe(symbol, start, end)
    info = ALL_STRATEGIES[strategy]
    bt_result = run_backtest(info["class"], df, initial_cash=100000, **info.get("params", {}))

    # 加载纸上结果
    store2 = PaperTradeStore()
    paper_data = store2.load_instance(iid)
    store2.close()

    if not paper_data:
        print(f"实例 #{iid} 不存在。")
        return

    # 从 daily_log 重建 paper_result
    paper_result = {
        "strategy": paper_data.get("strategy", ""),
        "initial_cash": paper_data.get("initial_cash", 100000),
        "signals": [{"date": l.get("date"), "signal": l.get("signal")}
                     for l in paper_data.get("daily_log", [])
                     if l.get("signal")],
        "orders": paper_data.get("orders", []),
        "daily_log": paper_data.get("daily_log", []),
    }

    report = compare_paper_backtest(paper_result, bt_result, strategy, symbol)
    print("\n" + generate_diff_text(report))


def menu_paper_daemon():
    """纸上守护进程：长期运行管理"""
    print("\n" + "=" * 50)
    print("  纸上守护进程（长期运行+自动恢复）")
    print("=" * 50)

    from live.paper_daemon import PaperDaemon
    from live.execution.paper_store import PaperTradeStore

    store = PaperTradeStore()
    stats = store.stats()
    print(f"\n存储状态: {stats['total_instances']} 个实例, "
          f"{stats['running']} 个运行中, {stats['total_daily_logs']} 条日志")

    action = _pick([
        "查看运行中实例",
        "恢复未完成实例",
        "启动新实例",
        "查看排行榜",
        "停止实例",
        "返回",
    ], "选择操作")

    if "查看运行" in action:
        running = store.load_running_instances()
        if running:
            for inst in running:
                print(f"  #{inst['id']} {inst['strategy']} ({inst.get('symbol', '')}) "
                      f"从 {inst.get('started_at', '')[:19]}")
        else:
            print("  无运行中实例。")

    elif "恢复" in action:
        daemon = PaperDaemon()
        daemon.resume_all()
        print(f"已恢复 {len(daemon.active_traders)} 个实例。")

    elif "启动" in action or "新实例" in action:
        from backtest.strategy_market import ALL_STRATEGIES
        strategy = _pick(list(ALL_STRATEGIES.keys()), "选择策略")
        symbol = _pick(["000300", "000905", "000016", "399006"], "选择标的")
        cash = _input_num("初始资金（元）", 100000)

        daemon = PaperDaemon()
        info = ALL_STRATEGIES[strategy]
        iid = daemon.start_instance(strategy, symbol, cash, info.get("params", {}))
        print(f"实例 #{iid} 已创建: {strategy} ({symbol})")

    elif "排行" in action:
        lb = store.get_leaderboard(10)
        if not lb.empty:
            print("\n排行榜:")
            for _, row in lb.iterrows():
                blown = "⚠️" if row.get("blown") else "✅"
                print(f"  #{row.get('id')} {row.get('strategy', ''):20s} "
                      f"{row.get('total_return', 0):+.2%} {blown}")

    elif "停止" in action:
        running = store.load_running_instances()
        if running:
            names = [f"#{inst['id']} {inst['strategy']}" for inst in running]
            sel = _pick(names + ["返回"], "选择要停止的实例")
            if "返回" not in sel:
                iid = int(sel.split("#")[1].split()[0])
                store.close_instance(iid, "stopped")
                print(f"实例 #{iid} 已停止。")

    store.close()


def menu_live_crypto():
    """CCXT 加密货币实盘（默认测试网）-- 已禁用"""
    from config.loader import is_crypto_enabled
    if not is_crypto_enabled():
        print("\n加密货币实盘已禁用。如需启用，在 config/settings.yaml 中设置 markets.crypto=true")
        return
    from live.gateway.ccxt_gateway import CCXTGateway

    print("\n" + "=" * 60)
    print("  CCXT 加密货币实盘交易")
    print("=" * 60)
    print()
    print("  [安全提醒] 当前默认连接 Binance 测试网（假钱）。")
    print("  要在测试网下单，需要先注册并配置 API Key：")
    print("    1. 注册 https://testnet.binance.vision/")
    print("    2. 生成 API Key + Secret")
    print("    3. 填入 config/settings.local.yaml:")
    print("       ccxt:")
    print("         api_key: \"你的Key\"")
    print("         secret: \"你的Secret\"")
    print()
    print("  切换到真实交易需要改代码中的 testnet=False。")
    print("  正式真金白银前必须先纸上交易验证 4 周。")
    print("=" * 60)

    if input("\n连接 Binance 测试网（无需 API Key 也能看行情）？[y/N]: ").strip().lower() != "y":
        return

    gw = CCXTGateway(exchange="binance", testnet=True)
    if gw.connect():
        print("\n连接成功！")
        ticker = gw.get_ticker("BTC/USDT")
        if ticker:
            print(f"BTC/USDT: ${ticker['last']:,.2f}")
            print(f"  买一: ${ticker['bid']:,.2f}  卖一: ${ticker['ask']:,.2f}")
            print(f"  24h涨跌: {ticker.get('percentage', 0):.2f}%")

        # 如果配置了 API Key，可以查余额和尝试下单
        try:
            balance = gw.get_balance("USDT")
            if balance > 0:
                print(f"\nUSDT 余额: {balance}")
                if input("尝试模拟下单 0.001 BTC？[y/N]: ").strip().lower() == "y":
                    order = gw.market_buy("BTC/USDT", 0.001)
                    if order:
                        print(f"订单: {order}")
                    else:
                        print("下单失败（可能需要 API Key）")
        except Exception:
            pass

        gw.disconnect()
    else:
        print("连接失败（需要网络访问 Binance）")


def menu_live_vnpy():
    """vnpy A股/期货实盘（需 SimNow 账号）"""
    from live.gateway.vnpy_gateway import VnpyGateway, VNPY_CONFIG_TEMPLATE

    print("\n" + "=" * 60)
    print("  vnpy A股/期货实盘交易")
    print("=" * 60)
    print()
    print("  [安全提醒] vnpy 需要实盘/仿真账户才能真实下单。")
    print("  推荐先注册 SimNow（免费期货仿真）:")
    print("    1. 注册 https://www.simnow.com.cn/")
    print("    2. 拿到 broker_id / user_id / password")
    print("    3. 填入 config/settings.local.yaml")
    print()
    print("  当前如果 vnpy 未安装或未配置，自动降级为骨架模式（不下单）。")
    print("=" * 60)

    if input("\n启动 vnpy 网关？[y/N]: ").strip().lower() != "y":
        return

    gw = VnpyGateway()
    if gw.connect():
        print(f"\n连接成功！模式: {gw.mode}")

        balance = gw.get_balance()
        print(f"可用资金: {balance:,.0f}")

        positions = gw.get_positions()
        if positions:
            print(f"持仓 {len(positions)} 个:")
            for p in positions:
                print(f"  {p.get('symbol', '?')}: {p.get('volume', 0)}手 @ {p.get('price', 0)}")
        else:
            print("当前空仓")

        if gw.mode == "skeleton":
            print(f"\n配置模板（粘贴到 settings.local.yaml 后重新连接）:")
            print(VNPY_CONFIG_TEMPLATE)

        gw.disconnect()
    else:
        print("连接失败。")


def menu_guardian():
    """启动守护进程"""
    from live.monitor.guardian import Guardian

    print("\n" + "=" * 60)
    print("  策略守护进程")
    print("=" * 60)
    print("  功能:")
    print("    1. 数据源健康监控（自动切换备用源）")
    print("    2. 策略漂移检测（实盘信号偏离回测预期时告警）")
    print("    3. 紧急熔断（一键停止所有策略）")
    print("    4. 每日自动任务（拉数据 + 健康检查）")
    print("=" * 60)

    g = Guardian()
    status = g.status()
    print(f"\n  实盘模式: {'是（真金白银！）' if status['live_mode'] else '否（安全）'}")
    print(f"  数据源健康: {'OK' if status['data_health'] else '待检查'}")
    print(f"  上次熔断: {status.get('last_emergency', '无')}")

    print("\n操作:")
    print("  [1] 进入实盘模式（需要确认）")
    print("  [2] 运行数据源健康检查")
    print("  [3] 执行每日定时任务")
    print("  [4] 紧急熔断（停止一切）")
    print("  [0] 返回")

    c = input("\n> ").strip()
    if c == "1":
        g.enter_live_mode()
    elif c == "2":
        health = g.check_data_health()
        print(f"\n整体健康: {'OK' if health['all_healthy'] else '异常'}")
        for src, info in health.get("sources", {}).items():
            print(f"  {src}: {info.get('status', '?')}")
    elif c == "3":
        g.schedule_daily()
    elif c == "4":
        g.emergency_stop("用户手动触发")


def menu_vault():
    """本地数据仓库总览"""
    from data.vault import MarketVault, FactorStore, ParamStore, BacktestStore

    print("\n" + "=" * 60)
    print("  本地数据仓库总览")
    print("=" * 60)

    # 市场数据仓库
    mv = MarketVault()
    print("\n" + mv.info_text())

    # 因子库
    fs = FactorStore()
    print("\n" + fs.info_text())

    # 参数库
    ps = ParamStore()
    print("\n" + ps.info_text())

    # 回测结果库
    bs = BacktestStore()
    print("\n" + bs.info_text())

    # 操作菜单
    print("\n操作:")
    print("  [1] 增量更新所有已缓存指数")
    print("  [2] 拉取新指数（添加到仓库）")
    print("  [3] 删除旧回测记录（>90天）")
    print("  [0] 返回")

    c = input("\n> ").strip()
    if c == "1":
        results = mv.update_all_indexes()
        for r in results:
            print(f"  {r.get('new_rows', 0)} 条新数据")
        print(f"共更新 {len(results)} 个指数")
    elif c == "2":
        name = input("指数名（如 中证1000/科创50）: ").strip()
        if name:
            df = mv.get_index(name)
            print(f"已缓存: {name}, {len(df)} 条")
    elif c == "3":
        bs = BacktestStore()
        n = bs.delete_old(90)
        print(f"已删除 {n} 条旧记录")


def menu_full_backtest():
    """全量回测 — 策略库里所有策略全跑一遍"""
    from backtest.strategy_market import ALL_STRATEGIES
    from data.vault import MarketVault, BacktestStore

    print("\n" + "=" * 60)
    print("  全量回测 — 所有策略 + 自动存档")
    print("=" * 60)

    name_count = len(ALL_STRATEGIES)
    # 估算耗时（每个策略约2秒走Backtrader）
    est_seconds = name_count * 2
    est_minutes = est_seconds / 60
    print(f"策略库中共 {name_count} 个策略")
    if est_minutes >= 1:
        print(f"预计耗时: ~{est_minutes:.0f} 分钟（Backtrader，{name_count}×2s）")
    else:
        print(f"预计耗时: ~{est_seconds} 秒（Backtrader）")

    # M4: 首次进入提醒
    print("\n  [!] 提示：")
    print("  - 全量回测会逐一运行所有策略，耗时较长")
    print("  - 期间请勿关闭终端，完成后会自动输出 Top 10 排名")
    print("  - 结果会自动存档到 BacktestStore（SQLite），可后续对比演变趋势")
    print("  - 如需加速，后续可启用 VectorBT（需安装 vectorbt 包）")

    # M2: 估算耗时后加确认
    symbol = _pick(["沪深300", "中证500", "创业板指"], "选择标的")
    start, end = _pick_date_preset("20200101")
    cash = _input_num("初始资金（万）", 10) * 10000

    # 尝试用 VectorBT 加速
    try:
        import vectorbt
        use_vb = True
        print(f"VectorBT 可用 — 预计 ~{max(name_count // 10, 1)} 秒")
    except ImportError:
        use_vb = False

    ans = input(f"\n确认全量回测 {name_count} 个策略？(y/n): ").strip().lower()
    if ans != "y":
        print("已取消")
        return

    mv = MarketVault()
    print(f"\n获取 {symbol} 数据...")
    df = mv.get_index(symbol, start, end)
    print(f"共 {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")

    from backtest.engine.bt_runner import run_backtest
    from backtest.strategy_miner import score_strategy

    results = []
    total = len(ALL_STRATEGIES)
    for i, (name, info) in enumerate(ALL_STRATEGIES.items(), 1):
        try:
            r = run_backtest(
                info["class"], df, initial_cash=cash,
                auto_save=True, symbol=symbol,
                **info.get("params", {}),
            )
            score = score_strategy(r)
            results.append({"name": name, **r, "_score": score})
            print(f"  [{i}/{total}] {name:<14} 年化={r['annual_return']:.2%}  "
                  f"夏普={r['sharpe'] or 0:.2f} 评分={score:.3f}")
        except Exception as e:
            print(f"  [{i}/{total}] {name:<14} 失败: {e}")

    results.sort(key=lambda x: x["_score"], reverse=True)
    print(f"\n" + "=" * 60)
    print(f"全量回测完成 — Top 10")
    print(f"{'排名':<4} {'策略':<14} {'年化':>7} {'回撤':>7} {'夏普':>7} {'评分':>7}")
    print(f"{'-'*50}")
    for i, r in enumerate(results[:10]):
        s = r.get("sharpe")
        print(f"{i+1:<4} {r['name']:<14} {r['annual_return']:>6.1%} "
              f"{r['drawdown']:>6.1%} {f'{s:.2f}' if s else 'N/A':>7} "
              f"{r['_score']:>7.3f}")

    # 查看演变
    bs = BacktestStore()
    print("\n" + bs.compare_strategies(365))


def menu_dashboard():
    """启动仪表盘"""
    import subprocess
    print("\n启动 Streamlit 仪表盘...")
    print("浏览器打开后可以看到市场数据、策略对比、持仓监控、盈亏分析。")
    print("按 Ctrl+C 可以退出。")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


# ═══════════════════════════════════════════
# 自动管线向导
# ═══════════════════════════════════════════

def _menu_pipeline_wizard():
    """CI/CD全流程向导：5步配置→自动执行→看报告"""
    print("\n" + "=" * 55)
    print("  自动管线向导")
    print("=" * 55)

    # Step 1: 数据
    print("\n── 1/5 数据 ──")
    mode = _pick(["单指数", "单指数+多股票面板（截面选股）", "全部标的（沪深300+中证500+上证50+创业板指）"], "数据模式")
    use_panel = "面板" in mode
    use_all = "全部" in mode
    if use_all:
        symbol = "沪深300"  # 主力标的
    else:
        symbol = _pick(["沪深300", "中证500", "上证50", "创业板指"], "标的")
    start, end = _pick_date_preset("20200101")
    print("  拉取数据...")
    import numpy as np, pandas as pd

    try:
        from data.fetchers.fallback import fetch_index_daily_safe as fd
        df = fd(symbol, start, end)
        print(f"  {symbol}: {len(df)}条")
    except Exception:
        print("  拉取失败，用合成数据")
        np.random.seed(42)
        n = 2000
        dates = pd.date_range(start, periods=n, freq='B')
        ret = np.random.randn(n) * 0.012 + 0.0003
        p = 3500 * np.cumprod(1 + ret)
        df = pd.DataFrame({'date': dates, 'open': p * 0.998, 'high': p * 1.012,
                           'low': p * 0.988, 'close': p,
                           'volume': np.random.randint(1, 100, n).astype(float) * 1e7})

    panel = None
    if use_panel:
        try:
            from data.fetchers.multi_stock import fetch_multi_stock_daily
            panel = fetch_multi_stock_daily(n_stocks=20, start=start)
            print(f"  面板: {len(panel)}只股票")
        except Exception:
            print("  面板拉取失败，跳过截面")
            use_panel = False

    # Step 2: 研究源头
    print("\n── 2/5 研究源头 ──")
    use_sources = "是" in _pick(["是（推荐）", "否"], "扫描arXiv+另类数据+日历?")
    # Step 3: 管线参数
    print("\n── 3/5 管线参数 ──")
    auto_rec = "是" in _pick(["是（推荐）", "否"], "淘汰时自动调Miner重挖?")
    until_opt = _pick(["VALIDATE（验证阶段）", "PAPER（纸上交易）", "BACKTEST（仅回测）"], "推进到")
    # Step 4: 执行
    print("\n── 4/5 确认执行 ──")
    print(f"  {symbol} | 面板={'是' if use_panel else '否'} | "
          f"源头={'是' if use_sources else '否'} | 恢复={'是' if auto_rec else '否'} | {until_opt}")
    if "返回" in _pick(["开始执行 [AI]", "返回主菜单"], "确认"):
        return

    import time as _t
    from core.pipeline import Pipeline, Stage
    t0 = _t.time()
    pl = Pipeline(df=df, panel=panel, cash=100000, max_workers=4, symbol=symbol)
    pl.add_track("Factor_Baseline", factors=["momentum_smooth", "volatility_1m",
                                               "vol_trend", "dollar_volume"])
    if use_sources:
        from core.research_source import ResearchSource
        print("  ResearchSource扫描...")
        rs = ResearchSource(df=df)
        signals = rs.scan_all(use_cache=False)
        rs.print_report(signals)
        n = pl.feed_signals(signals)
        if n > 0:
            print(f"  arXiv→LLM: {n}条因子Track")
        rs.save_signals(signals)

    until_map = {"VALIDATE": Stage.VALIDATE, "PAPER": Stage.PAPER, "BACKTEST": Stage.BACKTEST}
    until_stage = until_map.get(until_opt[:8], Stage.VALIDATE)

    print(f"\n  推进管线 ({len(pl.tracks)}条Track)...")
    tracks = pl.run(until_stage=until_stage, auto_recover=auto_rec)
    elapsed = _t.time() - t0

    # Step 5: 报告
    print("\n── 5/5 报告 ──")
    print(pl.report())
    ready = pl.ready_strategies()
    factor_t = sum(1 for t in tracks if t.name.startswith("factor_"))
    cs_t = sum(1 for t in tracks if "CrossSection" in t.name)
    recovered = sum(1 for t in tracks if "_v" in t.name and t.stage != Stage.ARCHIVED)
    archived = sum(1 for t in tracks if t.stage == Stage.ARCHIVED)
    print(f"\n  耗时: {elapsed:.1f}s | {len(tracks)}条Track")
    print(f"  READY: {len(ready)} | 淘汰: {archived} | 恢复: {recovered}")
    if factor_t: print(f"  因子驱动: {factor_t}条")
    if cs_t: print(f"  截面选股: {cs_t}条")
    if ready:
        print(f"\n  *** {len(ready)}条到达实盘候选! ***")
    else:
        print("\n  无策略到达READY——管线诚实淘汰。")
    _pick(["返回主菜单"], "Enter")


# ═══════════════════════════════════════════
# 二级子菜单
# ═══════════════════════════════════════════

def menu_factor():
    """因子分析 — IC/ICIR/衰减/相关性"""
    print("\n" + "=" * 50)
    print("  因子分析（IC/ICIR/衰减/相关性）")
    print("=" * 50)

    symbol = _pick(["沪深300", "中证500", "创业板指", "上证50"], "选择标的")
    start, end = _pick_date_preset("20200101")

    import numpy as np
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.factor_miner import FactorMiner, compute_ic, compute_rank_ic, ic_decay_analysis, factor_correlation_matrix

    print(f"\n拉取 {symbol} 数据...")
    df = fetch_index_daily_safe(symbol, start, end)
    print(f"共 {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")

    print("\n计算34个因子 + IC评估...")
    fm = FactorMiner(df)
    result = fm.mine()

    results = result["results"]
    results.sort(key=lambda x: x["abs_ic"], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  {symbol} 因子分析 — Top 15（按|IC|排名）")
    print(f"{'=' * 70}")
    print(f"  {'排名':<4} {'因子':<22} {'IC':>7} {'|IC|':>7} {'ICIR':>7} {'解读'}")
    print(f"  {'-' * 64}")
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:<4} {r['factor'][:21]:<22} {r['ic']:>+7.4f} {r['abs_ic']:>7.4f} "
              f"{r.get('ic_ir', 0):>7.3f}  {r['interpretation']}")

    # IC 分布统计
    ics = [r["ic"] for r in results]
    abs_ics = [r["abs_ic"] for r in results]
    print(f"\n  IC 分布: 均值={np.mean(ics):.4f}  中位数={np.median(ics):.4f}  "
          f"正值比例={sum(1 for x in ics if x > 0)/len(ics):.0%}")
    print(f"  |IC|>0.05（中等以上）: {sum(1 for x in abs_ics if x > 0.05)} 个")
    print(f"  |IC|>0.10（强预测力）: {sum(1 for x in abs_ics if x > 0.10)} 个")

    # IC 衰减分析（Top5 因子）
    print(f"\n{'=' * 70}")
    print(f"  IC 衰减分析（Top5 |IC| 因子 — 预测力持续多久）")
    print(f"{'=' * 70}")
    for r in results[:5]:
        fname = r["factor"]
        decay = ic_decay_analysis(fm.df, fname)
        periods_str = "  ".join(f"T+{p}:{decay['decay'][p]['ic']:.4f}" for p in [1, 5, 10, 21, 63])
        print(f"  {fname:<22} {periods_str}")

    # 可选：相关性矩阵
    if input("\n查看因子相关性矩阵（识别冗余因子）？[y/N]: ").strip().lower() == "y":
        corr = factor_correlation_matrix(fm.df)
        if not corr.empty:
            # 找高相关对
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            if high_corr:
                print(f"\n  高相关因子对（|corr|>0.7，建议只保留IC更高的）:")
                for a, b, v in sorted(high_corr, key=lambda x: -abs(x[2])):
                    print(f"    {a} <-> {b}: {v:+.3f}")
            else:
                print("  无高度相关因子对（|corr|均≤0.7），因子间冗余度低")
        else:
            print("  因子数不足，无法计算相关性矩阵")


def _menu_tools():
    """单步工具 — 所有功能按类别分块"""
    while True:
        print("\n" + "─" * 50)
        print("  单步工具")
        print("─" * 50)
        print("  ── 因子与策略生成 ──")
        print("  [1]  因子分析（IC/ICIR/衰减）")
        print("  [2]  策略挖掘（入场×出场组合）[AI]")
        print("  [3]  遗传算法(实验) [AI]")
        print("  [4]  贝叶斯优化(推荐) [AI]")
        print("  ── 回测验证 ──")
        print("  [5]  单策略回测")
        print("  [6]  策略大比武（全部排名）[AI]")
        print("  [7]  策略市场扫描（社区策略）[AI]")
        print("  [8]  全量回测 (耗时长，需二次确认) [AI]")
        print("  [9]  分钟级回测")
        print("  ── 实盘与风控 ──")
        print("  [10] 纸上交易（模拟逐日决策）[AI]")
        print("  [11] 风控参数（查看/修改）")
        print("  [12] 实盘就绪检查（6项自评）")
        print("  [13] vnpy A股/期货（需SimNow）")
        print("  [14] 启动守护进程")
        print("  ── 纸上增强（虚拟盘）──")
        print("  [19] 纸上联赛（多策略并行竞赛）")
        print("  [20] 纸上对比（回测vs纸上diff）")
        print("  [21] 纸上守护（长期运行+自动恢复）")
        print("  ── 监控与分析 ──")
        print("  [15] 启动监控仪表盘")
        print("  [16] 市场区制检测（Bull/Bear）[AI]")
        print("  [17] 另类数据扫描（情绪+地缘）[AI]")
        print("  [18] 通知渠道测试")
        print("  [22] Agent环境研判（多角色辩论）[AI]")
        print("  [0]  返回")

        try: n = int(input("> "))
        except ValueError: continue
        if n == 0: break
        try: _TOOLS[n]()
        except KeyError: pass
        except Exception as e: print(f"[{type(e).__name__}] {e}")

def _menu_data():
    """数据管理"""
    while True:
        print("\n" + "─" * 50)
        print("  数据管理")
        print("─" * 50)
        print("  [1] 拉取行情数据 → 自动入仓")
        print("  [2] 策略仓库（导入/搜索/对比）")
        print("  [3] 研究源头扫描 [AI]")
        print("  [4] 数据源健康检查")
        print("  [5] 本地数据仓库总览")
        print("  [0] 返回")

        try: n = int(input("> "))
        except ValueError: continue
        if n == 0: break
        try: _DATA[n]()
        except KeyError: pass
        except Exception as e: print(f"[{type(e).__name__}] {e}")


def main():
    print("=" * 60)
    print("  量化鲲鹏 (QuantP) — 个人量化交易系统")
    print("=" * 60)
    print("  记不住命令没关系，选数字就行。")


def _checklist_item(label, passed, detail="", fail_hint=""):
    """就绪检查单项"""
    if passed:
        print(f"  [PASS] {label}")
        if detail:
            print(f"         {detail}")
    else:
        print(f"  [FAIL] {label}")
        if detail:
            print(f"         原因: {detail}")
        if fail_hint:
            print(f"         -> {fail_hint}")


def menu_pipeline():
    """敏捷管线 — 因子驱动基线 + 研究源头信号 → 自动推进"""
    print("\n" + "=" * 60)
    print("  敏捷量化管线 (Pipeline) — 因子驱动自动推进")
    print("=" * 60)
    print("  Stage1: 34因子IC分析 + LLM生成新因子 → 强因子自动建Track")
    print("  Stage2-4: 回测→验证→纸上，淘汰时自动重挖")
    print("  ResearchSource: arXiv论文→LLM提取因子→IC验证→自动Track\n")

    try:
        from data.fetchers.fallback import fetch_index_daily_safe as fd
        from core.pipeline import Pipeline, Stage
        from core.research_source import ResearchSource

        symbol = _pick(["沪深300", "中证500"], "选择标的")
        start, end = _pick_date_preset("20200101")

        print(f"\n拉取 {symbol} 数据...")
        df = fd(symbol, start, end)
        print(f"  数据: {len(df)} 行")

        pl = Pipeline(df, cash=100000, max_workers=2)

        # ── 基线: 因子驱动（1条FactorAuto + Stage1自动生成的强因子Track）──
        pl.add_track("Factor_Baseline", factors=["momentum_smooth", "volatility_1m",
                                                   "dollar_volume", "ret_3m"])

        # ── ResearchSource → arxiv因子信号 → LLM提取 → 自动Track ──
        print("  扫描研究源头...")
        rs = ResearchSource(df=df)
        signals = rs.scan_all(use_cache=False)
        n = pl.feed_signals(signals)
        if n > 0:
            print(f"  arXiv→LLM: {n} 条新因子Track")
        rs.save_signals(signals)

        print(f"\n推进管线 ({len(pl.tracks)} tracks, auto_recover=on)...")
        tracks = pl.run(until_stage=Stage.VALIDATE, auto_recover=True)

        print(pl.report())

        ready = pl.ready_strategies()
        factor_tracks = [t for t in tracks if t.name.startswith("factor_")]
        recovered = [t for t in tracks if "_v" in t.name and t.stage != Stage.ARCHIVED]
        if ready:
            print(f"\n  {len(ready)} 条 Track 到达实盘候选！")
        if factor_tracks:
            print(f"  因子驱动基线: {len(factor_tracks)} 条强因子Track自动生成")
        if recovered:
            print(f"  自动恢复: {len(recovered)} 条重生并通过验证")
        if not ready and not recovered:
            print("\n  所有 Track 未通过验证——管线正确淘汰了不合格策略。")

        _pick(["返回主菜单"], "按 Enter 返回")

    except Exception as e:
        print(f"\n管线运行异常: {e}")
        import traceback
        traceback.print_exc()
        _pick(["返回主菜单"], "按 Enter 返回")


def menu_research_source():
    """研究源头扫描——检测外部变化，生成新的研究信号"""
    print("\n" + "=" * 55)
    print("  研究源头扫描（外部新鲜输入→研究信号→管线Track）")
    print("=" * 55)

    from core.research_source import ResearchSource

    use_data = input("是否加载行情数据用于结构检测? (y/n, 默认n): ").strip().lower()

    rs = ResearchSource()
    if use_data == "y":
        try:
            from data.fetchers.fallback import fetch_index_daily_safe
            symbol = input("标的 (默认沪深300): ").strip() or "沪深300"
            start = input("起始日期 (默认20230101): ").strip() or "20230101"
            print(f"  拉取{symbol} {start}~今日...")
            df = fetch_index_daily_safe(symbol, start)
            rs = ResearchSource(df=df)
            print(f"  {len(df)}条数据")
        except Exception as e:
            print(f"  数据拉取失败: {e}，使用无数据模式")
            rs = ResearchSource()

    print("\n  扫描中...")
    signals = rs.scan_all()
    report = rs.print_report(signals)
    print(report)

    if signals:
        saved = rs.save_signals(signals)
        print(f"\n  信号已存储: {saved}条 (ResearchStore)")

        high_priority = [s for s in signals if s.priority >= 4]
        if high_priority:
            print(f"\n  ⚠ 高优先级信号 {len(high_priority)} 条需要关注:")
            for s in high_priority:
                review = " [需人工确认]" if s.needs_review else ""
                print(f"    P{s.priority} [{s.source}] {s.title}{review}")

        print("\n  提示: 在[管线]菜单中可自动消费这些信号生成Track")

def _show_kronos_banner():
    """主菜单 Kronos 状态横幅（仅异常/未完成状态时显示）"""
    try:
        from core.kronos.engine import KronosEngine, _KRONOS_AVAILABLE
        from config.loader import get_kronos_config
        cfg = get_kronos_config()

        deps_ok = _KRONOS_AVAILABLE
        enabled = cfg.get("source_enabled", False)
        dl = KronosEngine.check_models_downloaded(cfg.get("model", "small")) if deps_ok else None

        if not deps_ok:
            print("  [Kronos] 依赖未安装 (pip install torch huggingface_hub einops)")
        elif not enabled:
            print("  [Kronos] 未启用 — [5] 查看配置指引")
        elif dl and not dl["all_downloaded"]:
            print(f"  [Kronos] 模型未下载(~{dl['download_size_mb']}MB) — [5] 查看详情")
        # 全部正常则不显示（静默）
    except Exception:
        pass  # Kronos 模块不可用时静默


def menu_kronos_status():
    """Kronos 状态查看 — 模型下载/设备/开关/配置指引"""
    print("\n" + "=" * 55)
    print("  Kronos 量化K线预测模型 — 状态与配置")
    print("=" * 55)

    # 0. 加载配置
    from config.loader import get_kronos_config
    cfg = get_kronos_config()

    # 1. 依赖检查
    from core.kronos.engine import _KRONOS_AVAILABLE, _import_error
    if not _KRONOS_AVAILABLE:
        print(f"\n  [依赖] 未安装")
        print(f"    缺失: {_import_error}")
        print(f"    安装: pip install torch huggingface_hub einops")
        print(f"\n  Kronos 是什么？")
        print(f"    基于 AAAI 2026 BSQ 离散 token 的自回归 Transformer")
        print(f"    输入历史OHLCV K线 → 预测未来价格序列")
        print(f"    纯本地运行，不需要 API Key，不需要网络（仅首次下载模型）")
        print(f"    项目: https://github.com/NeoQuasar/Kronos (MIT 协议)")
        _pick(["返回主菜单"], "按 Enter 返回")
        return

    # 2. 模型下载状态
    from core.kronos.engine import KronosEngine
    model_size = cfg.get("model", "small")
    dl = KronosEngine.check_models_downloaded(model_size)

    print(f"\n  [依赖] 已安装 ✓")
    print(f"  [模型规格] Kronos-{model_size}")
    print(f"  [推理设备] {cfg.get('device', 'auto')}")
    print(f"  [模型缓存] {dl['cache_dir']}")

    print(f"\n  ── 模型下载状态 ──")
    print(f"  Tokenizer ({dl['tokenizer_repo']}): {'已下载 ✓' if dl['tokenizer'] else '未下载'}")
    print(f"  Predictor ({dl['model_repo']}): {'已下载 ✓' if dl['model'] else '未下载'}")

    if not dl["all_downloaded"]:
        print(f"\n  需下载约 {dl['download_size_mb']}MB 模型文件（仅首次一次）")
        print(f"  方式1: 首次调用 predict() 时自动下载")
        print(f"  方式2: 手动预下载（推荐）:")
        print(f"    from huggingface_hub import snapshot_download")
        print(f"    snapshot_download('{dl['tokenizer_repo']}')")
        print(f"    snapshot_download('{dl['model_repo']}')")
    else:
        print(f"\n  模型已就绪 ✓ — 可以开始预测")

    # 3. 功能开关
    print(f"\n  ── 功能开关 (在 settings.local.yaml 中修改) ──")
    switches = [
        ("source_enabled", "信号源节点（预测→因子→Track）"),
        ("factor_registration", "因子自动入库+独特性检验"),
        ("l4_agent", "L4 Agent 决策层接入"),
        ("risk_guard", "guardian 风险预警接入"),
        ("webui_enabled", "WebUI 预测可视化"),
        ("token_analysis", "Token 熵分析"),
    ]
    for key, desc in switches:
        val = cfg.get(key, False)
        mark = "开" if val else "关"
        print(f"  [{mark}] {key}: {desc}")

    # 4. 预测参数
    print(f"\n  ── 默认预测参数 ──")
    print(f"  pred_len={cfg.get('pred_len', 20)}  T={cfg.get('T', 1.0)}  "
          f"top_p={cfg.get('top_p', 0.9)}  sample_count={cfg.get('sample_count', 5)}")
    print(f"  lookback={cfg.get('lookback', 60)}  ic_threshold={cfg.get('ic_threshold', 0.03)}")

    # 5. 快速操作
    print(f"\n  ── 操作 ──")
    if cfg.get("source_enabled", False) and dl["all_downloaded"]:
        print(f"  输入 'test' 运行 Kronos 烟雾测试")
    if not cfg.get("source_enabled", False):
        print(f"  输入 'on' 启用 Kronos 信号源")
    print(f"  输入 'guide' 打印完整配置指引")
    print(f"  按 Enter 返回主菜单")

    choice = input("\n> ").strip().lower()
    if choice == "test":
        print("\n  创建 KronosEngine...")
        try:
            engine = KronosEngine(
                model_size=cfg.get("model", "small"),
                device=cfg.get("device", "auto"),
                project_path=cfg.get("project_path", ""),
            )
            engine.print_setup_guide()
            if engine.is_downloaded:
                print("  模型已就绪。如需实际预测，请在管线中启用 Kronos 信号源后运行。")
        except Exception as e:
            print(f"  引擎创建失败: {e}")
        _pick(["返回主菜单"], "按 Enter 返回")
    elif choice == "on":
        print("\n  要启用 Kronos 信号源，请在以下文件中修改:")
        print("    config/settings.local.yaml")
        print("    kronos:")
        print("      source_enabled: true")
        print("\n  修改后重新启动 interactive.py 即可生效。")
        _pick(["返回主菜单"], "按 Enter 返回")
    elif choice == "guide":
        try:
            engine = KronosEngine(
                model_size=cfg.get("model", "small"),
                device=cfg.get("device", "auto"),
                project_path=cfg.get("project_path", ""),
            )
            engine.print_setup_guide()
        except Exception as e:
            print(f"  引擎创建失败: {e}")
        _pick(["返回主菜单"], "按 Enter 返回")


def menu_kronos_predict():
    """Kronos 预测 + 可视化报告"""
    print("\n" + "=" * 55)
    print("  Kronos 预测 + 可视化")
    print("=" * 55)

    try:
        from core.kronos.engine import KronosEngine
        from core.kronos.viz import prediction_report
        from config.loader import get_kronos_config
    except ImportError as e:
        print(f"  导入失败: {e}")
        _pick(["返回"], "按 Enter 返回")
        return

    cfg = get_kronos_config()
    if not cfg.get("source_enabled", False):
        print("  Kronos source 未启用 (source_enabled=false)")
        print("  在 settings.local.yaml 中设置 kronos.source_enabled=true")
        _pick(["返回"], "按 Enter 返回")
        return

    print("  正在加载 Kronos 引擎...")
    try:
        engine = KronosEngine(
            model_size=cfg.get("model", "small"),
            device=cfg.get("device", "auto"),
            tokenizer_path=cfg.get("tokenizer_path", ""),
            model_path=cfg.get("model_path", ""),
            project_path=cfg.get("project_path", ""),
        )
    except Exception as e:
        print(f"  引擎创建失败: {e}")
        _pick(["返回"], "按 Enter 返回")
        return

    print("  正在拉取数据...")
    try:
        from data.fetchers.fallback import fetch_index_daily_safe
        df = fetch_index_daily_safe("沪深300", "20240101", "")
        if df is None or len(df) < 60:
            print("  数据不足")
            _pick(["返回"], "按 Enter 返回")
            return
    except Exception as e:
        print(f"  数据获取失败: {e}")
        _pick(["返回"], "按 Enter 返回")
        return

    lookback = int(cfg.get("lookback", 60))
    pred_len = int(cfg.get("pred_len", 20))

    print(f"  数据: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")
    print(f"  回看: {lookback} | 预测: {pred_len} 步")
    print()

    report = prediction_report(engine, df, lookback=lookback, pred_len=pred_len)
    print(report)

    engine.unload()
    _pick(["返回主菜单"], "按 Enter 返回")


def menu_go_live():
    """实盘就绪检查——逐项确认后才能解锁实盘建议"""
    print("\n" + "=" * 55)
    print("  实盘就绪检查（Phase 4 最终确认清单）")
    print("=" * 55)
    print("  以下 6 项全部 PASS 后，方可开始最小实盘。")
    print()
    print("  [!] 免责声明：")
    print("  量化交易存在重大亏损风险。回测结果不代表未来表现。")
    print("  本检查清单仅为自我评估工具，不构成投资建议。")
    print("  任何实盘交易决策的风险均由使用者自行承担。")
    print("  建议仅使用可承受全额亏损的资金进行量化交易。\n")

    from config.loader import get_config

    passed = 0
    failed = 0
    total = 6

    # 1. 回测周期
    print("-- 1/6 回测周期覆盖完整牛熊 --")
    ans = _pick(["是(>=3年，含2015股灾+熊市+修复)",
                 "否(<3年或只在牛市区间)",
                 "不确定"],
                "回测数据是否覆盖>=3年完整牛熊周期？")
    if "是" in ans:
        _checklist_item("回测周期覆盖", True, "已覆盖完整牛熊周期")
        passed += 1
    else:
        _checklist_item("回测周期覆盖", False,
                        "回测至少需从2015年至今",
                        "菜单[1]拉数据时起始日期填20150101")
        failed += 1

    # 2. 样本外验证
    print("\n-- 2/6 样本外验证 --")
    ans = _pick(["是(测试集收益衰减<=30%%)",
                 "否(未做或衰减>30%%)",
                 "不确定"],
                "策略是否通过了样本外验证？")
    if "是" in ans:
        _checklist_item("样本外验证", True, "测试集收益衰减可接受")
        passed += 1
    else:
        _checklist_item("样本外验证", False,
                        "样本外未通过=策略很可能过拟合",
                        "回测时用auto_validate=True")
        failed += 1

    # 3. 跨品种
    print("\n-- 3/6 跨品种验证 --")
    ans = _pick(["是(至少2个品种)",
                 "否(只在单一品种上测试)",
                 "仅ETF/指数(合规安全)"],
                "策略是否在多个品种上验证过？")
    if "是" in ans or "ETF" in ans:
        _checklist_item("跨品种验证", True, "多品种或ETF策略")
        passed += 1
    else:
        _checklist_item("跨品种验证", False,
                        "单一品种有效可能是巧合",
                        "换中证500或创业板指重跑")
        failed += 1

    # 4. 风控配置(自动)
    print("\n-- 4/6 风控参数 --")
    cfg = get_config()
    risk = cfg.get("risk", {})
    if risk.get("max_position_pct") and risk.get("max_drawdown_pct"):
        _checklist_item("风控参数", True,
                        f"仓位{risk.get('max_position_pct',0)*100:.0f}%% "
                        f"止损{risk.get('max_single_loss_pct',0)*100:.0f}%% "
                        f"回撤{risk.get('max_drawdown_pct',0)*100:.0f}%%")
        passed += 1
    else:
        _checklist_item("风控参数", False,
                        "风控参数缺失",
                        "在settings.yaml的risk段填写")
        failed += 1

    # 5. 通知配置(自动)
    print("\n-- 5/6 通知渠道 --")
    notif = cfg.get("notification", {})
    has_notif = notif.get("telegram_bot_token") or notif.get("wecom_webhook")
    if has_notif:
        _checklist_item("通知渠道", True, "已配置通知")
        passed += 1
    else:
        _checklist_item("通知渠道", False,
                        "未配置通知——异常时无法告警",
                        "在settings.local.yaml配置telegram或wecom")
        failed += 1

    # 6. 纸上交易
    print("\n-- 6/6 纸上交易 --")
    ans = _pick(["是(已完成>=4周)",
                 "否(未做或不足4周)",
                 "正在做"],
                "是否已完成连续4周纸上交易？")
    if "是" in ans:
        _checklist_item("纸上交易", True, "已完成4周模拟")
        passed += 1
    elif "正在做" in ans:
        _checklist_item("纸上交易", False, "进行中", "完成4周后再检查")
        failed += 1
    else:
        _checklist_item("纸上交易", False,
                        "实盘前最重要一步",
                        "菜单[10]纸上交易，至少跑4周")
        failed += 1

    # 汇总
    print("\n" + "=" * 55)
    print(f"  就绪检查: {passed}/{total} 项通过")
    print("=" * 55)

    if passed == total:
        print("\n[OK] 全部通过！可以开始最小实盘。")
        print("\n  资金公式: MAX(能笑着烧掉的金额, 月收入*50%%, 流动资产*2%%)")
        print("  建议: 3,000 - 10,000 元")
        print("\n  实盘入口: 菜单[实盘] vnpy A股/期货（需SimNow账号）")
        print("  当前状态: vnpy_ctp 编译阻塞，待解决后可下单")
        print("\n  免责: 量化交易存在重大亏损风险。回测不代表未来。风险自担。")
    else:
        print(f"\n还有 {failed} 项未通过。请先完成再考虑实盘。")
        print("每一项 FAIL 都对应已知失败案例——不要跳。")


    while True:
        print("\n" + "=" * 55)
        print("  量化鲲鹏 (QuantP)")
        print("=" * 55)
        # Kronos 状态提醒（模型未下载或未启用时显示）
        _show_kronos_banner()
        print("  [1] 5分钟体验   — 零配置快速验证系统是否正常")
        print("  [2] 自动管线    — CI/CD全流程：源头→因子(含LLM)→策略→回测→验证→截面→重挖")
        print("  [3] 单步工具    — 因子/挖掘/优化/回测/纸上/风控/实盘/区制/另类/仪表盘")
        print("  [4] 数据管理    — 拉取行情/策略仓库/源头扫描/健康检查/仓库总览")
        print("  [5] Kronos状态  — 模型下载/设备/开关状态")
        print("  [6] Kronos预测  — 运行预测+可视化报告")
        print("  [0] 退出")

        try:
            n = int(input("\n> "))
        except ValueError:
            print("输入数字")
            continue

        if n == 0:
            print("再见！")
            break
        elif n == 1:
            menu_quick()
        elif n == 2:
            _menu_pipeline_wizard()
        elif n == 3:
            _menu_tools()
        elif n == 4:
            _menu_data()
        elif n == 5:
            menu_kronos_status()
        elif n == 6:
            menu_kronos_predict()
        else:
            print("输入 0-6 之间的数字")


_TOOLS = {
    1:  menu_factor,       2:  menu_mine,       3:  menu_ga,
    4:  menu_bayes,        5:  menu_backtest,    6:  menu_shootout,
    7:  menu_market,       8:  menu_full_backtest, 9: menu_minute_backtest,
    10: menu_paper,        11: menu_risk,        12: menu_go_live,
    13: menu_live_vnpy,    14: menu_guardian,
    15: menu_dashboard,    16: menu_regime,      17: menu_alternative,
    18: menu_notify_test,
    19: menu_paper_league, 20: menu_paper_diff,  21: menu_paper_daemon,
    22: menu_agent_decision,
}
_DATA = {
    1: menu_data, 2: menu_repo_mgmt, 3: menu_research_source,
    4: menu_health, 5: menu_vault,
}

if __name__ == "__main__":
    main()
