#!/usr/bin/env python
"""
统一策略运行入口 — 参考 BsStrategy 统一量化工作台

一键运行策略，选择模式：
  --mode backtest  回测（快速验证策略逻辑）
  --mode paper     纸上交易（模拟真实逐日决策）
  --mode live      实盘交易（Binance Testnet 免费假钱）
  --mode shootout  策略大比武（全部策略对比排名）

用法
--------
  # 回测双均线
  python run_strategy.py --mode backtest --strategy 双均线交叉 --symbol 沪深300

  # 纸上交易
  python run_strategy.py --mode paper --strategy 双均线交叉 --symbol 沪深300

  # 策略大比武
  python run_strategy.py --mode shootout --symbol 沪深300

  # 实盘（需要先配 API key）
  python run_strategy.py --mode live --symbol BTC/USDT
"""

import argparse
import sys
from pathlib import Path

# 确保项目根在 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_backtest(args):
    """回测模式"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.engine.bt_runner import run_backtest
    from backtest.analysis.report import generate_report, plot_equity_curve
    from backtest.strategy_market import ALL_STRATEGIES

    info = ALL_STRATEGIES[args.strategy]
    strategy_class = info["class"]

    print(f"回测: {args.strategy} on {args.symbol}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)
    result = run_backtest(
        strategy_class, df,
        initial_cash=args.cash,
        **info.get("params", {}),
    )

    print(generate_report(result))

    if args.plot:
        plot_equity_curve(result, save_path=f"notebooks/{args.strategy}_equity.png")


def cmd_paper(args):
    """纸上交易模式"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_market import ALL_STRATEGIES
    from live.paper_trader import run_paper_trading

    info = ALL_STRATEGIES[args.strategy]
    strategy_class = info["class"]

    print(f"纸上交易: {args.strategy} on {args.symbol}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)
    result = run_paper_trading(
        strategy_class, df,
        initial_cash=args.cash,
        **info["params"],
    )

    print(result["summary"])

    if result["warnings"]:
        print(f"\n风控拦截 {len(result['warnings'])} 次")


def cmd_market(args):
    """市场扫描：从社区策略库导入+内置策略，逐个回测排名"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_market import scan_market

    print(f"策略市场扫描: {args.symbol}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)
    scan_market(df, cash=args.cash)


def cmd_mine(args):
    """策略挖掘：元素组合 → 自动回测 → 优选输出"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_miner import StrategyMiner

    print(f"策略挖掘: {args.symbol}, 最多 {args.max_combos} 组合")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)
    miner = StrategyMiner(df, cash=args.cash)
    results = miner.mine(max_combinations=args.max_combos, min_trades=args.min_trades)
    print(results["summary"])


def cmd_live(args):
    """实盘模式（Binance Testnet）"""
    from live.gateway.ccxt_gateway import CCXTGateway

    print(f"实盘模式: {args.symbol}")
    print("（当前为测试网模式，使用 Binance Testnet 免费假钱）")

    gw = CCXTGateway(exchange="binance", testnet=True)
    if gw.connect():
        ticker = gw.get_ticker(args.symbol)
        if ticker:
            print(f"{args.symbol}: ${ticker['last']:,.2f}")

        balance = gw.get_balance("USDT")
        print(f"USDT 余额: {balance}")

        if args.dry_run:
            print("模拟下单...")
            order = gw.market_buy(args.symbol, 0.001)
            if order:
                print(f"模拟成交: {order['id']}")

        gw.disconnect()


def cmd_shootout(args):
    """策略大比武"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.analysis.shootout import run_shootout

    print(f"策略大比武: {args.symbol}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)
    run_shootout(df, cash=args.cash)


def _build_param_space(strategy_name: str, scale: float = 0.5) -> dict:
    """根据策略的默认参数构建搜索空间，上下浮动 scale 倍"""
    from backtest.strategy_market import ALL_STRATEGIES
    info = ALL_STRATEGIES[strategy_name]
    params = info.get("params", {})
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


def cmd_ga(args):
    """遗传算法优化模式"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_market import ALL_STRATEGIES
    from backtest.analysis.genetic_miner import GeneticOptimizer

    info = ALL_STRATEGIES[args.strategy]
    param_space = _build_param_space(args.strategy)
    print(f"遗传算法优化: {args.strategy} on {args.symbol}, 参数空间={param_space}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)

    go = GeneticOptimizer(df, cash=args.cash)
    best = go.evolve(
        info["class"],
        param_space=param_space,
        pop_size=args.pop_size,
        generations=args.generations,
    )
    print(f"\n最优参数: {best['params']}")
    print(f"夏普={best['sharpe']:.2f}  年化={best['annual_return']:.2%}  回撤={best['drawdown']:.2%}")


def cmd_bayes(args):
    """贝叶斯优化模式"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from backtest.strategy_market import ALL_STRATEGIES
    from backtest.analysis.bayesian_opt import BayesianOptimizer

    info = ALL_STRATEGIES[args.strategy]
    param_space = _build_param_space(args.strategy)
    print(f"贝叶斯优化: {args.strategy} on {args.symbol}, 参数空间={param_space}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)

    bo = BayesianOptimizer(df, cash=args.cash)
    best = bo.optimize(
        info["class"],
        param_space=param_space,
        n_trials=args.trials,
    )
    print(f"\n最优参数: {best['params']}")
    print(f"夏普={best['sharpe']:.2f}  年化={best['annual_return']:.2%}  回撤={best['drawdown']:.2%}")

    importance = bo.get_importance()
    if importance:
        print(f"参数重要性: {importance}")


def cmd_rl(args):
    """强化学习模式"""
    from data.fetchers.fallback import fetch_index_daily_safe
    from experimental.rl_trader import RLTrainer

    print(f"RL训练: {args.symbol}")
    df = fetch_index_daily_safe(args.symbol, args.start, args.end)
    trainer = RLTrainer(df, cash=args.cash)
    result = trainer.train(timesteps=args.timesteps)
    print(f"\n测试收益: {result['test_return']:.2%} (买入持有: {result['buy_hold_return']:.2%})")
    print(f"夏普: {result['test_sharpe']:.2f}  交易: {result['n_trades']}次")


def main():
    parser = argparse.ArgumentParser(description="量化鲲鹏 (QuantP) 统一运行入口")
    parser.add_argument(
        "--mode", "-m",
        choices=["backtest", "paper", "live", "shootout", "market", "mine", "ga", "bayes", "rl"],
        default="backtest",
        help="运行模式: backtest=回测, paper=纸上交易, live=实盘, shootout=策略对比, market=市场扫描, mine=策略挖掘, ga=遗传算法优化, bayes=贝叶斯优化, rl=强化学习",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="双均线交叉",
        help="策略名称（如: 双均线交叉、布林带回归、动量策略、均值回归、网格交易）",
    )
    parser.add_argument(
        "--symbol",
        default="沪深300",
        help="交易标的（如: 沪深300、中证500、BTC/USDT）",
    )
    parser.add_argument("--start", default="20230101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="", help="结束日期 YYYYMMDD")
    parser.add_argument("--cash", type=float, default=100000, help="初始资金（元）")
    parser.add_argument("--plot", action="store_true", help="画权益曲线图")
    parser.add_argument("--dry-run", action="store_true", help="模拟下单不真买")
    parser.add_argument("--max-combos", type=int, default=100, help="策略挖掘最大组合数")
    parser.add_argument("--min-trades", type=int, default=5, help="最少交易次数过滤")
    parser.add_argument("--pop-size", type=int, default=30, help="遗传算法种群大小")
    parser.add_argument("--generations", type=int, default=20, help="遗传算法代数")
    parser.add_argument("--trials", type=int, default=50, help="贝叶斯优化采样次数")
    parser.add_argument("--timesteps", type=int, default=50000, help="RL训练步数")

    args = parser.parse_args()

    if args.mode == "backtest":
        cmd_backtest(args)
    elif args.mode == "paper":
        cmd_paper(args)
    elif args.mode == "live":
        cmd_live(args)
    elif args.mode == "shootout":
        cmd_shootout(args)
    elif args.mode == "market":
        cmd_market(args)
    elif args.mode == "mine":
        cmd_mine(args)
    elif args.mode == "ga":
        cmd_ga(args)
    elif args.mode == "bayes":
        cmd_bayes(args)
    elif args.mode == "rl":
        cmd_rl(args)


if __name__ == "__main__":
    main()
