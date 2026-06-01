# QuantP — Full Documentation

[中文](README_zh.md) | [Short Version](README.md)

## What Is This

QuantP is a quantitative trading research toolkit covering the full pipeline from data ingestion to paper trading. It is built with a focus on learning quantitative methodology — all features are functional at the code level, but depth and robustness are at learning-project level, not production-grade.

**Who this is for**: Developers who want to understand the full quant workflow, strategy researchers, and quant beginners.

**Current status**: Backtesting / analysis / validation pipeline is operational (WFA + DSR + CSCV multi-validation + FWER batch correction). Live trading gateway is skeleton-only — no real-money trades have been executed.

## Architecture

```
                        ┌───────────────────────────┐
                        │   Streamlit Dashboard (5p)  │
                        └─────────────┬─────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────┐
│                    Research Sources → Pipeline                           │
│                                                                          │
│  6 external signal sources → Parallel Tracks → Stage Gates → Adv/Prune   │
│  Pruned tracks auto-trigger strategy miner → new Track → immediate run  │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────┐
│               Backtest Engine + Strategies + Analysis                    │
│                                                                          │
│  Backtrader event-driven · 20+ strategies (4 groups) · Strategy Miner   │
│  WFA · DSR · CSCV PBO · FWER/FDR batch correction                       │
│  Factor research: ~34 factors · LLM factor mining (DeepSeek)            │
│  Optimizers: Bayesian (Optuna) · Genetic (experimental)                  │
│  Performance: Attribution · Statistical tests · Regime detection        │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────┐
│                            Data Layer                                     │
│                                                                          │
│  AkShare / Tushare / Baostock → ETL → MarketVault / FactorStore          │
│  Daily · Minute-level · Adjustment · Trading Calendar                    │
│  Storage: Parquet + SQLite + JSON + JSONL                                │
└─────────────────────────────────────────────────────────────────────────┘

Cross-cutting modules:
  ▪ Paper Trading — Daily simulation + Daemon + Multi-strategy League
  ▪ Live Gateway  — vnpy (A-share skeleton) + CCXT
  ▪ Risk Engine   — 8 hard-coded checks + StrategyGuard 4-dim circuit breaker
```

> Data flows bottom-up: Data Layer → Backtest/Analysis → Pipeline orchestration → Dashboard on top.

---

## Key Features

### Factor & Strategy Research

- **Factor-driven baseline**: Pipeline Stage 1 creates a strategy Track for every strong factor (|IC| > 0.05)
- **LLM factor mining**: DeepSeek API generates factor formulas → IC validation → FactorStore persistence
- **Research source auto-scanning**: 6 external signal sources (arXiv / data freshness / market structure / alternative data / strategy health / calendar events)
- **Cross-sectional multi-factor**: Multi-stock panel → daily ranking → buy Top N → periodic rebalance
- **Strategy miner**: Entry rules × Exit rules → overfitting filter → ranked output

### Overfitting Control (Six-Layer Defense)

| Layer | Method | Description |
|-------|--------|-------------|
| 1. Logic pre-filter | min_trades ≥ 30, Sharpe ≥ 0 | Eliminate meaningless results |
| 2. Three-way split | Train / Validation / Test independent | Prevent data leakage |
| 3. Walk-forward | WFA with rolling windows | Detect parameter instability |
| 4. Decay rate filter | OOS/IS return ratio ≥ 0.5 | Exclude severe overfitting |
| 5. Batch correction | FWER (Bonferroni/Holm) + FDR | Multiple testing correction |
| 6. PBO quantification | CSCV | Numerical overfitting probability |

### Trading & Risk Control

- **Multi-asset cost models**: Stocks/ETFs/Convertible bonds — config-driven
- **8 hard-coded risk checks**: Each from a real failure case
- **Volatility-targeted position sizing**: Auto-reduce on high vol, expand on low vol

---

## Quick Start

```bash
git clone https://github.com/quantp/quantp.git
cd quantp
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
# source venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
python interactive.py
```

Requires Python 3.12+. Optional dependencies gracefully degrade if not installed.

`python interactive.py` provides 4 entry points: 5-Minute Tour / Auto Pipeline / Single Tools (22 items) / Data Management.

### Auto Pipeline Wizard

5 steps: Select data → Configure research sources → Set pipeline parameters → Confirm & execute → View reports.

### Single Tools (22 tools)

| Group | Tools |
|-------|-------|
| Factor & Strategy | Factor Analysis / Strategy Mining / Genetic (experimental) / Bayesian (recommended) |
| Backtest | Single Backtest / Shootout / Market Scan / Full Backtest / Minute-level |
| Live & Risk | Paper Trading / Risk Parameters / Readiness Check / vnpy (skeleton) / Daemon |
| Paper Enhancements | Paper League / Paper-vs-Backtest Diff / Paper Daemon |
| Monitor & Analysis | Dashboard / Regime Detection / Alternative Data / Agent Assessment |

---

## Project Structure

```
quantp/
├── core/                  # Core engine (Pipeline / Research sources / Task scheduler)
├── backtest/
│   ├── engine/            #   Backtest engine + Strategy registry
│   ├── strategies/        #   Strategy layer (builtin / community / experimental / imported)
│   ├── analysis/          #   Analysis toolkit
│   ├── strategy_miner.py  #   Strategy miner
│   └── strategy_market.py #   Strategy market scanner
├── live/                  # Live/Paper trading (Gateway / Execution / Risk / Monitor)
├── data/                  # Data layer (Fetchers / Vault / Alternative / Cleaner)
├── config/                # Configuration management
├── dashboard/             # Streamlit dashboard
├── strategies_repo/       # Strategy repository tools
├── experimental/          # Experimental modules
├── notebooks/             # Jupyter exploration
├── tests/                 # Tests
├── interactive.py         # Interactive menu
└── run_strategy.py        # CLI mode
```

---

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.12 | Full-stack |
| Data | AkShare + Tushare Pro + Baostock | Three-tier fallback |
| Backtest | Backtrader (event-driven) | AShareCommission multi-asset cost models |
| Factors | FactorMiner (~34 factors) + LLMFactorMiner (DeepSeek) | IC/RankIC/decay |
| Optimizer | Bayesian (Optuna) + Genetic (experimental) | |
| Storage | Parquet + SQLite + JSON + JSONL | Files as database |
| Visualization | Streamlit + Matplotlib | Python-native |
| AI | DeepSeek API (factor mining / paper extraction) | Optional |

### Supported Asset Types

| Asset | Commission | Stamp Duty | T+N | Price Limit |
|-------|-----------|------------|-----|:-----------:|
| Stock | 0.025% | 0.05% (sell) | T+1 | ±10% |
| ETF | 0.025% | None | T+1 | ±10%/20% |
| Convertible Bond | 0.005% | None | T+0 | None |

---

## Comparison with Similar Tools

| Dimension | QuantP | vnpy | Freqtrade | Backtrader | JoinQuant |
|-----------|--------|------|-----------|------------|-----------|
| **Positioning** | Learning & research | A-share live trading | Crypto live trading | General backtest | Cloud platform |
| **Live Trading** | Skeleton | 40+ broker APIs | Multi-exchange | None | Broker direct |
| **Backtest** | Backtrader wrapper | Built-in event-driven | Vectorized/event | Native event-driven | Cloud engine |
| **Factor Research** | ~34 factors + LLM | Limited | Limited | None | Factor lib + custom |
| **Strategies** | 20+ (4 groups) | 100+ community | Hundreds | Write your own | Community shared |
| **Overfitting Control** | 6-layer + CSCV PBO | None | None | None | Limited |
| **AI Integration** | LLM mining + Agent | None | None | None | AI stock picking (paid) |
| **Deployment** | Local | Local/Server | Docker/Cloud | Local | Cloud |

**Key Differences**:

- **vs vnpy**: QuantP focuses on research & validation; vnpy focuses on live execution.
- **vs Freqtrade**: QuantP supports A-share asset types and cost models; Freqtrade specializes in crypto.
- **vs Backtrader**: QuantP uses Backtrader as engine, adding Pipeline / Factor Mining / Paper Trading layers.
- **vs JoinQuant**: JoinQuant is commercial cloud; QuantP runs locally with full control.

---

## Current Limitations

| Limitation | Details | Status |
|------------|---------|:------:|
| A-share live disconnected | vnpy CTP gateway requires VS Build Tools; skeleton only | Blocked |
| Crypto not supported | Strategies/data/gateway are A-share oriented | By design |
| PineScript needs translation | TradingView crawler stores metadata only | Manual |
| Single market | A-shares only; no futures/options/HK/US | Long-term |
| Survivorship bias | Only surviving stocks; delisted data not available | Planned |

---

## Contributing

Issues and Pull Requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Acknowledgements

**Core dependencies**: Backtrader · AkShare · Freqtrade · vnpy · Hikyuu · MyTT · CCXT

**Architecture & workflow references**:
- [openfinclaw-cli](https://github.com/mirror29/openfinclaw-cli) — Strategy package standard format, community strategy workflow
- [markov-hedge-fund-method](https://github.com/jackson-video-resources/markov-hedge-fund-method) — Markov regime detection framework

**Reference project**:
- [Kronos](https://huggingface.co/NeoQuasar/Kronos-small) (NeoQuasar, MIT) — AAAI 2026 quantitative K-line prediction model

---

## References

### Primary Sources
- [Backtrader Docs](https://www.backtrader.com/)
- [VectorBT GitHub](https://github.com/polakowo/vectorbt)
- [vnpy Website](https://www.vnpy.com/)
- [Freqtrade GitHub](https://github.com/freqtrade/freqtrade)
- [AkShare GitHub](https://github.com/akfamily/akshare)
- [Tushare Pro](https://tushare.pro/)
- [Qlib (Microsoft)](https://github.com/microsoft/qlib)
- [CCXT GitHub](https://github.com/ccxt/ccxt)
- [Streamlit](https://streamlit.io/)

### Suggested Reading
1. *Inside the Black Box* — Rishi K. Narang
2. *Python for Finance* — Yves Hilpisch
3. *Quantitative Trading* — Ernie Chan

---

## Disclaimer

1. **Educational Purpose**: For personal learning and research only. Not investment advice.
2. **Risk of Loss**: Quantitative trading carries significant risk of loss, including total loss of principal.
3. **Data Accuracy**: Market data from third-party free sources — not guaranteed.
4. **Backtest Limitations**: Survivorship bias, look-ahead bias, overfitting — live results may differ significantly.
5. **Compliance**: Users must comply with applicable laws and regulations.
6. **Risk Assumption**: All risks borne solely by the user.
7. **No Warranty**: Provided "as is" without warranty of any kind.
8. **Third-Party Services**: Integrated services are responsibility of respective providers.

---

## License

[MIT](LICENSE)
