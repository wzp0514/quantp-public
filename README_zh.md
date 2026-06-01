# QuantP（量化鲲鹏）— 详细文档

[English](README_en.md) | [简版](README.md)

## 这是什么

QuantP 是一套量化交易研究工具，目标是搭建覆盖从数据获取到模拟交易全流程的可运行管线。项目以学习量化方法论为核心，代码真实可运行，但深度和健壮性与商业产品有显著差距。

**适合谁**：想理解量化交易全流程的开发者、策略研究者、量化初学者。

**当前状态**：回测/分析/验证链路可运行（WFA+DSR+CSCV 多重验证 + FWER 批量校正）；实盘网关为骨架模式，尚未执行过真实交易。

## 系统架构

```
                        ┌───────────────────────────┐
                        │   Streamlit Dashboard (5页)  │
                        └─────────────┬─────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────┐
│                        研究源头 → 管线调度                                │
│                                                                          │
│  6大外部信号源扫描 ──→ Track 并行推进 ──→ 阶段门控 ──→ 晋级/淘汰         │
│  淘汰时自动调策略挖掘器重挖 ──→ 新Track自动注册 ──→ 当场推进              │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────┐
│                     回测引擎 + 策略 + 分析验证                             │
│                                                                          │
│  Backtrader 事件驱动逐Bar模拟                                             │
│  20+策略（4组：内置/社区/实验/导入） · 策略挖掘器(入场×出场组合)            │
│                                                                          │
│  多重验证: WFA · DSR · CSCV PBO · FWER/FDR批量校正                        │
│  因子研究: ~34因子IC/RankIC · LLM因子挖掘(DeepSeek)                       │
│  优化器:   贝叶斯优化(Optuna) · 遗传算法(实验)                             │
│  绩效分析: 归因(Alpha/Beta/R²) · 统计显著性 · 区制检测                    │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────┴───────────────────────────────────┐
│                              数据层                                       │
│                                                                          │
│  AkShare / Tushare / Baostock → ETL → MarketVault / FactorStore          │
│  日频 · 分钟级 │ 复权处理 │ 交易日历 │ 三级降级链                          │
│  存储: Parquet + SQLite + JSON + JSONL                                   │
└─────────────────────────────────────────────────────────────────────────┘

纵切面（辅助模块）:
  ▪ 模拟交易平台 — 逐日模拟 + 守护进程 + 多策略联赛
  ▪ 实盘网关     — vnpy (A股骨架) + CCXT
  ▪ 风控引擎     — 8项硬编码检查 + StrategyGuard 4维熔断
```

> 架构说明：数据层为底部基础 → 回测/分析层在其上做计算 → 管线层负责调度编排 → 仪表盘为最上层展示。

---

## 核心特性

以下功能在代码层面真实可运行，深度与健壮性属学习项目级别，非生产系统。

### 因子与策略研究

- **因子驱动基线**：Pipeline Stage1 自动为每个强因子（|IC|>0.05）创建策略 Track
- **LLM 因子挖掘**：DeepSeek API 生成因子公式 → IC 验证 → FactorStore 持久化
- **研究源头自动扫描**：6 大外部信号源（arXiv 论文 / 数据新鲜度 / 市场结构 / 另类数据 / 策略健康 / 日历事件）
- **截面多因子选股**：多股票面板 → 每日截面排名 → 买 Top N → 定期调仓
- **策略挖掘器**：入场规则 × 出场规则自动组合 → 过拟合过滤 → 排名输出

### 过拟合控制（六重过滤）

| 过滤层 | 方法 | 说明 |
|--------|------|------|
| 1. 逻辑预筛选 | min_trades ≥ 30, 夏普 ≥ 0 | 排除无意义结果 |
| 2. 三步数据分割 | 训练/验证/测试 独立切分 | 杜绝数据泄露 |
| 3. 滚动回测 | WFA (Walk-Forward Analysis) | 检测参数稳定性 |
| 4. 衰退率硬过滤 | 样本外/样本内收益比 ≥ 0.5 | 排除严重过拟合 |
| 5. 批量校正 | FWER (Bonferroni/Holm) + FDR | 多重测试校正 |
| 6. PBO 量化 | CSCV (Combinatorially Symmetric Cross-Validation) | 过拟合概率数值化 |

### 交易与风控

- **多品种成本模型**：股票/ETF/可转债，佣金/印花税/过户费/T+N/涨跌停全部配置驱动
- **8 项硬编码风控**：每项对应真实失败案例，不可手动绕过
- **波动率目标仓位**：高波动自动缩仓、低波动扩仓

### 工程实践

- **Pipeline 闭环**：策略淘汰时立刻调挖掘器重挖 → 新 Track 自动注册 → 当场推进
- **三级数据降级链**：AkShare → Tushare → Baostock，30 分钟自动重试

---

## 快速开始

```bash
git clone https://github.com/quantp/quantp.git
cd quantp
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
# source venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
python interactive.py
```

**前置要求**：Python 3.12+。可选依赖（Tushare Pro Token、vnpy 等）安装失败不影响核心功能。

---

## 使用指南

`python interactive.py` 提供 4 个入口：

```
[1] 5分钟体验    — 零配置快速验证系统是否正常
[2] 自动管线     — 全流程向导（5步配置 → 自动执行 → 看报告）
[3] 单步工具     — 22 项功能，按 5 组排列
[4] 数据管理     — 行情拉取 / 策略仓库 / 源头扫描 / 健康检查
```

### [2] 自动管线向导

5 步完成：选数据（指数/多股票面板）→ 配研究源头（arXiv/另类等）→ 设管线参数（自动恢复/推进阶段）→ 确认执行 → 看报告。

### [3] 单步工具（22 项）

| 分组 | 功能 |
|------|------|
| 因子与策略生成 | 因子分析 / 策略挖掘 / 遗传算法(实验) / 贝叶斯优化(推荐) |
| 回测验证 | 单策略回测 / 策略大比武 / 市场扫描 / 全量回测 / 分钟级回测 |
| 实盘与风控 | 模拟交易 / 风控参数 / 就绪检查 / vnpy（骨架）/ 守护进程 |
| 模拟增强 | 模拟联赛 / 模拟 vs 回测对比 / 模拟守护（长期运行）|
| 监控与分析 | 仪表盘 / 区制检测 / 另类数据 / Agent 环境研判 |

### [4] 数据管理

拉取行情数据 / 策略仓库（导入/搜索/对比）/ 研究源头扫描 / 数据源健康检查 / 本地数据仓库总览

---

## 典型工作流

```
首次验证:  [1] 快速体验 → 确认系统正常
日常研究:  [2] 自动管线向导 → 全部选完一键跑 → 看报告
深入探索:  [3] 单步工具 → 策略挖掘 → Bayes 精调 → 模拟交易
数据维护:  [4] 数据管理 → 拉取行情 → 源头扫描 → 健康检查
```

---

## 项目结构

```
quantp/
├── core/                  # 核心引擎（管线/研究源头/任务调度）
├── backtest/
│   ├── engine/            #   回测引擎 + 策略注册中心
│   ├── strategies/        #   策略层（builtin/community/experimental/imported）
│   ├── analysis/          #   分析工具集（因子/验证/优化/归因/区制/报告）
│   ├── strategy_miner.py  #   策略挖掘器
│   └── strategy_market.py #   策略市场扫描器
├── live/                  # 实盘/模拟交易（网关/执行/风控/监控/守护）
├── data/                  # 数据层（获取/存储/另类数据/清洗）
├── config/                # 配置管理
├── dashboard/             # Streamlit 监控面板
├── strategies_repo/       # 策略仓库管理工具
├── experimental/          # 实验模块
├── notebooks/             # Jupyter 数据探索
├── tests/                 # 测试
├── interactive.py         # 交互菜单
└── run_strategy.py        # 命令行模式
```

---

## 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.12 | 全栈统一 |
| 数据获取 | AkShare + Tushare Pro + Baostock | 三级降级链 |
| 回测引擎 | Backtrader（事件驱动）| AShareCommission 多品种成本模型 |
| 因子引擎 | FactorMiner (~34因子) + LLMFactorMiner (DeepSeek) | IC/RankIC/衰减分析 |
| 优化器 | 贝叶斯优化 (Optuna) + 遗传算法 (实验) | |
| 数据存储 | Parquet + SQLite + JSON + JSONL | 文件即数据库 |
| 可视化 | Streamlit + Matplotlib | Python 原生 |
| AI 集成 | DeepSeek API（因子挖掘/论文提取）| 可选 |

### 支持的品种

| 品种 | 佣金 | 印花税 | T+N | 涨跌停 |
|------|------|--------|-----|:---:|
| 股票 (stock) | 万 2.5 | 万 5 (卖) | T+1 | ±10% |
| ETF | 万 2.5 | 免 | T+1 | ±10%/20% |
| 可转债 (convertible_bond) | 万 0.5 | 免 | T+0 | 无 |

---

## 与同类工具的对比

QuantP 是学习项目，与以下成熟框架在定位、功能深度和适用场景上有本质区别：

| 维度 | QuantP | vnpy | Freqtrade | Backtrader | 聚宽 (JoinQuant) |
|------|--------|------|-----------|------------|-----------------|
| **定位** | 学习研究 | A 股实盘 | 加密货币实盘 | 通用回测框架 | 云端量化平台 |
| **实盘** | 骨架（未连通）| 40+ 券商接口 | 多交易所实盘 | 无内置 | 券商直连 |
| **回测** | Backtrader 封装 | 内置事件驱动 | 内置向量化/事件 | 原生事件驱动 | 云端引擎 |
| **因子研究** | ~34 因子 + LLM 挖掘 | 有限 | 有限 | 无 | 因子库 + 自定义 |
| **策略数量** | 20+ (4组) | 100+ 社区贡献 | 数百社区策略 | 需自行编写 | 社区分享 |
| **过拟合控制** | 六重过滤 + CSCV PBO | 无内置 | 无内置 | 无内置 | 有限 |
| **AI 集成** | LLM 因子挖掘 + Agent 决策 | 无 | 无 | 无 | AI 选股（商业版）|
| **部署** | 本地单机 | 本地/服务器 | Docker/云 | 本地 | 云端（免费额度）|
| **适合人群** | 学习量化方法论 | A 股实盘交易者 | 加密货币交易者 | 策略研究原型 | 不想本地部署的用户 |

**核心差异**：

- **vs vnpy**：QuantP 侧重研究和验证，vnpy 侧重实盘执行。
- **vs Freqtrade**：QuantP 支持 A 股品种和成本模型，Freqtrade 主打加密货币。
- **vs Backtrader**：QuantP 将 Backtrader 作为底层引擎，在其上加了管线/因子挖掘/模拟交易三层抽象。
- **vs 聚宽**：聚宽是商业化云端平台，QuantP 本地运行，完全可控。

### 常用数据源

| 数据源 | 费用 | 覆盖 | 特点 |
|--------|------|------|------|
| **AkShare** | 免费 | A 股/港股/美股/期货/加密等 20+ 品类 | 最广覆盖 |
| **Tushare Pro** | 基础免费→¥200-2000+/年 | A 股为主 | 数据质量高 |
| **Baostock** | 免费 | A 股历史日线 | 稳定低频 |
| **CCXT** | 开源免费 | 100+ 加密交易所 | 加密统一 API 标准 |

---

## 当前局限

| 局限 | 说明 | 状态 |
|------|------|:---:|
| A 股实盘未连通 | vnpy CTP 网关需 VS Build Tools 编译，当前为骨架模式 | 阻塞 |
| 加密货币未支持 | 策略/数据/网关均为 A 股设计 | 设计决策 |
| PineScript 策略需翻译 | TradingView 爬虫仅存元数据，不可直接回测 | 需人工 |
| 单一市场 | 仅沪深 A 股，无期货/期权/港股/美股 | 远期 |
| 幸存者偏差 | 当前数据只有存活股票，不包含退市股 | 计划中 |

---

## 贡献

欢迎提交 Issue 和 Pull Request。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 致谢

本项目在学习过程中参考了以下开源项目：

**核心依赖**：Backtrader · AkShare · Freqtrade · vnpy · Hikyuu · MyTT · CCXT

**架构与工作流参考**：
- [openfinclaw-cli](https://github.com/mirror29/openfinclaw-cli) — 策略包标准格式（`fep.yaml` + `compute(data, context)`）、社区策略工作流
- [markov-hedge-fund-method](https://github.com/jackson-video-resources/markov-hedge-fund-method) — 马尔可夫区制检测框架（Roan @RohOnChain）

**参考项目**：
- [Kronos](https://huggingface.co/NeoQuasar/Kronos-small) (NeoQuasar, MIT) — AAAI 2026 量化K线预测模型

---

## 参考文献

数据截至 2026-05-24，一手来源优先。

### 一手来源
- [Backtrader 官方文档](https://www.backtrader.com/)
- [VectorBT GitHub](https://github.com/polakowo/vectorbt)
- [vnpy 官网](https://www.vnpy.com/) · [vnpy GitHub](https://github.com/vnpy/vnpy)
- [NautilusTrader GitHub](https://github.com/nautechsystems/nautilus_trader)
- [Freqtrade GitHub](https://github.com/freqtrade/freqtrade)
- [AkShare GitHub](https://github.com/akfamily/akshare)
- [Tushare Pro](https://tushare.pro/)
- [Qlib (Microsoft)](https://github.com/microsoft/qlib)
- [聚宽 (JoinQuant)](https://www.joinquant.com/)
- [CCXT GitHub](https://github.com/ccxt/ccxt)
- [Streamlit](https://streamlit.io/)

### 入门读物
1. 《打开量化交易的黑箱》— Rishi K. Narang
2. 《Python 金融大数据分析》— Yves Hilpisch
3. 《量化交易：如何建立自己的算法交易事业》— Ernie Chan

---

## 免责声明

**请在使用前仔细阅读：**

1. **学习研究用途**：本项目仅供个人学习研究使用，不构成任何投资建议、理财推荐或交易指令。
2. **重大亏损风险**：量化交易存在重大亏损风险，可能导致全部本金损失。历史回测表现不代表未来收益。
3. **数据不保真**：行情数据来自第三方免费数据源（AkShare、Tushare、Baostock 等），不保证完整性、准确性和时效性。
4. **回测局限性**：回测结果存在幸存者偏差、未来函数、过拟合等固有局限。实盘结果可能与回测存在显著偏差。
5. **合规责任**：使用者应遵守所在国家/地区法律法规。不同市场对程序化交易有不同的监管要求。
6. **风险自担**：使用本项目或其任何衍生代码进行实盘交易的一切风险和损失，由使用者自行承担。项目作者不对因使用本项目而产生的任何直接或间接损失负责。
7. **无担保**：本项目按"现状"提供，不提供任何明示或暗示的担保，包括但不限于适销性、特定用途适用性或不侵权的担保。
8. **第三方服务风险**：本项目集成的第三方服务（DeepSeek API 等）的可用性、安全性和隐私保护由各自提供商负责。

---

## 许可证

[MIT](LICENSE)
