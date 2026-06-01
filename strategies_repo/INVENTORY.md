# 策略资产盘点 (S1)

> 盘点日期：2026-05-26
> 后续步骤：S2 绩效审查 → S3 三级分类 → S4 外来策略清理 → S5 健康追踪

---

## 一、可运行策略（16 个）

已注册于 `backtest/strategy_market.py` → `ALL_STRATEGIES`，每个都有可执行的 Python 策略类。

### 1.1 内置策略（5 个）— source: builtin

| # | 策略名 | 类型 | 参数 | 说明 |
|---|--------|------|------|------|
| 1 | 双均线交叉 | 趋势跟踪 | fast=5, slow=20 | 快线上穿慢线买入，下穿卖出 |
| 2 | 布林带回归 | 均值回归 | period=20, devfactor=2.0 | 触下轨买入，回中轨卖出 |
| 3 | 动量策略 | 趋势跟踪 | lookback=126, hold=21 | 买过去涨最多的，持有到期卖出 |
| 4 | 均值回归 | 均值回归 | period=20, threshold=2.0 | 超跌买入，回归均线卖出 |
| 5 | 网格交易 | 震荡 | grid=10, ±10% | 价格区间内低买高卖赚波动 |

### 1.2 社区导入策略（6 个）— source: imported

| # | 策略名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 6 | vnpy双均线 | 趋势跟踪 | vnpy 社区 | 经典双均线交叉 |
| 7 | 海龟交易 | 突破 | Richard Dennis, 1983 | 突破20日高点买入，跌破10日低点卖出 |
| 8 | Freqtrade RSI | 均值回归 | Freqtrade 社区 | RSI<30超卖买入，RSI>70超买卖出 |
| 9 | 聚宽MACD | 趋势跟踪 | 聚宽社区 | MACD金叉买入，死叉卖出 |
| 10 | Donchian通道 | 突破 | Backtrader 社区 | 突破N日最高价买入，跌破N日最低价卖出 |
| 11 | ATR移动止损 | 趋势跟踪 | 量化经典模式 | 均线入场 + ATR动态止损 |

### 1.3 实验性策略（3 个）— source: experimental

| # | 策略名 | 类型 | 说明 |
|---|--------|------|------|
| 12 | 多信号共振 | 共振 | 多信号共振+区制过滤+因子嵌入 |
| 13 | 因子驱动策略 | 多因子 | 因子得分驱动交易，波动率目标仓位 |
| 14 | 截面多因子选股 | 多因子 | 多股票面板截面排名，买Top N定期调仓 |

### 1.4 FEP 引擎无关策略（1 个）— source: manual

| # | 策略名 | 类型 | 来源 | 说明 |
|---|--------|------|------|------|
| 15 | 双均线交叉(fep) | 趋势跟踪 | strategies_repo/custom/ma_cross | 引擎无关格式，UniversalStrategy 适配 |

### 1.5 自建策略（1 个）— source: custom

| # | 策略名 | 类型 | 位置 | 说明 |
|---|--------|------|------|------|
| 16 | ma_cross | 趋势跟踪 | strategies_repo/custom/ma_cross/ | fep 格式自建示例 |

---

## 二、文件系统策略库（604 个目录）

位于 `strategies_repo/market/`，每个目录含 `meta.yaml` + `strategy.py` + `result.json`。
大部分无对应的 Backtrader 策略类，仅在文件系统层面存在。

### 2.1 按来源分类

| 来源 | 数量 | 命名前缀 | 说明 |
|------|------|----------|------|
| Freqtrade (v1) | ~73 | `local_freqtrade_XXX` | 第一版 Freqtrade 加密策略 |
| Freqtrade (v2) | ~458 | `local_freqtrade_v2_XXX` | 第二版，最大子集（含 NFI/Schism/Elliot 等系列） |
| Freqtrade (v3) | ~50 | `local_freqtrade_v3_XXX` | 第三版 |
| Freqtrade (WTC) | 1 | `local_freqtrade_wtc` | WTC 变体 |
| vnpy CTA | ~10 | `local_vnpy_cta_XXX` | vnpy CTA 策略 |
| vnpy Main | 1 | `local_vnpy_main_XXX` | vnpy 主策略 |
| 内置直导 | 11 | 中文名（见 1.1-1.2） | import_all_builtin() 导入 |
| **合计** | **604** | | |

### 2.2 关键发现

- 所有 meta.yaml 的 `last_backtest` 均为 null——策略库中的策略从未在统一框架下正式回测
- `active/` `candidate/` `archived/` 三级归档目录尚未创建（代码已预留 `classify()` 方法）
- 582 个 Freqtrade 策略为加密策略，需要评估是否适合 A 股

---

## 三、策略类型分布（可运行策略 16 个）

| 类型 | 数量 | 策略 |
|------|------|------|
| 趋势跟踪 | 7 | 双均线交叉、动量策略、vnpy双均线、聚宽MACD、ATR移动止损、双均线交叉(fep)、ma_cross |
| 均值回归 | 3 | 布林带回归、均值回归、Freqtrade RSI |
| 突破 | 2 | 海龟交易、Donchian通道 |
| 多因子 | 2 | 因子驱动策略、截面多因子选股 |
| 震荡 | 1 | 网格交易 |
| 共振 | 1 | 多信号共振 |

---

## 四、标准化绩效矩阵 (S2)

> 回测参数：沪深300 2015-2023 日线 (2189行)，初始资金 100,000 元，无杠杆
> 回测日期：2026-05-26

| 策略 | 年化 | 回撤 | 夏普 | Calmar | 交易 | 胜率 | 终值 |
|------|------|------|------|--------|------|------|------|
| ATR移动止损 | 5.8% | 17.4% | 0.31 | 0.33 | 34 | 56% | 162,935 |
| Donchian通道 | 3.1% | 28.9% | 0.10 | 0.11 | 21 | 33% | 130,663 |
| 网格交易 | 1.4% | 26.2% | -0.04 | 0.05 | 1 | 0% | 112,686 |
| 海龟交易 | 0.7% | 19.1% | -0.23 | 0.04 | 32 | 38% | 106,393 |
| 多信号共振 | -0.2% | 37.2% | -0.07 | -0.01 | 30 | 37% | 98,324 |
| 布林带回归 | -1.3% | 26.9% | -0.54 | -0.05 | 34 | 56% | 89,183 |
| 均值回归 | -1.3% | 26.9% | -0.54 | -0.05 | 34 | 56% | 89,183 |
| 双均线交叉 | -2.7% | 44.3% | -0.31 | -0.06 | 66 | 33% | 79,160 |
| vnpy双均线 | -2.7% | 44.3% | -0.31 | -0.06 | 66 | 33% | 79,160 |
| 聚宽MACD | -3.5% | 39.2% | -0.26 | -0.09 | 82 | 29% | 73,513 |
| 动量策略 | — | — | — | — | 0 | — | 无交易信号 |
| 因子驱动策略 | — | — | — | — | — | — | 需 _score_series 参数 |
| Freqtrade RSI | — | — | — | — | — | — | 待回测 |
| 截面多因子选股 | — | — | — | — | — | — | 需多股票 panel 数据 |
| 双均线交叉(fep) | — | — | — | — | — | — | fep 需单独加载 |

### 4.1 关键发现

- **仅 ATR移动止损 夏普 > 0**（0.31），其余策略 2015-2023 年化收益均不理想
- 沪深300 2015-2023 区间本身几乎零收益（从 ~3500 到 ~3400），择时策略在这一阶段面临结构性困难
- vnpy双均线 = 双均线交叉（同为 MaCrossStrategy fast=5 slow=20）
- 布林带回归 ≈ 均值回归 结果高度相似，需审查是否使用了不同策略类
- 网格交易仅 1 笔交易（参数 grid=10 但波动不够触发网格）
- 动量策略 0 交易（lookback=126, hold=21 在震荡下跌市中无符合条件的信号）
- 因子驱动策略需要 `_score_series` 参数，无法通过 BatchRunner 直接回测

---

## 五、Python 策略文件清单

```
backtest/strategies/
├── ma_cross.py              # MaCrossStrategy
├── bollinger.py             # BollingerStrategy
├── momentum.py              # MomentumStrategy
├── mean_revert.py           # MeanRevertStrategy
├── grid.py                  # GridStrategy
├── community.py             # 6 个社区策略 + MARKET_STRATEGIES 字典
├── resonance.py             # ResonanceStrategy（实验性）
├── factor_strategy.py       # FactorStrategy（实验性）
├── cross_section_strategy.py # CrossSectionStrategy（实验性）

strategies_repo/
├── market/                  # 604 个策略目录
├── custom/ma_cross/         # 1 个自建 fep 策略
├── mined/                   # 空（挖掘策略待产出）
├── active/                  # 待创建（S3）
├── candidate/               # 待创建（S3）
├── archived/                # 待创建（S3）
├── repo.py                  # StrategyRepo 核心
├── crawler.py / importer.py / sync.py / scoring.py / export.py
└── .sources.json            # 外部源注册表
```
