"""
L4 ML信号链路 — RL+另类数据+LightGBM因子三源融合。

真实调用: AlternativeData管道 + RLTrainer + FactorML → 加权综合信号。

用法
--------
>>> from backtest.analysis.l4_integration import L4SignalChain
>>> l4 = L4SignalChain(df)
>>> result = l4.run()  # 默认只跑 alt + ml，跳过RL训练
>>> print(result["action"], result["confidence"])

>>> l4 = L4SignalChain(df, use_rl=True, model_cache_dir="data/vault/vault_data/models")
>>> result = l4.run()  # 含RL训练（首次慢，之后读缓存）

贝叶斯权重更新
--------
>>> l4.update_posterior(trade_won=True, signal_source="ml")  # 交易获胜，ML信号+1
>>> w = l4.get_posterior_weights()  # 获取后验权重
>>> result = l4.run()  # 自动使用后验权重
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("l4_integration")


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


class L4SignalChain:
    """
    L4 多模态信号链（真实实现）。

    三个信号源加权融合：
      - 另类数据（AlternativeData管道）→ 权重 0.3
      - RL 交易器（PPO训练/缓存加载）→ 权重 0.3
      - LightGBM 因子预测（FactorML）→ 权重 0.4

    RL训练默认关闭（耗时5-10分钟），开启后首次训练自动缓存模型。

    贝叶斯权重更新：每次交易结果出来后调用 update_posterior()，
    beta-二项共轭模型动态调整三信号源权重。冷启动用固定权重作 Beta 先验。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        use_rl: bool = False,
        use_alt: bool = True,
        use_ml: bool = True,
        use_kronos: bool = False,
        model_cache_dir: str = "data/vault/vault_data/models",
        rl_timesteps: int = 30000,
        symbol: str = "",
    ):
        self.df = df.copy()
        self.use_rl = use_rl
        self.use_alt = use_alt
        self.use_ml = use_ml
        self.use_kronos = use_kronos
        self.model_cache_dir = str(Path(model_cache_dir))
        self.rl_timesteps = rl_timesteps
        self.symbol = symbol

        # 预计算因子（共享给 RL 和 FactorML）
        self.factor_df = None
        self.factor_names: list[str] = []
        self._init_factors()

        # RL 模型
        self.rl_model = None
        self.rl_signal_series: Optional[pd.Series] = None
        if use_rl:
            self._init_rl()

        # Kronos 引擎（延迟加载）
        self._kronos_engine = None

        # 贝叶斯权重：Beta(alpha, beta) 先验 = 固定权重 * 10
        self._prior_alpha = {"alt": 3.0, "rl": 3.0, "ml": 4.0, "kronos": 2.5}
        self._prior_beta = {"alt": 7.0, "rl": 7.0, "ml": 6.0, "kronos": 7.5}
        self._posterior_alpha = dict(self._prior_alpha)
        self._posterior_beta = dict(self._prior_beta)
        self._update_count = 0

    def _init_factors(self):
        """预计算因子（FactorMiner）并缓存"""
        try:
            from backtest.analysis.factor_miner import compute_factors
            self.factor_df = compute_factors(self.df)
            self.factor_names = [
                c for c in self.factor_df.columns
                if c not in ("date", "open", "high", "low", "close", "volume")
            ]
            logger.info(f"因子计算完成: {len(self.factor_names)} 个")
        except Exception as e:
            logger.warning(f"因子计算失败，跳过 ML/RL 信号: {e}")
            self.factor_df = self.df.copy()

    def _init_rl(self):
        """初始化 RL 模型（加载缓存或标记待训练）"""
        cache_path = Path(self.model_cache_dir) / "l4_rl_model.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self.rl_model = pickle.load(f)
                logger.info(f"RL模型已从缓存加载: {cache_path}")
            except Exception as e:
                logger.warning(f"RL模型缓存加载失败: {e}，将重新训练")

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def run(self, weights: tuple | None = None) -> dict:
        """
        运行完整 L4 信号链。

        权重为 None 时自动使用后验权重（如有更新则用后验，否则用先验）。

        返回
        -------
        dict: signal, action, confidence, components, summary
        """
        if weights is None:
            weights = self.get_posterior_weights()

        signals = {}

        if self.use_alt:
            signals["alt"] = self._alt_data_score()
        else:
            signals["alt"] = 0.5

        if self.use_rl:
            signals["rl"] = self._rl_score()
        else:
            signals["rl"] = 0.5

        if self.use_ml:
            signals["ml"] = self._factor_ml_score()
        else:
            signals["ml"] = 0.5

        if self.use_kronos:
            signals["kronos"] = self._kronos_score()
        else:
            signals["kronos"] = 0.5

        # 加权融合
        w = weights
        composite = (
            w[0] * signals["alt"] + w[1] * signals["rl"] +
            w[2] * signals["ml"] + w[3] * signals["kronos"]
        )

        if composite > 0.6:
            action, confidence = "buy", composite
        elif composite < 0.4:
            action, confidence = "sell", 1 - composite
        else:
            action, confidence = "hold", 1 - abs(composite - 0.5) * 2

        result = {
            "signal": round(composite, 4),
            "action": action,
            "confidence": round(confidence, 4),
            "components": {
                k: round(v, 4) for k, v in signals.items()
            },
            "summary": (
                f"L4综合信号={composite:.3f} | "
                f"另类={signals['alt']:.3f} RL={signals['rl']:.3f} ML={signals['ml']:.3f} | "
                f"决策={action}(置信度{confidence:.2f})"
            ),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(result["summary"])
        return result

    def update_posterior(self, trade_won: bool, signal_source: str):
        """Beta-binomial conjugate update for signal source posterior.

        trade_won=True increments alpha, else increments beta.
        signal_source must be one of: alt, rl, ml.
        """
        if signal_source not in self._posterior_alpha:
            logger.warning(f"Unknown signal source: {signal_source}, skip update")
            return
        if trade_won:
            self._posterior_alpha[signal_source] += 1
        else:
            self._posterior_beta[signal_source] += 1
        self._update_count += 1
        w = self.get_posterior_weights()
        logger.info(
            f"Bayesian update #{self._update_count}: {signal_source} "
            f"{'win' if trade_won else 'loss'} -> posterior weights={w}"
        )

    def get_posterior_weights(self) -> tuple[float, float, float, float]:
        """Return posterior-mean-normalized weights for the four signal sources."""
        means = {}
        for k in ["alt", "rl", "ml", "kronos"]:
            a = self._posterior_alpha[k]
            b = self._posterior_beta[k]
            means[k] = a / (a + b)
        total = sum(means.values())
        if total == 0:
            return (0.3, 0.3, 0.25, 0.15)
        return (
            round(means["alt"] / total, 4),
            round(means["rl"] / total, 4),
            round(means["ml"] / total, 4),
            round(means["kronos"] / total, 4),
        )

    def reset_posterior(self):
        """Reset posterior distributions to prior."""
        self._posterior_alpha = dict(self._prior_alpha)
        self._posterior_beta = dict(self._prior_beta)
        self._update_count = 0
        logger.info("Posterior reset to prior")

    # ═══════════════════════════════════════════════════════════
    # 信号子组件
    # ═══════════════════════════════════════════════════════════

    def _alt_data_score(self) -> float:
        """另类数据管道 → 归一化分数 (0=极度风险, 1=极度乐观)"""
        try:
            from data.alternative.pipeline import AlternativeData
            ad = AlternativeData()
            result = ad.full_scan(symbol=self.symbol)

            # risk_score: 0-100（高=危险），反转
            risk = result.get("risk_score", 50) / 100.0
            risk_signal = 1.0 - risk

            # sentiment_score: -1~1，映射到 0~1
            sentiment = result.get("sentiment_score", 0)
            sentiment_signal = (sentiment + 1) / 2

            # position_multiplier: 0~1
            position = result.get("position_multiplier", 0.5)

            score = risk_signal * 0.3 + sentiment_signal * 0.3 + position * 0.4
            logger.info(
                f"另类数据: risk={risk:.2f} sent={sentiment_signal:.2f} "
                f"pos={position:.2f} → score={score:.3f}"
            )
            return _normalize(score)
        except Exception as e:
            logger.warning(f"另类数据获取失败: {e}，返回中性值 0.5")
            return 0.5

    def _rl_score(self) -> float:
        """RL交易器 → 预测信号归一化 (0=看空, 1=看多)"""
        if self.rl_model is not None:
            return self._rl_predict_from_model()
        return self._rl_train_and_cache()

    def _rl_predict_from_model(self) -> float:
        """用已加载的模型逐 bar 预测，返回信号序列"""
        try:
            from experimental.rl_trader import TradingEnv

            env = TradingEnv(self.df, initial_cash=100000)
            obs = env.reset()
            done = False
            actions = []

            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=True)
                actions.append(action)
                obs, _, done, _ = env.step(action)

            if not actions:
                return 0.5

            # action: 0=hold, 1=buy, 2=sell
            buy_ratio = sum(1 for a in actions if a == 1) / len(actions)
            sell_ratio = sum(1 for a in actions if a == 2) / len(actions)

            # 买入多→偏多，卖出多→偏空
            score = 0.5 + buy_ratio * 0.3 - sell_ratio * 0.3
            # 最后一步的动作权重更高
            last = actions[-1]
            if last == 1:
                score += 0.1
            elif last == 2:
                score -= 0.1

            logger.info(f"RL信号: buy={buy_ratio:.1%} sell={sell_ratio:.1%} → score={score:.3f}")
            self.rl_signal_series = pd.Series(actions)  # 缓存供外部使用
            return _normalize(score)
        except Exception as e:
            logger.warning(f"RL预测失败: {e}，返回中性值 0.5")
            return 0.5

    def _rl_train_and_cache(self) -> float:
        """训练 RL 模型并缓存"""
        try:
            from experimental.rl_trader import RLTrainer

            logger.info(f"RL训练启动 ({self.rl_timesteps}步)...")
            trainer = RLTrainer(self.df, cash=100000)
            result = trainer.train(timesteps=self.rl_timesteps)
            self.rl_model = trainer.model

            # 缓存
            cache_path = Path(_ensure_dir(self.model_cache_dir)) / "l4_rl_model.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(self.rl_model, f)
            logger.info(f"RL模型已缓存: {cache_path}")

            test_ret = result.get("test_return", 0)
            # 收益率映射到信号
            score = 0.5 + test_ret * 2
            logger.info(f"RL训练完成: 测试收益={test_ret:.2%} → score={score:.3f}")
            return _normalize(score)
        except ImportError:
            logger.warning("stable-baselines3 未安装，RL信号不可用")
            return 0.5
        except Exception as e:
            logger.warning(f"RL训练失败: {e}，返回中性值 0.5")
            return 0.5

    def _factor_ml_score(self) -> float:
        """LightGBM因子预测 → 最新一期概率"""
        if not self.factor_names:
            logger.warning("无可用因子，ML信号返回中性值")
            return 0.5

        try:
            from backtest.analysis.ml_factor import FactorML

            ml = FactorML(self.factor_df, factors=self.factor_names[:10])
            ml_result = ml.run(horizon=5, train_window=min(252, len(self.factor_df) // 2))

            predictions = ml_result.get("predictions", [])
            if not predictions:
                logger.warning("LightGBM无有效预测，返回中性值")
                return 0.5

            # 最新一期预测概率
            latest_prob = predictions[-1]["prob"]
            accuracy = ml_result.get("accuracy", 0.5)

            # 用准确率调整置信度：高准确率→信号更极端，低准确率→更接近中性
            adj = (latest_prob - 0.5) * min(accuracy * 2, 1.0) + 0.5

            logger.info(
                f"ML因子: prob={latest_prob:.3f} accuracy={accuracy:.1%} → score={adj:.3f}"
            )
            return _normalize(adj)
        except ImportError:
            logger.warning("lightgbm 未安装，ML信号不可用")
            return 0.5
        except Exception as e:
            logger.warning(f"FactorML运行失败: {e}，返回中性值 0.5")
            return 0.5

    def _kronos_score(self) -> float:
        """Kronos 量化模型预测 → 预测方向 + 置信度"""
        try:
            from core.kronos.engine import KronosEngine
            from config.loader import get_kronos_config
        except ImportError as e:
            logger.debug(f"Kronos 导入失败: {e}")
            return 0.5

        cfg = get_kronos_config()
        if not cfg.get("l4_agent", False):
            return 0.5

        if self._kronos_engine is None:
            try:
                self._kronos_engine = KronosEngine(
                    model_size=cfg.get("model", "small"),
                    device=cfg.get("device", "auto"),
                    tokenizer_path=cfg.get("tokenizer_path", ""),
                    model_path=cfg.get("model_path", ""),
                    project_path=cfg.get("project_path", ""),
                )
            except Exception as e:
                logger.warning(f"Kronos 引擎创建失败: {e}")
                return 0.5

        try:
            lookback = cfg.get("lookback", 60)
            pred_len = cfg.get("pred_len", 20)
            if len(self.df) < lookback:
                return 0.5

            window = self.df.tail(lookback).copy()
            x_ts = pd.Series(window.index)
            last_ts = window.index[-1]
            y_ts = pd.Series(pd.date_range(last_ts, periods=pred_len + 1, freq="B")[1:])

            pred = self._kronos_engine.predict(window, x_ts, y_ts, pred_len=pred_len)
            last_close = window["close"].iloc[-1]
            pred_close = pred["close"].values[-1]
            forecast_return = (pred_close / last_close - 1) if last_close > 0 else 0.0

            # 映射收益率到 [0, 1] 信号
            if forecast_return > 0.05:
                score = 0.8
            elif forecast_return > 0.02:
                score = 0.65
            elif forecast_return > 0:
                score = 0.55
            elif forecast_return > -0.02:
                score = 0.45
            elif forecast_return > -0.05:
                score = 0.35
            else:
                score = 0.2

            logger.info(
                f"Kronos: return={forecast_return:.4f} → score={score:.3f}"
            )
            return _normalize(score)
        except Exception as e:
            logger.warning(f"Kronos 预测失败: {e}")
            return 0.5

    # ═══════════════════════════════════════════════════════════
    # 兼容旧接口
    # ═══════════════════════════════════════════════════════════

    def generate(self, weights: tuple | None = None) -> dict:
        """旧接口兼容，等同于 run()"""
        return self.run(weights=weights)
