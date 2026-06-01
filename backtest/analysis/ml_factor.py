"""
LightGBM 因子非线性组合 — ML 分类器学习因子→N日后超额收益的映射。

在 FactorMiner 因子计算完成后使用，滚动窗口训练，输出预测概率+SHAP特征重要性。

依赖: pip install lightgbm shap

用法
--------
>>> from backtest.analysis.ml_factor import FactorML
>>> ml = FactorML(df, factors=["mom_20", "vol_60", "rsi_14"])
>>> result = ml.run(horizon=5, train_window=252)
>>> print(result["feature_importance"])
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.log import get_logger

logger = get_logger("ml_factor")


class FactorML:
    """LightGBM 因子非线性组合器"""

    def __init__(self, df: pd.DataFrame, factors: list[str]):
        """
        参数
        ----------
        df : DataFrame
            OHLCV + 因子值，需含 date/open/high/low/close/volume + 因子列
        factors : list[str]
            用于 ML 模型的因子列名列表
        """
        self.df = df.copy()
        self.factors = [f for f in factors if f in df.columns]
        if len(self.factors) < len(factors):
            missing = set(factors) - set(self.factors)
            logger.warning(f"因子不存在于数据中，已跳过: {missing}")
        if not self.factors:
            raise ValueError("没有可用的因子列")

    def run(self, horizon: int = 5, train_window: int = 252,
            benchmark_col: str = "") -> dict:
        """
        滚动窗口训练 LightGBM 二分类器。

        参数
        ----------
        horizon : int
            预测未来 N 日是否跑赢基准
        train_window : int
            滚动训练窗口长度（交易日）
        benchmark_col : str
            基准收益列名，为空则用 close 自身计算

        返回
        -------
        dict: predictions, feature_importance, shap_values, summary
        """
        # ── 构建标签 ──
        if benchmark_col and benchmark_col in self.df.columns:
            bench_ret = self.df[benchmark_col].pct_change(horizon)
        else:
            bench_ret = self.df["close"].pct_change(horizon)
        stock_ret = self.df["close"].pct_change(horizon)
        excess = stock_ret - bench_ret
        label = (excess > 0).astype(int)
        self.df["_label"] = label

        # ── 准备特征矩阵 ──
        X = self.df[self.factors].copy()
        # 填充缺失值
        X = X.fillna(X.median())

        # ── 滚动窗口预测 ──
        predictions = []
        feature_importances = []
        shap_values_list = []

        min_samples = train_window // 2
        n_total = len(X)

        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm 未安装，请执行: pip install lightgbm")
            return {"predictions": [], "feature_importance": [], "shap_values": [], "summary": "lightgbm 未安装"}

        try:
            import shap as _shap_mod
            _has_shap = True
        except ImportError:
            _has_shap = False

        for t in range(train_window, n_total):
            X_train = X.iloc[t - train_window:t]
            y_train = label.iloc[t - train_window:t].dropna()
            # 对齐
            common_idx = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_idx]
            y_train = y_train.loc[common_idx]

            if len(y_train) < min_samples or y_train.nunique() < 2:
                continue

            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, num_leaves=31,
                learning_rate=0.05, verbose=-1, random_state=42,
            )
            model.fit(X_train, y_train)

            X_pred = X.iloc[t:t + 1]
            prob = model.predict_proba(X_pred)[0][1] if model.classes_[1] == 1 else model.predict_proba(X_pred)[0][0]
            predictions.append({
                "date": str(self.df["date"].iloc[t]),
                "prob": round(float(prob), 4),
                "label": int(label.iloc[t]),
            })

            # 特征重要性
            imp = dict(zip(self.factors, model.feature_importances_))
            imp["date"] = str(self.df["date"].iloc[t])
            feature_importances.append(imp)

            # SHAP（仅最后 20 期以减少计算量）
            if _has_shap and n_total - t <= 20:
                explainer = _shap_mod.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_pred)
                # shap_vals shape: (1, n_features) for binary
                shap_row = {"date": str(self.df["date"].iloc[t])}
                for j, f in enumerate(self.factors):
                    shap_row[f] = round(float(shap_vals[0][j]), 6)
                shap_values_list.append(shap_row)

        # ── 汇总 ──
        pred_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()
        if not pred_df.empty:
            accuracy = (pred_df["prob"].round() == pred_df["label"]).mean()
            mean_prob = pred_df["prob"].mean()
        else:
            accuracy = 0
            mean_prob = 0

        # 平均特征重要性
        if feature_importances:
            avg_imp = {}
            for f in self.factors:
                avg_imp[f] = round(np.mean([fi[f] for fi in feature_importances]), 4)
            sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_imp = []

        summary = (
            f"LightGBM 滚动训练完成: {len(predictions)} 期预测, "
            f"准确率={accuracy:.2%}, 平均概率={mean_prob:.3f}"
        )
        logger.info(summary)

        return {
            "predictions": predictions,
            "feature_importance": sorted_imp,
            "shap_values": shap_values_list,
            "summary": summary,
            "accuracy": round(float(accuracy), 4),
            "horizon": horizon,
            "factors": self.factors,
        }
