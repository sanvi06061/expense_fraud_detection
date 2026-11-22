"""
models.py
Comprehensive suite of Supervised, Unsupervised, and Hybrid Anomaly Detection models.
Directly addresses Research Gap 4 (Supervised vs Unsupervised vs Hybrid) and
literature findings (Table 1: Papers [3], [5], [8], [11], [14], [28], [30], [35]).
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


class SupervisedPipeline:
    """Wrapper for supervised models with built-in preprocessing and latency tracking."""
    def __init__(self, model_type: str = "rf", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.train_time = 0.0

    def _build_pipeline(self):
        if self.model_type == "rf":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1
            )
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf)
            ])
        elif self.model_type == "hgb":
            # Fast modern gradient boosting with class weights
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=8,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=self.random_state
            )
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf)
            ])
        elif self.model_type == "lr":
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=self.random_state
            )
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf)
            ])
        else:
            raise ValueError(f"Unknown supervised model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        t0 = time.perf_counter()
        self.pipeline.fit(X, y)
        self.train_time = time.perf_counter() - t0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class UnsupervisedAnomalyDetector:
    """
    Unsupervised Outlier & Anomaly Detection (Table 1: Papers [3], [5], [14], [28]).
    Supports Isolation Forest and Local Outlier Factor (LOF).
    Calibrates continuous anomaly scores to [0, 1] risk probability.
    """
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.02, random_state: int = 42):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = None
        self.train_time = 0.0
        self._score_min = -1.0
        self._score_max = 1.0

    def fit(self, X: pd.DataFrame):
        t0 = time.perf_counter()
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        if self.method == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_scaled)
            # Calibrate score range on training sample
            scores = -self.model.score_samples(X_scaled)
            self._score_min = np.percentile(scores, 1)
            self._score_max = np.percentile(scores, 99)
        elif self.method == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=25,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
            self.model.fit(X_scaled)
            scores = -self.model.score_samples(X_scaled)
            self._score_min = np.percentile(scores, 1)
            self._score_max = np.percentile(scores, 99)
        else:
            raise ValueError(f"Unknown unsupervised method: {self.method}")

        self.train_time = time.perf_counter() - t0
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns normalized anomaly score between 0 and 1 (1 = extreme anomaly)."""
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        # raw score: lower means more anomalous in sklearn; we invert it so higher = anomaly
        raw_scores = -self.model.score_samples(X_scaled)
        # Min-max normalization with clipping
        denom = max(1e-5, self._score_max - self._score_min)
        norm_scores = np.clip((raw_scores - self._score_min) / denom, 0.0, 1.0)
        return norm_scores

    def predict(self, X: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
        scores = self.predict_anomaly_score(X)
        return (scores >= threshold).astype(int)


class HybridFusionPipeline:
    """
    Hybrid Architecture combining Unsupervised Anomaly Scoring + Supervised Boosting.
    Directly reflects Table 1 Paper 11 (Isolation Forest + Boosting) and
    Paper 3 (Unsupervised Learning Fusion).
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.iso_detector = UnsupervisedAnomalyDetector("isolation_forest", contamination=0.02, random_state=random_state)
        self.classifier = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state
        )
        self.imputer = SimpleImputer(strategy="median")
        self.train_time = 0.0
        self.feature_names = []

    def _augment_features(self, X: pd.DataFrame, is_fit: bool = False) -> np.ndarray:
        if is_fit:
            X_clean = self.imputer.fit_transform(X)
            self.iso_detector.fit(X)
        else:
            X_clean = self.imputer.transform(X)

        iso_scores = self.iso_detector.predict_anomaly_score(X).reshape(-1, 1)
        # Combine original features with unsupervised meta-feature
        X_augmented = np.hstack([X_clean, iso_scores])
        return X_augmented

    def fit(self, X: pd.DataFrame, y: pd.Series):
        t0 = time.perf_counter()
        self.feature_names = list(X.columns) + ["unsupervised_iso_anomaly_score"]
        X_aug = self._augment_features(X, is_fit=True)
        self.classifier.fit(X_aug, y)
        self.train_time = time.perf_counter() - t0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aug = self._augment_features(X, is_fit=False)
        return self.classifier.predict_proba(X_aug)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
