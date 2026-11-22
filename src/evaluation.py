"""
evaluation.py
Standardized Evaluation Framework for Financial Transaction Anomaly & Fraud Detection.
Directly implements the comprehensive evaluation criteria specified in Section 6 and
Research Gap 8 (Precision, Recall, F1, PR-AUC, ROC-AUC, FPR, Latency, and Computational Cost).
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float = 0.5,
    is_unsupervised: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a single model against standardized metrics including
    classification performance, latency, and computational cost.
    """
    y_true = np.asarray(y_test, dtype=int)

    # 1. Measure Inference Latency
    t0 = time.perf_counter()
    if is_unsupervised:
        scores = model.predict_anomaly_score(X_test)
        y_prob = scores
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    inference_total_time = time.perf_counter() - t0
    latency_ms_per_txn = (inference_total_time / len(X_test)) * 1000.0

    # 2. Thresholding
    y_pred = (y_prob >= threshold).astype(int)

    # 3. Core Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        fpr, specificity = 0.0, 1.0

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    # PR and ROC curves for visualization
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)

    train_time = getattr(model, "train_time", 0.0)

    return {
        "model_name": name,
        "paradigm": "Unsupervised" if is_unsupervised else ("Hybrid Fusion" if "Hybrid" in name else "Supervised"),
        "threshold": threshold,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "fpr": round(float(fpr), 4),
        "specificity": round(float(specificity), 4),
        "latency_ms_per_txn": round(float(latency_ms_per_txn), 4),
        "throughput_txns_per_sec": int(len(X_test) / max(1e-6, inference_total_time)),
        "train_time_sec": round(float(train_time), 2),
        "pr_curve": {"precision": precision_vals.tolist(), "recall": recall_vals.tolist()},
        "roc_curve": {"fpr": fpr_vals.tolist(), "tpr": tpr_vals.tolist()},
        "confusion_matrix": cm.tolist()
    }


def compare_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    thresholds_dict: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Runs standardized comparative evaluation across a suite of models
    and returns a tidy benchmark summary dataframe.
    """
    records = []
    if thresholds_dict is None:
        thresholds_dict = {}

    for name, model_info in models_dict.items():
        model = model_info["model"]
        is_unsupervised = model_info.get("is_unsupervised", False)
        t = thresholds_dict.get(name, 0.5 if not is_unsupervised else 0.7)
        res = evaluate_model(name, model, X_test, y_test, threshold=t, is_unsupervised=is_unsupervised)
        records.append({
            "Model": res["model_name"],
            "Paradigm": res["paradigm"],
            "PR-AUC (Primary)": res["pr_auc"],
            "ROC-AUC": res["roc_auc"],
            "F1-Score": res["f1"],
            "Recall (Catch Rate)": res["recall"],
            "Precision": res["precision"],
            "False Positive Rate": res["fpr"],
            "Accuracy": res["accuracy"],
            "Latency (ms/txn)": res["latency_ms_per_txn"],
            "Training Time (s)": res["train_time_sec"]
        })

    df_comp = pd.DataFrame(records).sort_values("PR-AUC (Primary)", ascending=False).reset_index(drop=True)
    return df_comp
