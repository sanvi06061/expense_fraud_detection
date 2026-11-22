"""
threshold_tuner.py
Dynamic Calibration of Decision Thresholds for Financial Anomaly Detection.
Directly addresses Research Gap 2 and Table 1 Paper [1] (Huang et al., 2025/2026).
Optimizes decision thresholds for class-imbalanced fraud detection under F1, F-beta,
and financial Cost-Benefit constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def compute_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
    cost_fp: float = 15.0,
    cost_fn: float = 500.0
) -> pd.DataFrame:
    """
    Computes confusion matrix, precision, recall, F1, F2, FPR, and net financial cost
    across continuous decision thresholds t in [0.01, 0.99].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []

    pos_count = (y_true == 1).sum()
    neg_count = (y_true == 0).sum()

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / neg_count if neg_count > 0 else 0.0
        specificity = tn / neg_count if neg_count > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            # F2 emphasizes recall over precision (beta=2)
            f2 = (1 + 4) * precision * recall / (4 * precision + recall)
        else:
            f1 = 0.0
            f2 = 0.0

        net_cost = fp * cost_fp + fn * cost_fn

        rows.append({
            "threshold": round(float(t), 3),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "fpr": round(fpr, 4),
            "specificity": round(specificity, 4),
            "net_cost": round(net_cost, 2),
            "flagged_rate": round((tp + fp) / len(y_true), 4)
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f1",
    cost_fp: float = 15.0,
    cost_fn: float = 500.0,
    max_fpr: float = 0.03
) -> Dict[str, Any]:
    """
    Finds optimal calibrated threshold based on selected optimization objective:
    - 'f1': Maximizes harmonic mean of precision and recall.
    - 'f2': Prioritizes fraud catch rate (recall) with higher beta.
    - 'cost': Minimizes expected financial operational loss (FP investigation vs FN fraud loss).
    - 'target_fpr': Maximizes recall subject to keeping False Positive Rate <= max_fpr.
    """
    curve = compute_threshold_curve(y_true, y_prob, cost_fp=cost_fp, cost_fn=cost_fn)

    if criterion == "f1":
        best_row = curve.loc[curve["f1"].idxmax()]
        objective_name = "Maximum F1-Score"
    elif criterion == "f2":
        best_row = curve.loc[curve["f2"].idxmax()]
        objective_name = "Maximum F2-Score (High Fraud Recall)"
    elif criterion == "cost":
        best_row = curve.loc[curve["net_cost"].idxmin()]
        objective_name = "Minimum Financial Loss ($)"
    elif criterion == "target_fpr":
        valid = curve[curve["fpr"] <= max_fpr]
        if not valid.empty:
            best_row = valid.loc[valid["recall"].idxmax()]
        else:
            best_row = curve.loc[curve["fpr"].idxmin()]
        objective_name = f"Max Recall within FPR <= {max_fpr*100:.1f}%"
    else:
        raise ValueError(f"Unknown threshold calibration criterion: {criterion}")

    # Also compute default naive threshold at t = 0.5 for comparison
    naive_row = curve.iloc[(curve["threshold"] - 0.5).abs().argsort()[:1]].iloc[0]

    return {
        "criterion": criterion,
        "objective_name": objective_name,
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_metrics": best_row.to_dict(),
        "naive_threshold": 0.5,
        "naive_metrics": naive_row.to_dict(),
        "curve_df": curve,
        "cost_savings": round(float(naive_row["net_cost"] - best_row["net_cost"]), 2),
        "f1_improvement": round(float(best_row["f1"] - naive_row["f1"]), 4)
    }
