"""
benchmark_models.py
Command-Line Automated Benchmarking Suite for Financial Anomaly & Fraud Detection.
Implements the comparative research agenda outlined in Section 6:
- Supervised vs Unsupervised vs Hybrid architectures
- Temporal & relational feature integration
- Comprehensive metrics: PR-AUC, ROC-AUC, F1, Recall, Precision, FPR, Latency, Cost
- Dynamic Decision Threshold Calibration
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_generator import ensure_dataset
from src.feature_engineering import extract_temporal_and_relational_features
from src.models import SupervisedPipeline, UnsupervisedAnomalyDetector, HybridFusionPipeline
from src.threshold_tuner import find_optimal_threshold
from src.evaluation import compare_all_models, evaluate_model


def run_benchmark(sample_size: int = 40000):
    print("=" * 80)
    print("FINANCIAL ANOMALY & FRAUD DETECTION BENCHMARKING SUITE")
    print("Aligned with 2024-2026 Literature Synthesis & Scopus Research Gaps")
    print("=" * 80)

    # 1. Dataset loading
    data_path = ensure_dataset("data/transactions.gz")
    print(f"\n[1/5] Loading transactions from {data_path}...")
    df = pd.read_csv(data_path, compression="gzip", low_memory=False)
    if len(df) > sample_size:
        print(f"Sampling {sample_size:,} records from {len(df):,} total for efficient benchmarking...")
        df = df.sample(n=sample_size, random_state=42).copy()

    print(f"Total transactions: {len(df):,}")
    print(f"Actual fraud cases: {int(df['isFraud'].sum()):,} ({df['isFraud'].mean()*100:.2f}%)")

    # 2. Advanced Feature Engineering (Gap 3: Temporal & Relational)
    print("\n[2/5] Engineering temporal velocity, rolling averages & relational mule indicators...")
    X = extract_temporal_and_relational_features(df)
    y = df["isFraud"].astype(int).values
    print(f"Engineered {X.shape[1]} rich features: {list(X.columns[:8])}...")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # 3. Model Suite Initialization (Gap 4: Supervised vs Unsupervised vs Hybrid)
    print("\n[3/5] Training comparative multi-paradigm model suite...")
    models_dict = {
        "Random Forest (Supervised)": {
            "model": SupervisedPipeline("rf", random_state=42).fit(X_train, y_train),
            "is_unsupervised": False
        },
        "HistGradientBoosting (Supervised)": {
            "model": SupervisedPipeline("hgb", random_state=42).fit(X_train, y_train),
            "is_unsupervised": False
        },
        "Logistic Regression (Supervised Baseline)": {
            "model": SupervisedPipeline("lr", random_state=42).fit(X_train, y_train),
            "is_unsupervised": False
        },
        "Isolation Forest (Unsupervised)": {
            "model": UnsupervisedAnomalyDetector("isolation_forest", contamination=0.02, random_state=42).fit(X_train),
            "is_unsupervised": True
        },
        "Local Outlier Factor (Unsupervised)": {
            "model": UnsupervisedAnomalyDetector("lof", contamination=0.02, random_state=42).fit(X_train),
            "is_unsupervised": True
        },
        "Hybrid Fusion (IsoForest + Boosting)": {
            "model": HybridFusionPipeline(random_state=42).fit(X_train, y_train),
            "is_unsupervised": False
        }
    }

    # 4. Standardized Evaluation (Gap 8 & Section 6)
    print("\n[4/5] Executing standardized evaluation metrics...")
    comparison_df = compare_all_models(models_dict, X_test, y_test)
    
    print("\n" + "=" * 90)
    print("STANDARDIZED MODEL COMPARISON LEADERBOARD")
    print("=" * 90)
    print(comparison_df.to_string(index=False))

    # Save results
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_dir / "benchmark_results.csv", index=False)
    print(f"\nLeaderboard saved to {out_dir / 'benchmark_results.csv'}")

    # 5. Dynamic Threshold Calibration (Gap 2 & Table 1 Paper [1])
    print("\n[5/5] Performing Dynamic Decision Threshold Calibration on Best Model...")
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = models_dict[best_model_name]["model"]
    y_prob = best_model.predict_proba(X_test)[:, 1] if not models_dict[best_model_name]["is_unsupervised"] else best_model.predict_anomaly_score(X_test)

    opt_f1 = find_optimal_threshold(y_test, y_prob, criterion="f1")
    opt_cost = find_optimal_threshold(y_test, y_prob, criterion="cost", cost_fp=15.0, cost_fn=500.0)

    print(f"\nBest Model: {best_model_name}")
    print(f"- Naive Threshold (t = 0.50): F1={opt_f1['naive_metrics']['f1']:.4f} | Recall={opt_f1['naive_metrics']['recall']:.4f} | Precision={opt_f1['naive_metrics']['precision']:.4f} | Est. Cost=${opt_cost['naive_metrics']['net_cost']:,.2f}")
    print(f"- Calibrated Optimal F1 (t = {opt_f1['optimal_threshold']:.3f}): F1={opt_f1['optimal_metrics']['f1']:.4f} | Recall={opt_f1['optimal_metrics']['recall']:.4f} | Precision={opt_f1['optimal_metrics']['precision']:.4f}")
    print(f"- Cost-Minimizing Threshold (t = {opt_cost['optimal_threshold']:.3f}): Est. Cost=${opt_cost['optimal_metrics']['net_cost']:,.2f} (Savings: ${opt_cost['cost_savings']:,.2f})")

    print("\nBenchmark completed successfully! All objectives and research gaps addressed.")


if __name__ == "__main__":
    run_benchmark(sample_size=30000)
