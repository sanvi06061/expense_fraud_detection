# 🛡️ Expense Anomaly & Fraud Detection System

**A Research-Grade Multi-Paradigm Machine Learning and AI Framework for Financial Transaction Risk Assessment**

Synthesizing bibliometric and empirical insights from **46 Scopus studies (2024–2026)** in financial fraud detection, anomaly detection, cybersecurity, and transaction risk.

---

## 🔬 Alignment with Research Gaps & Objectives

This project systematically addresses the **8 core research gaps** and the research direction outlined in the 2024–2026 literature review:

| Research Gap | Focus in Literature | Project Implementation | Key Literature Reference |
| :--- | :--- | :--- | :--- |
| **Gap 1: Cross-Domain Generalization** | Model transferability across varying banking, card, and payment schemas | Multi-scenario benchmark data generator (`src/data_generator.py`) with automated fallback and schema resilience | Corpus Synthesis (2024–2026) |
| **Gap 2: Dynamic Decision Thresholds** | Static $t=0.5$ fails under severe class imbalance (<2.5% fraud) | Dynamic Decision Threshold Calibration module (`src/threshold_tuner.py`) optimizing for $F_1$, $F_\beta$, and net financial loss ($C_{FP}$ vs $C_{FN}$) | Huang et al. (2025/2026) [1] |
| **Gap 3: Temporal & Relational Information** | Beyond isolated rows to sequences and account networks | Feature engineering module (`src/feature_engineering.py`) with inter-transaction intervals ($\Delta t$), 1h/24h velocity bursts, rolling spend deviation, and composite mule account risk | Patle et al. (2026) *Mule Trace* [2], Kavitha et al. (2026) [18]/[30] |
| **Gap 4: Supervised vs Unsupervised vs Hybrid** | Systematic comparison between label-dependent, label-light, and fused methods | Benchmarking suite (`src/models.py`, `src/evaluation.py`) comparing Random Forest, HistGradientBoosting, Logistic Regression, Isolation Forest, Local Outlier Factor (LOF), and Hybrid Fusion | Abbassi et al. (2024) [3], Hitha & Dubey (2026) [8], Mohanakrishnan (2026) [11] |
| **Gap 5: Explainable AI (XAI)** | Transparent, actionable human decision support beyond static rules | Explainability engine (`src/explainability.py`) computing global feature importances, local instance attribution waterfall plots, and plain-English audit memorandums | Mohanakrishnan (2026) [11], Amiri et al. (2026) [15], Kungu et al. (2026) [17] |
| **Gap 6: Real-Time & Cost-Aware Deployment** | Operational latency and compute resource constraints | Two-tier cost-aware screening pipeline (`src/stream_simulator.py`) using sub-millisecond Tier 1 filtering for 85%+ benign events | Abbassi et al. (2024) [3], Żelasko (2026) [12] |
| **Gap 7: Privacy & Security** | Anonymized distributed transaction processing | Decoupled feature abstraction, tokenized customer/merchant IDs, and zero-leakage test splits | Xu et al. (2026) [25] |
| **Gap 8: Standardized Evaluation** | Comprehensive metrics beyond plain accuracy | Standardized evaluation framework reporting PR-AUC, ROC-AUC, $F_1$, Recall, Precision, FPR, inference latency (ms/txn), and training wall-clock time | Section 6 Proposed Research Direction |

---

## 🏗️ System Architecture

```text
expense_fraud_detection_project/
├── data/
│   └── transactions.gz        # Gzip compressed transactions (Kaggle or auto-generated benchmark)
├── models/
│   └── benchmark_results.csv  # Standardized comparative metrics leaderboard
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # High-fidelity synthetic transaction engine with realistic fraud patterns
│   ├── feature_engineering.py # Temporal sequences, velocity bursts, and mule account analytics
│   ├── models.py              # Supervised, Unsupervised, and Hybrid Anomaly Detection pipelines
│   ├── threshold_tuner.py     # Dynamic decision threshold calibration & financial loss optimization
│   ├── explainability.py      # XAI engine: global importances, local attribution waterfalls, investigator memos
│   ├── evaluation.py          # Standardized evaluation metrics (PR-AUC, ROC-AUC, F1, Latency)
│   └── stream_simulator.py    # Real-time streaming simulation & two-tier cost-aware latency profiler
├── app.py                     # Full-featured 8-tab interactive Streamlit dashboard
├── benchmark_models.py        # CLI benchmarking script generating comparative leaderboard
├── check_dataset.py           # Dataset sanity verification script
├── data_analysis.py           # Exploratory data profiling & distribution analysis
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive documentation
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

```powershell
# Clone or open terminal in the project directory
python -m pip install -r requirements.txt
```

### 2. Verify Dataset

Run `check_dataset.py` to inspect the dataset. If `data/transactions.gz` is not yet placed, the system will automatically synthesize a realistic benchmark transactions dataset:

```powershell
python check_dataset.py
```

To run detailed statistical data analysis:
```powershell
python data_analysis.py
```

### 3. Run Automated Model Benchmark (CLI)

Run the automated evaluation suite comparing all 6 models across PR-AUC, ROC-AUC, F1, Latency, and Cost:

```powershell
python benchmark_models.py
```

### 4. Launch the Streamlit Interactive Dashboard

```powershell
python -m streamlit run app.py
# or run the helper script:
.\run_dashboard.ps1
```

The browser will open with the full 8-tab dashboard:
1. **Overview & Analytics**: High-level portfolio KPIs, class distributions, and spend histograms.
2. **Model Benchmarking**: Standardized leaderboard, interactive Precision-Recall (PR) curves, and ROC curves.
3. **Threshold Calibration**: Precision vs Recall vs F1 curves, Cost-Benefit loss minimization, and live comparison table.
4. **Temporal & Relational**: Inter-arrival time distributions, hourly fraud clustering, and mule risk scores.
5. **Explainable AI (XAI)**: Global feature importance and single-transaction local attribution waterfall charts with compliance memos.
6. **Stream & Cost-Aware Profiler**: Simulated streaming feed with two-tier cost-aware throughput benchmarks.
7. **Fraud Alerts Queue**: Dynamic filtered queue of flagged transactions with 1-click CSV download.
8. **Literature Synthesis & Gaps**: Searchable matrix of the 20 representative Scopus papers from Table 1.

---

## 📊 Benchmark Findings Summary

Empirical results from testing on 30,000 transactions (2.32% fraud rate):

| Model | Paradigm | PR-AUC (Primary) | ROC-AUC | F1-Score | Recall | Precision | FPR | Latency (ms/txn) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Fusion (IsoForest + Boosting)** | **Hybrid** | **0.9764** | **0.9989** | **0.9379** | **0.9540** | **0.9222** | **0.0019** | **0.0536 ms** |
| HistGradientBoosting | Supervised | 0.9683 | 0.9976 | 0.9300 | 0.9540 | 0.9071 | 0.0023 | 0.0159 ms |
| Random Forest | Supervised | 0.9478 | 0.9972 | 0.9235 | 0.9023 | 0.9458 | 0.0012 | 0.0107 ms |
| Logistic Regression | Supervised | 0.9186 | 0.9898 | 0.5466 | 0.9598 | 0.3822 | 0.0369 | 0.0022 ms |
| Isolation Forest | Unsupervised | 0.6964 | 0.9563 | 0.5000 | 0.7471 | 0.3757 | 0.0295 | 0.0189 ms |
| Local Outlier Factor | Unsupervised | 0.0628 | 0.5925 | 0.1336 | 0.1897 | 0.1031 | 0.0392 | 0.1215 ms |

### Key Insights:
1. **Hybrid Superiority**: Hybrid Fusion combining unsupervised Isolation Forest anomaly scoring with gradient tree boosting achieves the top PR-AUC (0.9764), verifying literature findings (Mohanakrishnan 2026; Abbassi et al. 2024).
2. **Threshold Optimization**: Dynamically calibrating the decision threshold drops net expected operational financial loss from **\$4,225.00** (under naive $t=0.5$) down to **\$2,495.00**—saving **\$1,730.00** per 7,500 transactions.
3. **Operational Efficiency**: Two-tier cost-aware screening handles over 85% of transactions with sub-millisecond fast pass, delivering multi-fold throughput increases.
