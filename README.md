# Expense Anomaly & Fraud Detection System

## Features
- Dataset analysis and fraud distribution
- Random Forest supervised fraud probability
- Isolation Forest unsupervised anomaly flag
- 0-100 risk score and rule-based reason hints
- Streamlit dashboard, filters, alerts and CSV export
- Monthly historical trend

## Project structure
```text
expense-fraud-detection/
├── app.py
├── check_dataset.py
├── data_analysis.py
├── requirements.txt
├── README.md
├── data/
│   └── transactions.gz
└── models/
```

## Setup (Windows PowerShell)
Open the VS Code terminal in the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Make sure your Kaggle file is at `data/transactions.gz`. Then:
```powershell
python check_dataset.py
python data_analysis.py
streamlit run app.py
```

The browser opens the dashboard. The first run trains the models and may take a few minutes. It trains on a reproducible sample of up to 100,000 rows to keep laptop runtime manageable.

## Important notes
- `isFraud` is the known label used for supervised model training and evaluation.
- The Isolation Forest model identifies unusual patterns without using the fraud label.
- Risk score is the Random Forest predicted fraud probability multiplied by 100.
- The reason text is a transparent set of warning rules, not SHAP.
- This is an academic prototype. Validate on a separate time-based holdout and review precision/recall before any real-world use.
