import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

DATA_PATH = Path("data/transactions.gz")
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

def make_features(df):
    x = pd.DataFrame(index=df.index)
    numeric = [
        "transactionAmount", "creditLimit", "availableMoney",
        "currentBalance", "posEntryMode", "posConditionCode"
    ]
    for col in numeric:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["cardPresent", "expirationDateKeyInMatch"]:
        if col in df.columns:
            x[col] = df[col].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            )

    if {"transactionAmount", "creditLimit"}.issubset(x.columns):
        x["amount_to_limit"] = x["transactionAmount"] / x["creditLimit"].replace(0, np.nan)
    if {"currentBalance", "creditLimit"}.issubset(x.columns):
        x["balance_to_limit"] = x["currentBalance"] / x["creditLimit"].replace(0, np.nan)
    if {"transactionAmount", "availableMoney"}.issubset(x.columns):
        x["amount_to_available"] = x["transactionAmount"] / x["availableMoney"].replace(0, np.nan)

    if "transactionDateTime" in df.columns:
        dt = pd.to_datetime(df["transactionDateTime"], errors="coerce")
        x["hour"] = dt.dt.hour
        x["dayofweek"] = dt.dt.dayofweek
        x["month"] = dt.dt.month
        x["is_night"] = ((dt.dt.hour < 5) | (dt.dt.hour >= 23)).astype(float)

    # Consistent category encoding learned from the sampled dataset.
    for col in ["merchantCategoryCode", "transactionType", "acqCountry", "merchantCountryCode"]:
        if col in df.columns:
            x[col + "_code"] = pd.factorize(df[col].fillna("Unknown").astype(str))[0]

    return x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

def choose_f2_threshold(y_true, probabilities):
    best = None
    # Validation only. The final test set is never used for threshold selection.
    for threshold in np.arange(0.02, 0.801, 0.01):
        pred = (probabilities >= threshold).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        row = (f2, threshold, p, r)
        if best is None or row[0] > best[0]:
            best = row
    return best[1], best[2], best[3], best[0]

def evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_p, val_r, val_f2 = choose_f2_threshold(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    return {
        "Model": name,
        "Threshold (validation F2)": round(float(threshold), 3),
        "Test accuracy": round(accuracy_score(y_test, test_pred), 4),
        "Fraud precision": round(precision_score(y_test, test_pred, zero_division=0), 4),
        "Fraud recall": round(recall_score(y_test, test_pred, zero_division=0), 4),
        "Fraud F1": round(f1_score(y_test, test_pred, zero_division=0), 4),
        "Fraud F2": round(fbeta_score(y_test, test_pred, beta=2, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, test_prob), 4),
        "TP": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[1, 1]),
        "FP": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[0, 1]),
        "FN": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[1, 0]),
        "TN": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[0, 0]),
        "Validation precision": round(float(val_p), 4),
        "Validation recall": round(float(val_r), 4),
        "Validation F2": round(float(val_f2), 4),
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)
    if "isFraud" not in df:
        raise ValueError("Dataset must contain the isFraud column.")

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).copy()
    y = sample["isFraud"].astype(int)
    X = make_features(sample)

    # Split first. Resampling occurs only inside the SMOTE training pipeline.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Sample: {len(sample):,} rows | Features: {X.shape[1]}")
    print(f"Train fraud: {int(y_train.sum()):,} | Validation fraud: {int(y_val.sum()):,} | Test fraud: {int(y_test.sum()):,}")
    print("\nTraining and comparing models. This can take several minutes...")

    models = [
        (
            "Random Forest (class-weighted)",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=120, max_depth=14, min_samples_leaf=2,
                    class_weight="balanced_subsample", random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ])
        ),
        (
            "SMOTE + Random Forest",
            ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(sampling_strategy=0.25, random_state=RANDOM_STATE, k_neighbors=5)),
                ("model", RandomForestClassifier(
                    n_estimators=120, max_depth=14, min_samples_leaf=2,
                    class_weight=None, random_state=RANDOM_STATE, n_jobs=-1
                ))
            ])
        ),
        (
            "Class-weighted Logistic Regression",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced", max_iter=1000,
                    solver="liblinear", random_state=RANDOM_STATE
                ))
            ])
        )
    ]

    results = []
    for name, model in models:
        print(f"\nTraining: {name}")
        result = evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(result)
        print(
            f"  Precision={result['Fraud precision']:.4f} | "
            f"Recall={result['Fraud recall']:.4f} | "
            f"F2={result['Fraud F2']:.4f} | "
            f"ROC-AUC={result['ROC-AUC']:.4f} | "
            f"Threshold={result['Threshold (validation F2)']:.2f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        ["Fraud F2", "Fraud recall"], ascending=False
    )
    output = Path("model_comparison_results.csv")
    results_df.to_csv(output, index=False)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON (test metrics; thresholds selected on validation)")
    print("=" * 80)
    print(results_df[[
        "Model", "Threshold (validation F2)", "Test accuracy",
        "Fraud precision", "Fraud recall", "Fraud F1", "Fraud F2", "ROC-AUC",
        "TP", "FP", "FN", "TN"
    ]].to_string(index=False))
    print(f"\nSaved results to: {output.resolve()}")
    print("\nChoose based on fraud recall/F2 and an acceptable false-positive count, not accuracy alone.")

if __name__ == "__main__":
    main()
