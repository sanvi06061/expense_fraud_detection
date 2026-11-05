import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, fbeta_score, accuracy_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline

st.set_page_config(page_title="Expense Fraud Detection", page_icon="🛡️", layout="wide")
DATA_PATH = Path("data/transactions.gz")
RANDOM_STATE = 42
SAMPLE_SIZE = 100_000

@st.cache_data(show_spinner="Loading transaction dataset...")
def load_data():
    if not DATA_PATH.exists():
        st.error("Dataset not found. Put transactions.gz inside the data folder.")
        st.stop()
    return pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)

def prepare_features(df, category_maps=None):
    x = pd.DataFrame(index=df.index)
    numeric_cols = [
        "transactionAmount", "creditLimit", "availableMoney", "currentBalance",
        "posEntryMode", "posConditionCode"
    ]
    for col in numeric_cols:
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

    for col in ["merchantCategoryCode", "transactionType", "acqCountry", "merchantCountryCode"]:
        if col in df.columns:
            values = df[col].fillna("Unknown").astype(str)
            if category_maps is not None and col in category_maps:
                x[col + "_code"] = values.map(category_maps[col]).fillna(-1)
            else:
                x[col + "_code"] = pd.factorize(values)[0]

    return x.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

def threshold_by_f2(y_true, probabilities):
    best_threshold, best_f2 = 0.27, -1
    for threshold in np.arange(0.02, 0.801, 0.01):
        pred = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if score > best_f2:
            best_threshold, best_f2 = float(threshold), float(score)
    return best_threshold

def risk_reason(row):
    reasons = []
    amount = pd.to_numeric(row.get("transactionAmount"), errors="coerce")
    limit = pd.to_numeric(row.get("creditLimit"), errors="coerce")
    available = pd.to_numeric(row.get("availableMoney"), errors="coerce")
    if pd.notna(amount) and amount > 500:
        reasons.append("High transaction amount")
    if pd.notna(amount) and pd.notna(limit) and limit > 0 and amount / limit > 0.5:
        reasons.append("Transaction exceeds 50% of credit limit")
    if pd.notna(amount) and pd.notna(available) and amount > available:
        reasons.append("Amount exceeds available money")
    dt = pd.to_datetime(row.get("transactionDateTime"), errors="coerce")
    if pd.notna(dt) and (dt.hour < 5 or dt.hour >= 23):
        reasons.append("Unusual transaction time")
    if str(row.get("cardPresent", "")).lower() in ("false", "0"):
        reasons.append("Card-not-present transaction")
    return "; ".join(reasons) if reasons else "Flagged by model; review transaction details"

@st.cache_resource(show_spinner="Training class-weighted Random Forest and anomaly detector...")
def train_models(_df):
    sample = _df.sample(n=min(SAMPLE_SIZE, len(_df)), random_state=RANDOM_STATE).copy()
    y = sample["isFraud"].astype(int)

    # Build one stable category mapping from the complete dataset and reuse it
    # for training, dashboard scoring and SHAP explanations.
    category_columns = [
        "merchantCategoryCode", "transactionType", "acqCountry", "merchantCountryCode"
    ]
    category_maps = {}
    for col in category_columns:
        if col in _df.columns:
            categories = sorted(_df[col].fillna("Unknown").astype(str).unique())
            category_maps[col] = {value: code for code, value in enumerate(categories)}

    X = prepare_features(sample, category_maps)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Imputation and model training are fitted on training data only.
    rf = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)

    # Select threshold on validation data, not on the final test set.
    val_prob = rf.predict_proba(X_val)[:, 1]
    threshold = threshold_by_f2(y_val, val_prob)

    test_prob = rf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    report = classification_report(y_test, test_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "f2": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
        "threshold": threshold,
        "report": report,
        "confusion": cm,
        "test_size": int(len(y_test)),
        "test_fraud": int(y_test.sum())
    }

    iso = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=100, contamination=0.01, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    iso.fit(X)
    return rf, iso, list(X.columns), metrics, category_maps

st.title("🛡️ Expense Anomaly & Fraud Detection System")
st.caption("Class-weighted Random Forest, anomaly detection, risk scoring and transaction analytics")

df = load_data()
if "isFraud" not in df.columns:
    st.error("Dataset must contain an 'isFraud' column.")
    st.stop()

rf, iso, feature_cols, metrics, category_maps = train_models(df)

# Score a manageable, reproducible sample for interactive dashboard display.
display_df = df.sample(n=min(30000, len(df)), random_state=7).copy()
X_display = prepare_features(display_df, category_maps).reindex(columns=feature_cols)
probabilities = rf.predict_proba(X_display)[:, 1]
display_df["fraud_probability"] = probabilities
display_df["risk_score"] = (probabilities * 100).round(1)
display_df["model_prediction"] = (probabilities >= metrics["threshold"]).astype(int)
display_df["anomaly_flag"] = (iso.predict(X_display) == -1)
display_df["risk_level"] = np.select(
    [
        probabilities >= metrics["threshold"],
        probabilities >= metrics["threshold"] * 0.60
    ],
    ["High", "Medium"],
    default="Low"
)
display_df["reason"] = display_df.apply(risk_reason, axis=1)

st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk level", ["Low", "Medium", "High"], default=["Medium", "High"]
)
flagged_only = st.sidebar.checkbox("Show model-flagged or anomalous transactions only")
amounts = pd.to_numeric(display_df["transactionAmount"], errors="coerce").fillna(0)
min_amount, max_amount = float(amounts.min()), float(amounts.max())
amount_range = st.sidebar.slider(
    "Transaction amount range",
    min_value=min_amount,
    max_value=max(max_amount, min_amount + 1.0),
    value=(min_amount, max_amount)
)

filtered = display_df[
    display_df["risk_level"].isin(risk_filter)
    & amounts.between(*amount_range)
]
if flagged_only:
    filtered = filtered[
        (filtered["model_prediction"] == 1) | filtered["anomaly_flag"]
    ]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions analysed", f"{len(df):,}")
c2.metric("Actual fraud cases", f"{int(df['isFraud'].sum()):,}")
c3.metric("Fraud rate", f"{df['isFraud'].mean() * 100:.2f}%")
c4.metric("High-risk in sample", f"{int((display_df['risk_level'] == 'High').sum()):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Fraud alerts", "Model performance", "Forecast & data", "Explainability (SHAP)"]
)

with tab1:
    left, right = st.columns(2)
    counts = df["isFraud"].value_counts().rename(
        index={False: "Legitimate", True: "Fraud"}
    ).reset_index()
    counts.columns = ["Transaction type", "Count"]
    with left:
        st.plotly_chart(
            px.pie(counts, names="Transaction type", values="Count",
                   title="Actual fraud distribution"),
            width="stretch"
        )
    with right:
        amount_data = df.sample(min(20000, len(df)), random_state=1).copy()
        amount_data["transactionAmount"] = pd.to_numeric(
            amount_data["transactionAmount"], errors="coerce"
        )
        st.plotly_chart(
            px.histogram(amount_data, x="transactionAmount", color="isFraud",
                         nbins=50, title="Transaction amount distribution"),
            width="stretch"
        )
    st.subheader("Predicted risk score distribution")
    st.plotly_chart(
        px.histogram(display_df, x="risk_score", color="risk_level", nbins=30),
        width="stretch"
    )

with tab2:
    st.subheader("Transactions requiring attention")
    alerts = filtered.sort_values(
        ["model_prediction", "risk_score", "anomaly_flag"],
        ascending=[False, False, False]
    )
    cols = [
        c for c in [
            "transactionDateTime", "accountNumber", "merchantName",
            "transactionAmount", "availableMoney", "fraud_probability",
            "risk_score", "risk_level", "model_prediction", "anomaly_flag",
            "reason", "isFraud"
        ] if c in alerts.columns
    ]
    st.dataframe(alerts[cols].head(500), width="stretch", height=450)
    st.download_button(
        "Download filtered alerts as CSV",
        alerts[cols].to_csv(index=False),
        "fraud_alerts.csv",
        "text/csv"
    )

with tab3:
    st.subheader("Class-weighted Random Forest performance")
    st.caption(
        "Threshold selected using validation data. Metrics below are from an untouched test set."
    )
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    b.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c.metric("Fraud precision", f"{metrics['precision'] * 100:.2f}%")
    d.metric("Fraud recall", f"{metrics['recall'] * 100:.2f}%")
    e, f = st.columns(2)
    e.metric("Fraud F1-score", f"{metrics['f1']:.4f}")
    f.metric("Fraud F2-score", f"{metrics['f2']:.4f}")
    st.metric("Validation-selected classification threshold", f"{metrics['threshold']:.2f}")
    st.write("Classification report")
    st.dataframe(pd.DataFrame(metrics["report"]).T, width="stretch")
    st.write("Confusion matrix (actual rows × predicted columns)")
    st.dataframe(
        pd.DataFrame(
            metrics["confusion"],
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"]
        ),
        width="stretch"
    )
    st.info(
        f"Model trained on a reproducible sample of up to {SAMPLE_SIZE:,} rows. "
        "Class weighting was used instead of SMOTE. The threshold was selected "
        "by maximizing validation F2-score; the test set was not used for tuning."
    )
    st.warning(
        "Fraud precision is low in this experiment, so many alerts may be false positives. "
        "Treat model alerts as review signals, not proof of fraud."
    )

with tab4:
    st.subheader("Historical transaction trend")
    dates = pd.to_datetime(df["transactionDateTime"], errors="coerce")
    trend = df.assign(month=dates.dt.to_period("M").astype(str))
    trend = trend[trend["month"] != "NaT"].groupby("month").agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum")
    ).reset_index()
    if not trend.empty:
        st.plotly_chart(
            px.line(trend, x="month", y=["transactions", "fraud_cases"],
                    title="Monthly transactions and fraud cases"),
            width="stretch"
        )
        st.caption("Historical trend only; this is not a validated future forecast.")
    st.subheader("Dataset preview")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df.head(20), width="stretch")


with tab5:
    st.subheader("Explainable AI — SHAP")
    st.caption(
        "SHAP estimates how each input feature contributes to the Random Forest's fraud probability. "
        "These are model explanations, not proof that a transaction is fraudulent."
    )

    # Explain a manageable background sample so the dashboard remains responsive.
    explain_sample = X_display.sample(
        n=min(250, len(X_display)), random_state=42
    )
    imputer = rf.named_steps["imputer"]
    classifier = rf.named_steps["classifier"]
    explain_matrix = imputer.transform(explain_sample)
    explainer = shap.TreeExplainer(classifier)

    try:
        shap_raw = explainer.shap_values(explain_matrix, check_additivity=False)
        if isinstance(shap_raw, list):
            global_shap_values = np.asarray(shap_raw[1])
        else:
            global_shap_values = np.asarray(shap_raw)
            if global_shap_values.ndim == 3:
                if global_shap_values.shape[-1] == 2:
                    global_shap_values = global_shap_values[:, :, 1]
                elif global_shap_values.shape[0] == 2:
                    global_shap_values = global_shap_values[1]

        if global_shap_values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP output shape: {global_shap_values.shape}"
            )

        # Global feature importance: mean absolute SHAP value.
        importance = pd.DataFrame({
            "Feature": feature_cols,
            "Mean absolute SHAP value": np.abs(global_shap_values).mean(axis=0)
        }).sort_values("Mean absolute SHAP value", ascending=False).head(15)

        st.markdown("**1. Global feature importance**")
        st.write(
            "Features at the top have the largest average influence on model predictions "
            "within this sample."
        )
        st.plotly_chart(
            px.bar(
                importance.sort_values("Mean absolute SHAP value"),
                x="Mean absolute SHAP value",
                y="Feature",
                orientation="h",
                title="Top 15 features influencing fraud predictions"
            ),
            width="stretch"
        )

        st.markdown("**2. Explain an individual transaction**")
        candidate_rows = display_df.index[
            (display_df["model_prediction"] == 1)
        ].tolist()
        if not candidate_rows:
            candidate_rows = display_df.nlargest(100, "fraud_probability").index.tolist()

        selected_row = st.selectbox(
            "Choose a model-flagged transaction",
            options=candidate_rows[:500],
            format_func=lambda idx: (
                f"Row {idx} | Amount: "
                f"{float(display_df.loc[idx, 'transactionAmount']):,.2f} | "
                f"Fraud probability: "
                f"{display_df.loc[idx, 'fraud_probability']:.1%}"
            )
        )

        one_row = X_display.loc[[selected_row]]
        one_transformed = imputer.transform(one_row)
        local_raw = explainer.shap_values(one_transformed, check_additivity=False)

        if isinstance(local_raw, list):
            local_values = np.asarray(local_raw[1])[0]
        else:
            local_array = np.asarray(local_raw)
            if local_array.ndim == 3:
                if local_array.shape[-1] == 2:
                    local_values = local_array[0, :, 1]
                elif local_array.shape[0] == 2:
                    local_values = local_array[1, 0, :]
                else:
                    raise ValueError(f"Unexpected local SHAP shape: {local_array.shape}")
            else:
                local_values = local_array[0]

        local_contrib = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP contribution": local_values,
            "Feature value": one_row.iloc[0].values
        })
        local_contrib["Absolute contribution"] = local_contrib["SHAP contribution"].abs()
        local_top = local_contrib.sort_values(
            "Absolute contribution", ascending=False
        ).head(12).sort_values("SHAP contribution")

        transaction = display_df.loc[selected_row]
        p1, p2, p3 = st.columns(3)
        p1.metric("Transaction amount", f"{float(transaction['transactionAmount']):,.2f}")
        p2.metric("Model fraud probability", f"{transaction['fraud_probability']:.1%}")
        p3.metric("Model decision", "Flagged" if transaction["model_prediction"] == 1 else "Not flagged")

        st.plotly_chart(
            go.Figure(go.Bar(
                x=local_top["SHAP contribution"],
                y=local_top["Feature"],
                orientation="h",
                marker_color=np.where(
                    local_top["SHAP contribution"] >= 0, "#E45756", "#4C78A8"
                ),
                customdata=local_top[["Feature value"]].values,
                hovertemplate=(
                    "Feature: %{y}<br>SHAP contribution: %{x:.5f}"
                    "<br>Encoded/processed feature value: %{customdata[0]}<extra></extra>"
                )
            )).update_layout(
                title="Features pushing this prediction higher or lower",
                xaxis_title="SHAP contribution to fraud model output",
                yaxis_title="Feature",
                height=500
            ),
            width="stretch"
        )
        st.caption(
            "Positive SHAP contributions push the model output toward the fraud class; "
            "negative contributions push it away. Magnitudes are in the model output's "
            "SHAP units, not currency or percentage points."
        )
        st.dataframe(
    local_top[
        ["Feature", "Feature value", "SHAP contribution"]
    ].sort_values(
        "SHAP contribution",
        key=lambda x: x.abs(),
        ascending=False
    ),
    width="stretch"
)
    except Exception as exc:
        st.error(f"SHAP explanation could not be generated: {exc}")
        st.info(
            "The classification dashboard is still available. If this error persists, "
            "send the complete error text so we can adjust the SHAP output handling."
        )

st.caption("Academic demonstration only. A model alert is a review signal, not proof of fraud.")
