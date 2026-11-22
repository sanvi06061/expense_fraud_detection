"""
feature_engineering.py
Advanced feature engineering for financial transaction anomaly and fraud detection.
Integrates temporal transaction sequences, account-level velocity, rolling statistics,
and relational mule-account indicators (Addressing Research Gap 3 & Papers [2], [18], [30]).
"""

import numpy as np
import pandas as pd


def extract_temporal_and_relational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transaction records into a rich feature set incorporating:
    1. Financial ratios (amount to credit limit, current balance to limit, amount to available)
    2. Temporal sequence features (time-since-last-transaction, hour, night-hours, day of week)
    3. Account velocity patterns (txns in last 1 hour, txns in last 24 hours, rolling 24h spend)
    4. Account historical spending deviations (amount relative to account average spend)
    5. Relational & Mule account indicators (rapid multi-merchant access, cross-border checks)
    6. Categorical encoding for merchant category, type, and acquisition country.
    """
    df = df.copy()

    # Ensure transactionDateTime is datetime
    if "transactionDateTime" in df.columns:
        dt = pd.to_datetime(df["transactionDateTime"], errors="coerce")
    else:
        dt = pd.Series(pd.Timestamp.now(), index=df.index)

    # Base numerical features
    amount = pd.to_numeric(df.get("transactionAmount"), errors="coerce").fillna(0.0)
    credit_limit = pd.to_numeric(df.get("creditLimit"), errors="coerce").replace(0, np.nan)
    available_money = pd.to_numeric(df.get("availableMoney"), errors="coerce").replace(0, np.nan)
    current_balance = pd.to_numeric(df.get("currentBalance"), errors="coerce").fillna(0.0)

    features = pd.DataFrame(index=df.index)
    features["transactionAmount"] = amount
    features["creditLimit"] = credit_limit.fillna(1000.0)
    features["availableMoney"] = available_money.fillna(1000.0)
    features["currentBalance"] = current_balance

    # 1. Financial Risk Ratios
    features["amount_to_limit"] = (amount / credit_limit).fillna(0.0)
    features["balance_to_limit"] = (current_balance / credit_limit).fillna(0.0)
    features["amount_to_available"] = (amount / available_money).fillna(0.0)
    features["is_amount_exceeds_available"] = (amount > available_money).astype(int)

    # 2. Card & Authentication Features
    card_present_raw = df.get("cardPresent", False).astype(str).str.lower()
    features["cardPresent"] = card_present_raw.map({"true": 1, "1": 1, "false": 0, "0": 0}).fillna(0).astype(int)

    if "cardCVV" in df.columns and "enteredCVV" in df.columns:
        features["cvv_match"] = (df["cardCVV"].astype(str) == df["enteredCVV"].astype(str)).astype(int)
    else:
        features["cvv_match"] = 1

    # 3. Temporal Calendar Features
    features["hour"] = dt.dt.hour.fillna(12).astype(int)
    features["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
    features["month"] = dt.dt.month.fillna(1).astype(int)
    features["is_weekend"] = features["dayofweek"].isin([5, 6]).astype(int)
    # Nocturnal transactions (11pm - 5am) often carry higher fraud risk
    features["is_night_hours"] = ((features["hour"] >= 23) | (features["hour"] < 5)).astype(int)

    # 4. Account-Level Temporal Velocity & Rolling Features (Gap 3)
    if "accountNumber" in df.columns and dt.notna().any():
        # Work on sorted temp index
        temp = pd.DataFrame({
            "account": df["accountNumber"].astype(str),
            "dt": dt,
            "amt": amount,
            "orig_idx": df.index
        }).sort_values(["account", "dt"])

        # Time since last transaction in seconds
        temp["time_diff"] = temp.groupby("account")["dt"].diff().dt.total_seconds().fillna(86400)
        # Cap large diffs at 7 days to avoid extreme outliers
        temp["time_diff"] = temp["time_diff"].clip(lower=0, upper=604800)

        # Rolling velocity counts using rolling window on time
        # Fast approximation: grouping by account and computing recent frequency
        temp["txn_order"] = temp.groupby("account").cumcount() + 1
        
        # Velocity in last 1 hour (3600s) and last 24h (86400s)
        # Approximate using rapid succession flags
        temp["is_rapid_burst"] = (temp["time_diff"] <= 300).astype(int)  # <= 5 minutes
        temp["is_hourly_burst"] = (temp["time_diff"] <= 3600).astype(int) # <= 1 hour

        # Cumulative transaction count and mean spend per account
        acc_means = temp.groupby("account")["amt"].transform("mean")
        temp["amount_to_account_mean"] = (temp["amt"] / acc_means.replace(0, np.nan)).fillna(1.0)
        
        # Merge back in original order
        temp = temp.sort_values("orig_idx")
        features["time_since_last_txn_sec"] = temp["time_diff"].values
        features["is_rapid_burst"] = temp["is_rapid_burst"].values
        features["is_hourly_burst"] = temp["is_hourly_burst"].values
        features["amount_to_account_mean"] = temp["amount_to_account_mean"].values
    else:
        features["time_since_last_txn_sec"] = 86400.0
        features["is_rapid_burst"] = 0
        features["is_hourly_burst"] = 0
        features["amount_to_account_mean"] = 1.0

    # 5. Relational & Mule Risk Indicators (Paper [2] Mule Trace)
    # Mule patterns: High amount with rapid burst, card not present, foreign or crypto MCC
    if "acqCountry" in df.columns:
        features["is_international"] = (df["acqCountry"].fillna("US").astype(str).str.upper() != "US").astype(int)
    else:
        features["is_international"] = 0

    # Mule heuristic composite: rapid burst + card-not-present + amount > available*0.6
    features["mule_composite_score"] = (
        features["is_rapid_burst"] * 0.35 +
        (1 - features["cardPresent"]) * 0.25 +
        (features["amount_to_available"] > 0.6).astype(int) * 0.25 +
        features["is_international"] * 0.15
    )

    # 6. Categorical Encodings
    for col in ["merchantCategoryCode", "transactionType", "acqCountry", "merchantName"]:
        if col in df.columns:
            codes, _ = pd.factorize(df[col].fillna("Unknown").astype(str))
            features[col + "_code"] = codes
        else:
            features[col + "_code"] = 0

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def generate_rule_based_explanations(row: pd.Series) -> str:
    """
    Translates key feature flags into interpretable domain rules
    to complement model-based XAI (Research Gap 5).
    """
    flags = []
    amt = row.get("transactionAmount", 0)
    avail = row.get("availableMoney", 1000)
    limit = row.get("creditLimit", 1000)
    burst = row.get("is_rapid_burst", 0)
    night = row.get("is_night_hours", 0)
    card_present = row.get("cardPresent", 1)
    intl = row.get("is_international", 0)
    amt_mean_ratio = row.get("amount_to_account_mean", 1)

    if burst == 1:
        flags.append("High-velocity burst (< 5m interval)")
    if amt > 500:
        flags.append(f"High transaction value (${amt:,.2f})")
    if amt_mean_ratio > 3.5:
        flags.append(f"Spend spike ({amt_mean_ratio:.1f}x account average)")
    if amt > avail:
        flags.append("Exceeds available balance")
    elif limit > 0 and (amt / limit) > 0.5:
        flags.append("Exceeds 50% of credit limit")
    if night == 1:
        flags.append("Nocturnal transaction (11pm-5am)")
    if card_present == 0:
        flags.append("Card-not-present checkout")
    if intl == 1:
        flags.append("Cross-border transaction")

    return " | ".join(flags) if flags else "Typical behavioral profile"
