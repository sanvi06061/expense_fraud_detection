"""
data_generator.py
Generates a realistic benchmark financial transactions dataset matching the
Capital One / financial fraud schema, embedding temporal sequences, velocity bursts,
relational mule account behavior, and realistic fraud anomalies.
"""

import gzip
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_transactions(
    n_samples: int = 50000,
    n_accounts: int = 1500,
    fraud_ratio: float = 0.022,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic credit card / expense transactions with realistic
    temporal, relational, and fraud attack distributions.
    """
    np.random.seed(random_state)

    # 1. Accounts profile
    account_ids = [100000000 + i for i in range(n_accounts)]
    credit_limits = np.random.choice(
        [1000.0, 2500.0, 5000.0, 7500.0, 10000.0, 15000.0, 20000.0],
        size=n_accounts,
        p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.07, 0.03]
    )
    mule_accounts = set(np.random.choice(account_ids, size=int(n_accounts * 0.03), replace=False))
    
    account_dict = {
        acc: {
            "creditLimit": limit,
            "currentBalance": np.random.uniform(0, limit * 0.35),
            "isMule": acc in mule_accounts,
            "avgSpend": np.random.uniform(25, 180)
        }
        for acc, limit in zip(account_ids, credit_limits)
    }

    # 2. Merchants profile
    merchants = [
        {"name": "uber_eats", "mcc": "fastfood", "avg": 24.0, "std": 12.0},
        {"name": "amazon_prime", "mcc": "online_retail", "avg": 65.0, "std": 45.0},
        {"name": "target_store", "mcc": "department_store", "avg": 85.0, "std": 50.0},
        {"name": "shell_gas", "mcc": "fuel", "avg": 42.0, "std": 15.0},
        {"name": "walmart_super", "mcc": "grocery", "avg": 110.0, "std": 60.0},
        {"name": "apple_store", "mcc": "electronics", "avg": 450.0, "std": 350.0},
        {"name": "delta_air", "mcc": "airline", "avg": 380.0, "std": 200.0},
        {"name": "luxury_jewel", "mcc": "luxury", "avg": 850.0, "std": 600.0},
        {"name": "crypto_exchange", "mcc": "crypto_finance", "avg": 700.0, "std": 500.0},
        {"name": "convenience_kiosk", "mcc": "convenience", "avg": 14.0, "std": 8.0},
    ]

    start_date = datetime(2025, 1, 1, 0, 0, 0)
    records = []

    # Target counts
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Generate Legitimate Transactions
    for _ in range(n_legit):
        acc = np.random.choice(account_ids)
        info = account_dict[acc]
        merchant = np.random.choice(merchants)
        
        # Temporal: higher density during day (8am - 10pm)
        hour = int(np.clip(np.random.normal(14, 4), 0, 23))
        day_offset = np.random.randint(0, 180)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Amount: log-normal around merchant average, respecting credit limit
        base_amt = float(np.abs(np.random.normal(merchant["avg"], merchant["std"])))
        amt = round(max(2.5, min(base_amt, info["creditLimit"] * 0.7)), 2)

        # Available money
        avail = max(0.0, round(info["creditLimit"] - info["currentBalance"], 2))
        card_present = np.random.choice([True, False], p=[0.72, 0.28])
        acq_country = np.random.choice(["US", "CA", "MX", "GB"], p=[0.92, 0.04, 0.02, 0.02])

        records.append({
            "accountNumber": acc,
            "customerId": f"CUST_{acc}",
            "creditLimit": info["creditLimit"],
            "availableMoney": avail,
            "currentBalance": round(info["currentBalance"], 2),
            "transactionDateTime": txn_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "transactionAmount": amt,
            "merchantName": merchant["name"],
            "acqCountry": acq_country,
            "merchantCountryCode": acq_country,
            "posEntryMode": np.random.choice(["05", "09", "81", "02"], p=[0.55, 0.25, 0.15, 0.05]),
            "posConditionCode": "01",
            "merchantCategoryCode": merchant["mcc"],
            "transactionType": "PURCHASE",
            "cardPresent": card_present,
            "cardCVV": "742",
            "enteredCVV": "742",
            "cardLast4Digits": str(acc)[-4:],
            "isFraud": 0
        })

    # Generate Fraudulent Transactions (Patterns from Literature Review)
    fraud_patterns = [
        "velocity_burst",      # Rapid succession of transactions
        "card_not_present_spike", # Large amount, card not present, odd hours
        "credit_exhaustion",   # Amount near or exceeding credit limit / available balance
        "mule_dispersion",     # High-value rapid transfers/purchases draining balance
        "unusual_luxury_crypto"# High-risk MCC at odd nocturnal times
    ]

    for _ in range(n_fraud):
        pattern = np.random.choice(fraud_patterns, p=[0.25, 0.25, 0.20, 0.15, 0.15])
        acc = np.random.choice(account_ids)
        info = account_dict[acc]
        day_offset = np.random.randint(0, 180)

        if pattern == "velocity_burst":
            # Rapid transactions within 1-5 minutes
            merchant = np.random.choice(merchants[:3])
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 50)
            second = np.random.randint(0, 50)
            txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
            amt = round(float(np.random.uniform(150, 450)), 2)
            card_present = False
            acq_country = np.random.choice(["US", "RU", "NG", "CN"], p=[0.5, 0.2, 0.15, 0.15])

        elif pattern == "card_not_present_spike":
            # Late-night nocturnal spike (1am - 4am), card not present
            merchant = np.random.choice(merchants[5:8]) # electronics, airline, luxury
            hour = np.random.choice([1, 2, 3, 4])
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            txn_time = start_date + timedelta(days=day_offset, hours=int(hour), minutes=minute, seconds=second)
            amt = round(float(np.random.uniform(500, min(2200, info["creditLimit"] * 0.95))), 2)
            card_present = False
            acq_country = np.random.choice(["US", "GB", "NL", "SG"], p=[0.4, 0.3, 0.15, 0.15])

        elif pattern == "credit_exhaustion":
            # High amount that exceeds 80% of limit or drains available money
            merchant = np.random.choice(merchants[5:])
            hour = np.random.randint(0, 24)
            txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=np.random.randint(0, 60), seconds=np.random.randint(0, 60))
            amt = round(float(info["creditLimit"] * np.random.uniform(0.75, 1.05)), 2)
            card_present = np.random.choice([True, False], p=[0.2, 0.8])
            acq_country = np.random.choice(["US", "MX", "CA"], p=[0.7, 0.15, 0.15])

        elif pattern == "mule_dispersion":
            # Mule account laundering signature (Literature Table 1 Paper 2 Mule Trace)
            mule_acc = list(mule_accounts)[np.random.randint(0, len(mule_accounts))] if mule_accounts else acc
            acc = mule_acc
            info = account_dict[acc]
            merchant = np.random.choice([m for m in merchants if m["mcc"] in ["crypto_finance", "luxury", "online_retail"]])
            hour = np.random.randint(0, 24)
            txn_time = start_date + timedelta(days=day_offset, hours=hour, minutes=np.random.randint(0, 60), seconds=np.random.randint(0, 60))
            amt = round(float(np.random.uniform(600, 1500)), 2)
            card_present = False
            acq_country = np.random.choice(["US", "CY", "MT", "KY"], p=[0.4, 0.2, 0.2, 0.2])

        else: # unusual_luxury_crypto
            merchant = np.random.choice([m for m in merchants if m["mcc"] in ["crypto_finance", "luxury"]])
            hour = np.random.choice([0, 1, 2, 3, 23])
            txn_time = start_date + timedelta(days=day_offset, hours=int(hour), minutes=np.random.randint(0, 60), seconds=np.random.randint(0, 60))
            amt = round(float(np.random.uniform(400, 1200)), 2)
            card_present = False
            acq_country = np.random.choice(["US", "DE", "FR", "RU"], p=[0.4, 0.2, 0.2, 0.2])

        avail = max(0.0, round(info["creditLimit"] - info["currentBalance"], 2))
        entered_cvv = "742" if np.random.rand() > 0.3 else "999" # 30% CVV mismatch on fraud

        records.append({
            "accountNumber": acc,
            "customerId": f"CUST_{acc}",
            "creditLimit": info["creditLimit"],
            "availableMoney": avail,
            "currentBalance": round(info["currentBalance"], 2),
            "transactionDateTime": txn_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "transactionAmount": amt,
            "merchantName": merchant["name"],
            "acqCountry": acq_country,
            "merchantCountryCode": acq_country,
            "posEntryMode": np.random.choice(["81", "02", "90"], p=[0.7, 0.2, 0.1]),
            "posConditionCode": "08",
            "merchantCategoryCode": merchant["mcc"],
            "transactionType": "PURCHASE",
            "cardPresent": card_present,
            "cardCVV": "742",
            "enteredCVV": entered_cvv,
            "cardLast4Digits": str(acc)[-4:],
            "isFraud": 1
        })

    df = pd.DataFrame(records)
    # Sort by datetime to ensure temporal ordering
    df["dt_temp"] = pd.to_datetime(df["transactionDateTime"])
    df = df.sort_values("dt_temp").drop(columns=["dt_temp"]).reset_index(drop=True)
    return df


def ensure_dataset(data_path: str = "data/transactions.gz", sample_size: int = 50000) -> Path:
    """
    Ensures dataset exists at data_path.
    If not, generates high-fidelity synthetic benchmark dataset and writes compressed gzip.
    """
    p = Path(data_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Dataset not found at {data_path}. Synthesizing realistic benchmark transactions ({sample_size} records)...")
        df = generate_synthetic_transactions(n_samples=sample_size, random_state=42)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
        print(f"Synthetic benchmark dataset saved to {p} ({p.stat().st_size / 1024:.1f} KB).")
    return p


if __name__ == "__main__":
    path = ensure_dataset()
    test_df = pd.read_csv(path, compression="gzip", nrows=5)
    print("Dataset verification sample:")
    print(test_df[["transactionDateTime", "accountNumber", "transactionAmount", "isFraud"]])
