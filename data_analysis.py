import pandas as pd

df = pd.read_csv("data/transactions.gz", compression="gzip", low_memory=False)
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head(15))
print("\nFraud distribution:\n", df["isFraud"].value_counts(dropna=False))
print("\nFraud percentage:\n", (df["isFraud"].value_counts(normalize=True) * 100).round(3))
print("\nTransaction amount summary:\n", df["transactionAmount"].describe())
print("\nSample records:\n", df.head())
