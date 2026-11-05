import pandas as pd
from pathlib import Path

path = Path("data/transactions.gz")
if not path.exists():
    raise FileNotFoundError("Place transactions.gz inside the data folder.")
df = pd.read_csv(path, compression="gzip", nrows=10)
print("Dataset loaded successfully!")
print("First 10 rows:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
