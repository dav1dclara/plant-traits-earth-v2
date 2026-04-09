"""
Inspect the partitioned parquet dataset at /scratch3/plant-traits-v2/data/1km/raw/Y.parquet
"""

import pandas as pd

PATH = "/scratch3/plant-traits-v2/data/1km/raw/Y.parquet"

df = pd.read_parquet(PATH)

print(f"Shape:   {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nDescribe:\n{df.describe(include='all')}")
print(f"\nFirst 5 rows:\n{df.head()}")
