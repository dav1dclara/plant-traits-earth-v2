"""
Inspect /scratch3/plant-traits-v2/data/1km/raw/eo_data/eo_predict_imputed.parquet
without loading the full file into memory.
"""

import pandas as pd
import pyarrow.parquet as pq

PATH = "/scratch3/plant-traits-v2/data/1km/raw/eo_data/eo_predict_imputed.parquet"

pf = pq.ParquetFile(PATH)
meta = pf.metadata

print("=" * 60)
print("FILE METADATA")
print("=" * 60)
print(f"Rows:        {meta.num_rows:,}")
print(f"Columns:     {meta.num_columns}")
print(f"Row groups:  {meta.num_row_groups}")

print("\n" + "=" * 60)
print("SCHEMA")
print("=" * 60)
print(pf.schema_arrow)

# Sample first batch only — avoids loading 128M rows
batch = next(pf.iter_batches(batch_size=5))
df = batch.to_pandas()

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.to_string())

print("\n" + "=" * 60)
print("COLUMN STATS (first batch, 5 rows)")
print("=" * 60)
print(df.describe(include="all"))

print("\n" + "=" * 60)
print("NULL COUNTS (first batch)")
print("=" * 60)
print(df.isnull().sum())
