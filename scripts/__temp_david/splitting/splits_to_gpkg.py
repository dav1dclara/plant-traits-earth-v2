from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

parquet_dir = Path("/scratch3/plant-traits-v2/data/22km/skcv_splits")
gpkg_dir = Path("/scratch3/plant-traits-v2/data/temp/splits")
gpkg_dir.mkdir(parents=True, exist_ok=True)
parquet_files = sorted(parquet_dir.glob("*.parquet"))

for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(x, y) for x, y in zip(df["x"], df["y"])], crs="EPSG:6933"
    )
    out_gpkg = gpkg_dir / f"{parquet_file.stem}.gpkg"
    # gdf.to_file(out_gpkg, driver="GPKG")
    # print(f"Saved {out_gpkg}")

# --- Rasterize first file ---
REF_TIF = Path(
    "/scratch3/plant-traits-v2/data/22km/eo_data/canopy_height/ETH_GlobalCanopyHeight_2020_v1.tif"
)

df = pd.read_parquet(parquet_files[0])
value_col = "fold"

with rasterio.open(REF_TIF) as ref:
    profile = ref.profile
    transform = ref.transform
    n_rows, n_cols = ref.height, ref.width

profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")

raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
col_idx, row_idx = ~transform * (df["x"].values, df["y"].values)
col_idx = col_idx.astype(int)
row_idx = row_idx.astype(int)

# Clip to valid bounds (drop any points outside the reference grid)
valid = (row_idx >= 0) & (row_idx < n_rows) & (col_idx >= 0) & (col_idx < n_cols)
raster[row_idx[valid], col_idx[valid]] = df[value_col].values[valid]

out_tif = gpkg_dir / f"{parquet_files[0].stem}.tif"
with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(raster, 1)

print(f"Saved {out_tif}")
