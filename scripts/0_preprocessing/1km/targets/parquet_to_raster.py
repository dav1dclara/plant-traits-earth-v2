"""
Convert all trait columns from the partitioned Y.parquet to GeoTIFF rasters.

Each output raster has two bands:
  - Band 1 ("mean"):   trait mean value
  - Band 2 ("source"): observation source (1=GBIF, 2=sPlot, NaN=nodata)

CRS: EPSG:6933, pixel size derived from the data.
Output filenames use the trait ID without the "_mean" suffix.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

PARQUET_PATH = Path("/scratch3/plant-traits-v2/data/1km/raw/Y.parquet")
OUT_DIR = PARQUET_PATH.parents[1] / "processed" / "targets"

print(f"Loading {PARQUET_PATH} ...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  {len(df):,} rows")

# Derive pixel size from median spacing of unique coordinates
xs = np.sort(df["x"].unique())
ys = np.sort(df["y"].unique())
dx = float(np.median(np.diff(xs)))
dy = float(np.median(np.diff(ys)))
print(f"  dx={dx:.4f} m  dy={dy:.4f} m")

x_min, x_max = xs[0], xs[-1]
y_min, y_max = ys[0], ys[-1]

n_cols = round((x_max - x_min) / dx) + 1
n_rows = round((y_max - y_min) / dy) + 1
print(f"  Grid: {n_rows} rows x {n_cols} cols")

# Map each point to its grid index
col_idx = np.round((df["x"].values - x_min) / dx).astype(int)
row_idx = np.round((y_max - df["y"].values) / dy).astype(int)

# Affine transform: top-left corner is (x_min - dx/2, y_max + dy/2)
transform = from_origin(x_min - dx / 2, y_max + dy / 2, dx, dy)


def write_tif(
    path, bands, band_names, crs, transform, dtype="float32", compress=None, tiled=False
):
    """Write one or more 2D arrays as bands to a GeoTIFF."""
    if isinstance(bands, np.ndarray) and bands.ndim == 2:
        bands = [bands]
    opts = dict(
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=len(bands),
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    )
    if compress:
        opts["compress"] = compress
    if tiled:
        opts["tiled"] = True
        opts["blockxsize"] = 256
        opts["blockysize"] = 256
    with rasterio.open(path, "w", **opts) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.astype(dtype), i)
            dst.update_tags(i, name=band_names[i - 1])


# Build source band (1=g, 2=s, NaN=nodata)
source_map = {"g": 1.0, "s": 2.0}
source_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
source_arr[row_idx, col_idx] = df["source"].map(source_map).values

trait_cols = [c for c in df.columns if c not in ("x", "y", "source")]
print(f"  {len(trait_cols)} traits to export\n")

crs = CRS.from_epsg(6933)
OUT_DIR.mkdir(parents=True, exist_ok=True)

for trait in trait_cols:
    trait_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    trait_arr[row_idx, col_idx] = df[trait].values

    trait_id = trait.removesuffix("_mean")
    out_path = OUT_DIR / f"{trait_id}.tif"
    write_tif(
        out_path,
        [trait_arr, source_arr],
        ["mean", "source"],
        crs,
        transform,
        dtype="float32",
        compress="zstd",
        tiled=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  {trait:<20} {size_mb:.2f} MB  ->  {out_path}")
