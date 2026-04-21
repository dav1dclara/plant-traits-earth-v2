"""
Convert all trait columns from the partitioned Y.parquet to GeoTIFF rasters.

Each output raster has two bands:
  - Band 1 ("mean"):   trait mean value
  - Band 2 ("source"): observation source (1=GBIF, 2=sPlot, NaN=nodata)

Grid is snapped to the EO reference raster (canopy height) to guarantee
pixel-perfect alignment with all predictor layers.
Output filenames use the trait ID without the "_mean" suffix.

Usage:
    python targets_to_tiff.py               # skip already-written files
    python targets_to_tiff.py --overwrite   # reprocess all traits
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

PARQUET_PATH = Path("/scratch3/plant-traits-v2/data/1km/raw/Y.parquet")
REF_PATH = Path(
    "/scratch3/plant-traits-v2/data/1km/raw/eo_data/canopy_height/ETH_GlobalCanopyHeight_2020_v1.tif"
)
OUT_DIR = PARQUET_PATH.parents[1] / "targets" / "comb"

parser = argparse.ArgumentParser(description="Convert trait parquet to GeoTIFFs.")
parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite existing files (default: skip)."
)
args = parser.parse_args()

console = Console()

# ---------------------------------------------------------------------------

with rasterio.open(REF_PATH) as ref:
    transform = ref.transform
    crs = ref.crs
    n_rows, n_cols = ref.shape
console.print(f"Reference grid: {n_rows} rows x {n_cols} cols  CRS: {crs}")

console.print(f"Loading {PARQUET_PATH} ...")
df = pd.read_parquet(PARQUET_PATH)
console.print(f"  {len(df):,} rows")

# Parquet x/y are pixel centres; transform.c/f are pixel edges — subtract 0.5 to align.
col_idx = np.round((df["x"].values - transform.c) / transform.a - 0.5).astype(np.int32)
row_idx = np.round((df["y"].values - transform.f) / transform.e - 0.5).astype(np.int32)

in_bounds = (row_idx >= 0) & (row_idx < n_rows) & (col_idx >= 0) & (col_idx < n_cols)
if not in_bounds.all():
    n_dropped = (~in_bounds).sum()
    console.print(f"  [yellow]Warning: dropping {n_dropped} out-of-bounds points")
    df = df[in_bounds].reset_index(drop=True)
    row_idx = row_idx[in_bounds]
    col_idx = col_idx[in_bounds]

source_map = {"g": 1.0, "s": 2.0}
source_vals = df["source"].map(source_map).values.astype(np.float32)
source_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
source_arr[row_idx, col_idx] = source_vals

trait_cols = [c for c in df.columns if c not in ("x", "y", "source")]
trait_cols = [
    t
    for t in trait_cols
    if args.overwrite or not (OUT_DIR / f"{t.removesuffix('_mean')}.tif").exists()
]
console.print(
    f"  {len(trait_cols)} traits to write  "
    f"({len([c for c in df.columns if c not in ('x', 'y', 'source')]) - len(trait_cols)} skipped)\n"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-allocate once and reuse — avoids repeated 2 GB allocation per trait
trait_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

with Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.description}"),
    TimeElapsedColumn(),
    console=console,
) as progress:
    for trait in trait_cols:
        task = progress.add_task(
            f"Writing {trait.removesuffix('_mean')} ...", total=None
        )

        trait_arr.fill(np.nan)
        trait_arr[row_idx, col_idx] = df[trait].values.astype(np.float32)

        trait_id = trait.removesuffix("_mean")
        out_path = OUT_DIR / f"{trait_id}.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=n_rows,
            width=n_cols,
            count=2,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=float("nan"),
            compress="zstd",
            zstd_level=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(trait_arr, 1)
            dst.write(source_arr, 2)
            dst.descriptions = ["mean", "source"]

        size_mb = out_path.stat().st_size / 1e6
        progress.update(
            task,
            description=f"[green]✓[/green] {trait_id:<25} {size_mb:.2f} MB",
            completed=1,
            total=1,
        )

console.print(f"\n[bold green]Done![/bold green] Output: {OUT_DIR}")
