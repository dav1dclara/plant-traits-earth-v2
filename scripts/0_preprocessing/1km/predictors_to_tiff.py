"""
Write predictor columns from eo_predict_imputed.parquet to GeoTIFF rasters.

Only the masked version of each predictor is written — imputed values (flagged by
eo_predict_mask.parquet) are set back to nodata, restoring the original missing values.

Grid is snapped to the EO reference raster (canopy height) to guarantee
pixel-perfect alignment with all other layers.
Output directory: /scratch3/plant-traits-v2/data/1km/predictors/

Usage:
    python predictors_to_tiff.py               # skip already-written files
    python predictors_to_tiff.py --overwrite   # reprocess all columns
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

PARQUET_PATH = Path(
    "/scratch3/plant-traits-v2/data/1km/raw/eo_data/eo_predict_imputed.parquet"
)
MASK_PATH = Path(
    "/scratch3/plant-traits-v2/data/1km/raw/eo_data/eo_predict_mask.parquet"
)
REF_PATH = Path(
    "/scratch3/plant-traits-v2/data/1km/raw/eo_data/canopy_height/ETH_GlobalCanopyHeight_2020_v1.tif"
)
OUT_BASE = Path("/scratch3/plant-traits-v2/data/1km/predictors")

# Maps column name prefix → output subfolder (mirrors the 22km structure)
SUBFOLDER_RULES = [
    ("ETH_", "canopy_height"),
    ("sur_refl_", "modis"),
    ("vodca_", "vodca"),
    ("wc2.1_", "worldclim"),
]
SOILGRIDS_SUBFOLDER = "soilgrids"  # fallback for all remaining non-coordinate columns


def col_subfolder(name: str) -> str:
    for prefix, folder in SUBFOLDER_RULES:
        if name.startswith(prefix):
            return folder
    return SOILGRIDS_SUBFOLDER


# Nodata sentinel per dtype — must be representable in the native integer type
NODATA = {"uint8": np.uint8(255), "int16": np.int16(-9999)}

parser = argparse.ArgumentParser(
    description="Convert EO predictor parquet to GeoTIFFs."
)
parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite existing files (default: skip)."
)
args = parser.parse_args()

console = Console()

# --- Schema inspection (commented out) ---

# schema      = pq.read_schema(PARQUET_PATH)
# mask_schema = pq.read_schema(MASK_PATH)
# table = Table(title=str(PARQUET_PATH.name), show_lines=False)
# table.add_column("#",            style="dim",   justify="right")
# table.add_column("Column",       style="bold cyan")
# table.add_column("dtype (eo)",   style="green")
# table.add_column("dtype (mask)", style="yellow")
# for i, (field, mfield) in enumerate(zip(schema, mask_schema)):
#     table.add_row(str(i), field.name, str(field.type), str(mfield.type))
# console.print(table)
# console.print(f"[dim]{len(schema)} columns total[/dim]")

# --- Pipeline ---

schema = pq.read_schema(PARQUET_PATH)
dtype_map = {field.name: str(field.type) for field in schema}
COLUMNS = [name for name in dtype_map if name not in ("x", "y")]

console.print(
    f"[bold]{len(COLUMNS)} predictor columns[/bold] across "
    f"{len(set(col_subfolder(c) for c in COLUMNS))} subfolders"
)

with rasterio.open(REF_PATH) as ref:
    transform = ref.transform
    crs = ref.crs
    n_rows, n_cols = ref.shape
console.print(f"Reference grid: {n_rows} rows x {n_cols} cols\n")

with Progress(
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.description}"),
    TimeElapsedColumn(),
    console=console,
) as progress:
    # --- Load x/y once to compute grid indices, then free ---
    task = progress.add_task("Loading x/y + snapping to grid ...", total=None)
    xy = pq.read_table(PARQUET_PATH, columns=["x", "y"]).to_pandas()
    col_idx = np.round((xy["x"].values - transform.c) / transform.a - 0.5).astype(
        np.int32
    )
    row_idx = np.round((xy["y"].values - transform.f) / transform.e - 0.5).astype(
        np.int32
    )
    in_bounds = (
        (row_idx >= 0) & (row_idx < n_rows) & (col_idx >= 0) & (col_idx < n_cols)
    )
    if not in_bounds.all():
        n_dropped = (~in_bounds).sum()
        console.print(f"  [yellow]Warning: dropping {n_dropped} out-of-bounds points")
    row_idx = row_idx[in_bounds]
    col_idx = col_idx[in_bounds]
    del xy  # free ~2 GB
    progress.update(
        task,
        description=f"Grid indices ready  ({in_bounds.sum():,} valid points)",
        completed=1,
        total=1,
    )

    # --- Write rasters — load one column at a time to keep memory low ---
    for col in COLUMNS:
        dtype_str = dtype_map[col]
        nodata = NODATA[dtype_str]
        np_dtype = np.dtype(dtype_str)
        out_dir = OUT_BASE / col_subfolder(col)
        existing = [out_dir / f"{col}.tif"]
        if not args.overwrite and all(p.exists() for p in existing):
            progress.console.print(f"  [dim]skip[/dim] {col}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        task = progress.add_task(f"Writing {col} ...", total=None)

        # Load one column at a time (~400 MB peak per iteration)
        eo_vals = (
            pq.read_table(PARQUET_PATH, columns=[col])
            .to_pandas()[col]
            .values[in_bounds]
        )
        mask_vals = (
            pq.read_table(MASK_PATH, columns=[col]).to_pandas()[col].values[in_bounds]
        )

        arr = np.full((n_rows, n_cols), nodata, dtype=np_dtype)
        arr[row_idx, col_idx] = eo_vals.astype(np_dtype)
        arr[row_idx[mask_vals], col_idx[mask_vals]] = nodata

        out_path = out_dir / f"{col}.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=n_rows,
            width=n_cols,
            count=1,
            dtype=dtype_str,
            crs=crs,
            transform=transform,
            nodata=int(nodata),
            compress="zstd",
            zstd_level=3,
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(arr, 1)
            dst.descriptions = [col]

        size_mb = out_path.stat().st_size / 1e6
        progress.update(
            task,
            description=f"[green]✓[/green] [{col_subfolder(col)}] {col}  {size_mb:.2f} MB",
            completed=1,
            total=1,
        )

console.print(f"\n[bold green]Done![/bold green] Output: {OUT_BASE}")
