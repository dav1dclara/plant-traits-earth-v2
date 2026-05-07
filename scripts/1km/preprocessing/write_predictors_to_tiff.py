"""
Write predictor columns from eo_predict_imputed.parquet to GeoTIFF rasters.
Band layout and output settings are defined in config/1km/preprocessing/preprocessing.yaml.

Usage:
    python write_predictors_to_tiff.py
"""

import os
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet as pq
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

console = Console()


def col_subfolder(name: str, cfg: DictConfig) -> str:
    for prefix, folder in cfg.predictors.subfolder_rules.items():
        if name.startswith(prefix):
            return folder
    return cfg.predictors.fallback_subfolder


@hydra.main(
    version_base=None,
    config_path="../../../config/1km/preprocessing",
    config_name="preprocessing",
)
def main(cfg: DictConfig) -> None:
    parquet_path = Path(cfg.predictors.parquet_file)
    ref_path = Path(cfg.ref_raster)
    out_base = Path(cfg.predictors.out_dir)
    output_cfg = cfg.predictors.output
    nodata_map = {k: int(v) for k, v in cfg.predictors.nodata.items()}

    console.rule("[bold blue]Preprocessing predictors")

    schema = pq.read_schema(parquet_path)
    dtype_map = {field.name: str(field.type) for field in schema}
    columns = [name for name in dtype_map if name not in ("x", "y")]
    console.print(
        f"[bold]{len(columns)} predictor columns[/bold] across "
        f"{len(set(col_subfolder(c, cfg) for c in columns))} subfolders"
    )

    with rasterio.open(ref_path) as ref:
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
        task = progress.add_task("Loading x/y + snapping to grid ...", total=None)
        xy = pq.read_table(parquet_path, columns=["x", "y"]).to_pandas()
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
            console.print(
                f"  [yellow]Warning: dropping {n_dropped} out-of-bounds points"
            )
        row_idx = row_idx[in_bounds]
        col_idx = col_idx[in_bounds]
        del xy
        progress.update(
            task,
            description=f"Grid indices ready  ({in_bounds.sum():,} valid points)",
            completed=1,
            total=1,
        )

        for col in columns:
            dtype_str = dtype_map[col]
            nodata = nodata_map[dtype_str]
            np_dtype = np.dtype(dtype_str)
            out_dir = out_base / col_subfolder(col, cfg)
            out_dir.mkdir(parents=True, exist_ok=True)

            task = progress.add_task(f"Writing {col} ...", total=None)

            eo_vals = (
                pq.read_table(parquet_path, columns=[col])
                .to_pandas()[col]
                .values[in_bounds]
            )

            arr = np.full((n_rows, n_cols), nodata, dtype=np_dtype)
            arr[row_idx, col_idx] = eo_vals.astype(np_dtype)

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
                nodata=nodata,
                compress=output_cfg.compress,
                zstd_level=output_cfg.zstd_level,
                predictor=output_cfg.predictor,
                tiled=True,
                blockxsize=output_cfg.blockxsize,
                blockysize=output_cfg.blockysize,
            ) as dst:
                dst.write(arr, 1)
                dst.descriptions = [col]

            size_mb = out_path.stat().st_size / 1e6
            progress.update(
                task,
                description=f"[green]✓[/green] [{col_subfolder(col, cfg)}] {col}  {size_mb:.2f} MB",
                completed=1,
                total=1,
            )

    console.print(f"\n[bold green]Done![/bold green] Output: {out_base}")


if __name__ == "__main__":
    main()
