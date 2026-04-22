"""
Convert trait columns from Y.parquet to GeoTIFFs, one file per trait.
Band layout and output settings are defined in config/preprocessing/1km.yaml.

Usage:
    python write_targets_to_tiff.py
"""

# TODO: normalization?
# TODO:

import json
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

console = Console()


@hydra.main(
    version_base=None,
    config_path="../../../config/preprocessing",
    config_name="1km",
)
def main(cfg: DictConfig) -> None:
    resolution_km = cfg.resolution_km

    console.rule(f"[bold blue]Preprocessing targets at {resolution_km} km resolution")

    console.print("[bold]Loading reference raster ...[/bold]")
    ref_raster = Path(cfg.ref_raster)
    console.print(f"Loading raster from: {ref_raster}")

    with rasterio.open(ref_raster) as ref:
        transform = ref.transform
        crs = ref.crs
        n_rows, n_cols = ref.shape
    console.print(f"Reference grid: {n_rows} rows x {n_cols} cols  CRS: {crs}")

    console.print(f"[bold]\nLoading targets parquet file ...[/bold]")
    parquet_file = cfg.targets.parquet_file
    console.print(f"Loading parquet from: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    console.print(f"  {len(df):,} rows")

    # Parquet x/y are pixel centres; transform.c/f are pixel edges — subtract 0.5 to align.
    col_idx = np.round((df["x"].values - transform.c) / transform.a - 0.5).astype(
        np.int32
    )
    row_idx = np.round((df["y"].values - transform.f) / transform.e - 0.5).astype(
        np.int32
    )

    in_bounds = (
        (row_idx >= 0) & (row_idx < n_rows) & (col_idx >= 0) & (col_idx < n_cols)
    )
    if not in_bounds.all():
        n_dropped = (~in_bounds).sum()
        console.print(f"  [yellow]Warning: dropping {n_dropped} out-of-bounds points")
        df = df[in_bounds].reset_index(drop=True)
        row_idx = row_idx[in_bounds]
        col_idx = col_idx[in_bounds]

    source_map = {k: float(v) for k, v in cfg.targets.source_map.items()}
    source_vals = df["source"].map(source_map).values.astype(np.float32)
    source_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    source_arr[row_idx, col_idx] = source_vals

    with open(cfg.trait_mapping) as f:
        trait_mapping = json.load(f)

    gbif_val = source_map.get("g")
    splot_val = source_map.get("s")
    trait_cols = [c for c in df.columns if c not in ("x", "y", "source")]
    n_total = len(df)

    table = Table(title="Trait data summary", show_lines=False)
    table.add_column("Trait", style="cyan", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Valid", justify="right")
    table.add_column("Valid %", justify="right")
    table.add_column("GBIF %", justify="right")
    table.add_column("sPlot %", justify="right")

    for col in trait_cols:
        trait_id = col.removesuffix("_mean").lstrip("X")
        short_name = trait_mapping.get(trait_id, {}).get("short", "")
        valid_mask = df[col].notna().values
        n_valid = int(valid_mask.sum())
        pct_valid = n_valid / n_total
        n_gbif = int(((source_vals == gbif_val) & valid_mask).sum())
        n_splot = int(((source_vals == splot_val) & valid_mask).sum())
        pct_gbif = n_gbif / n_valid if n_valid > 0 else 0.0
        pct_splot = n_splot / n_valid if n_valid > 0 else 0.0
        color = "green" if n_valid == n_total else "yellow"
        table.add_row(
            col.removesuffix("_mean"),
            short_name,
            f"[{color}]{n_valid:,}[/{color}]",
            f"[{color}]{pct_valid:.1%}[/{color}]",
            f"{pct_gbif:.1%}",
            f"{pct_splot:.1%}",
        )

    console.print(table)

    out_dir = Path(cfg.base_dir) / "targets" / cfg.targets.source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate once and reuse — avoids repeated 2 GB allocation per trait
    trait_arr = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    output_cfg = cfg.targets.output

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
            out_path = out_dir / f"{trait_id}.tif"
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
                compress=output_cfg.compress,
                zstd_level=output_cfg.zstd_level,
                tiled=True,
                blockxsize=output_cfg.blockxsize,
                blockysize=output_cfg.blockysize,
            ) as dst:
                band_data = {"mean": trait_arr, "source": source_arr}
                for band_idx, band_name in cfg.targets.bands.items():
                    dst.write(band_data[band_name], int(band_idx))
                dst.descriptions = list(cfg.targets.bands.values())

            size_mb = out_path.stat().st_size / 1e6
            progress.update(
                task,
                description=f"[green]✓[/green] {trait_id:<25} {size_mb:.2f} MB",
                completed=1,
                total=1,
            )

    console.print(f"\n[bold green]Done![/bold green] Output: {out_dir}")


if __name__ == "__main__":
    main()
