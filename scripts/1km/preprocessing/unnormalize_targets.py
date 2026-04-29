"""
Reverse Yeo-Johnson + z-score normalisation on 1km GBIF and/or sPlot target rasters.

Transform applied during preprocessing:
    raw  →  Yeo-Johnson(λ)  →  z-score(mean, scale)  →  TIF value

This script inverts both steps:
    TIF value  →  x * scale + mean  →  inverse Yeo-Johnson  →  original units

Parameters are read from power_transformer_params.csv (one row per trait).
Traits without a matching row in the CSV are skipped with a warning.

Usage:
    python unnormalize_targets.py                  # both (default)
    python unnormalize_targets.py --source gbif
    python unnormalize_targets.py --source splot
    python unnormalize_targets.py --source both
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from scipy.special import inv_boxcox1p

os.environ.setdefault("GDAL_NUM_THREADS", "1")

console = Console()

_source = "both"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--source" and _i + 1 < len(sys.argv):
        _source = sys.argv.pop(_i + 1)
        sys.argv.pop(_i)
        break
    elif _arg.startswith("--source="):
        _source = _arg.split("=", 1)[1]
        sys.argv.pop(_i)
        break

if _source not in {"gbif", "splot", "both"}:
    console.print(
        f"[red]Invalid --source '{_source}'. Choose from: gbif, splot, both[/red]"
    )
    sys.exit(1)


def _inverse_yeo_johnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    x = np.empty_like(y)
    pos = y >= 0
    neg = ~pos
    if pos.any():
        x[pos] = inv_boxcox1p(y[pos], lmbda)
    if neg.any():
        x[neg] = -inv_boxcox1p(-y[neg], 2.0 - lmbda)
    return x


def process_tif(
    tif: Path,
    out_dir: Path,
    lmbda: float,
    mean_: float,
    scale_: float,
    output_cfg: dict,
) -> tuple[str, float]:
    out_path = out_dir / tif.name

    with rasterio.open(tif) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=float("nan"),
            count=1,
            compress="zstd",
            zstd_level=output_cfg["zstd_level"],
            tiled=True,
            blockxsize=output_cfg["blockxsize"],
            blockysize=output_cfg["blockysize"],
        )
        nodata = src.nodata
        windows = list(src.block_windows(1))

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.update_tags(1, description="mean")
            for _, window in windows:
                data = src.read(1, window=window).astype(np.float64)
                mask = np.isnan(data) if nodata is None else (data == nodata)
                data[mask] = 0.0
                y = data * scale_ + mean_
                x = _inverse_yeo_johnson(y, lmbda)
                x[mask] = np.nan
                dst.write(x.astype(np.float32), 1, window=window)

    with rasterio.open(out_path, "r+") as dst:
        dst.build_overviews(output_cfg["overviews"], rasterio.enums.Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    return tif.stem, out_path.stat().st_size / 1e6


def run_source(
    label: str, in_dir: Path, out_dir: Path, params: pd.DataFrame, cfg
) -> None:
    output_cfg = {
        "zstd_level": cfg.output.zstd_level,
        "blockxsize": cfg.output.blockxsize,
        "blockysize": cfg.output.blockysize,
        "overviews": list(cfg.output.overviews),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = sorted(in_dir.glob("*.tif"))

    console.rule(f"[bold blue]{label}")
    console.print(f"Found {len(tifs)} rasters in {in_dir}\n")

    matched, skipped = [], []
    for tif in tifs:
        trait = tif.stem
        if trait in params.index:
            matched.append(tif)
        else:
            skipped.append(tif.name)

    if skipped:
        console.print(
            f"[yellow]Skipping {len(skipped)} traits with no transform params:[/yellow]"
        )
        for name in skipped:
            console.print(f"  [dim]{name}[/dim]")
        console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids = {
            tif.stem: progress.add_task(f"{tif.stem} ...", total=None)
            for tif in matched
        }

        with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
            futures = {
                pool.submit(
                    process_tif,
                    tif,
                    out_dir,
                    float(params.loc[tif.stem, "yeo_johnson_lambda"]),
                    float(params.loc[tif.stem, "standardize_mean"]),
                    float(params.loc[tif.stem, "standardize_scale"]),
                    output_cfg,
                ): tif
                for tif in matched
            }
            for fut in as_completed(futures):
                stem, size_mb = fut.result()
                progress.update(
                    task_ids[stem],
                    description=f"[green]✓[/green] {stem:<25} {size_mb:.1f} MB",
                    completed=1,
                    total=1,
                )

    console.print(
        f"\n[bold green]Done![/bold green] {len(matched)} rasters → {out_dir}\n"
    )


@hydra.main(
    version_base=None,
    config_path="../../../config/1km",
    config_name="preprocessing",
)
def main(cfg: DictConfig) -> None:
    params = pd.read_csv(cfg.unnormalize.transform_params, index_col="trait")
    console.print(f"Loaded transform params for {len(params)} traits.\n")

    if _source in {"gbif", "both"}:
        run_source(
            "GBIF",
            Path(cfg.unnormalize.gbif.in_dir),
            Path(cfg.unnormalize.gbif.out_dir),
            params,
            cfg.unnormalize.gbif,
        )
    if _source in {"splot", "both"}:
        run_source(
            "sPlot",
            Path(cfg.unnormalize.splot.in_dir),
            Path(cfg.unnormalize.splot.out_dir),
            params,
            cfg.unnormalize.splot,
        )


if __name__ == "__main__":
    main()
