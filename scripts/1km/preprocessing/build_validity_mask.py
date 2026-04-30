"""Build a predictor validity mask.

Reads all predictor rasters under predictors.out_dir and writes a uint8 GeoTIFF
where 1 means at least one predictor has valid (non-nodata) data at that pixel, and
0 means no predictor has data. Used by the chipping pipeline to include all
informative chips in the "all" split, regardless of H3 coverage.

Usage:
    python build_validity_mask.py
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import rasterio
import rasterio.windows
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

STRIP_HEIGHT = 512


@hydra.main(
    version_base=None,
    config_path="../../../config/1km",
    config_name="preprocessing",
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold]BUILDING PREDICTOR VALIDITY MASK[/bold]")

    predictors_dir = Path(cfg.predictors.out_dir)
    out_path = Path(cfg.validity_mask)

    predictor_paths = sorted(predictors_dir.glob("**/*.tif"))

    if not predictor_paths:
        raise ValueError(f"No .tif files found under {predictors_dir}")

    console.print(f"Predictors dir:  [cyan]{predictors_dir}[/cyan]")
    console.print(f"Output:          [cyan]{out_path}[/cyan]")
    console.print(f"Predictor files: [cyan]{len(predictor_paths)}[/cyan]")
    console.rule()

    srcs = [rasterio.open(p) for p in predictor_paths]
    ref = srcs[0]
    height, width, crs, transform = ref.height, ref.width, ref.crs, ref.transform

    console.print(f"Grid: [cyan]{height} × {width}[/cyan] pixels")
    console.print(f"CRS:  [cyan]EPSG:{crs.to_epsg()}[/cyan]")
    console.rule()

    valid_mask = np.zeros((height, width), dtype=np.uint8)
    n_strips = math.ceil(height / STRIP_HEIGHT)

    def _read_mask_strip(args: tuple) -> np.ndarray:
        src, window = args
        return src.read_masks(1, window=window)  # (h, w), values 0 or 255 — band 1 only

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Reading predictor masks", total=n_strips)
        with ThreadPoolExecutor(max_workers=min(32, len(srcs))) as executor:
            for i in range(n_strips):
                y0 = i * STRIP_HEIGHT
                y1 = min(y0 + STRIP_HEIGHT, height)
                window = rasterio.windows.Window(0, y0, width, y1 - y0)

                # executor.map completes all strip-i tasks before returning,
                # so the same file handle is never accessed by two threads at once.
                masks = list(
                    executor.map(_read_mask_strip, [(src, window) for src in srcs])
                )

                strip_valid = np.zeros((y1 - y0, width), dtype=bool)
                for mask in masks:
                    strip_valid |= mask > 0

                valid_mask[y0:y1] = strip_valid.astype(np.uint8)
                progress.advance(task)

    for src in srcs:
        src.close()

    n_valid = int((valid_mask > 0).sum())
    n_total = height * width
    console.print(
        f"\nValid pixels: [cyan]{n_valid:,}[/cyan] / {n_total:,} "
        f"([cyan]{100 * n_valid / n_total:.1f}%[/cyan])"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with console.status(f"Writing [cyan]{out_path}[/cyan]..."):
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            nodata=0,
            compress="zstd",
            zstd_level=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(valid_mask, 1)
        size_mb = out_path.stat().st_size / 1e6

    console.print(
        f"[green]Done.[/green] Saved [cyan]{out_path}[/cyan] ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
