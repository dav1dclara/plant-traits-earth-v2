"""
Recompress 1km prediction GeoTIFFs from float32 → float16 with ZSTD.

Typical result: 478 MB → ~90 MB per raster (~80% reduction), with max ~0.25
absolute error per pixel — negligible vs prediction std ≈ 15.

Usage:
    python compress_predictions.py
    python compress_predictions.py --in-dir DIR --out-dir DIR --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

DEFAULT_IN_DIR = Path("data/1km/predictions")
DEFAULT_OUT_DIR = DEFAULT_IN_DIR / "float16"

console = Console()


def compress_tif(tif: Path, out_dir: Path, zstd_level: int) -> tuple[str, float, float]:
    out_path = out_dir / tif.name

    with rasterio.open(tif) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float16",
            nodata=float("nan"),
            compress="zstd",
            zstd_level=zstd_level,
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window).astype(np.float16)
                dst.write(block, 1, window=window)

    return tif.stem, tif.stat().st_size / 1e6, out_path.stat().st_size / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zstd-level", type=int, default=15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tifs = sorted(args.in_dir.glob("*.tif"))
    console.print(f"Found {len(tifs)} rasters in {args.in_dir}\n")
    if not tifs:
        return

    total_in = total_out = 0.0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids = {
            tif.stem: progress.add_task(f"{tif.stem} ...", total=None) for tif in tifs
        }
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(compress_tif, tif, args.out_dir, args.zstd_level): tif
                for tif in tifs
            }
            for fut in as_completed(futures):
                stem, in_mb, out_mb = fut.result()
                total_in += in_mb
                total_out += out_mb
                ratio = 100 * (1 - out_mb / in_mb)
                progress.update(
                    task_ids[stem],
                    description=(
                        f"[green]done[/green] {stem:<25} "
                        f"{in_mb:.0f} -> {out_mb:.0f} MB ({ratio:.0f}%)"
                    ),
                    completed=1,
                    total=1,
                )

    console.print(
        f"\n[bold green]Done![/bold green] {len(tifs)} rasters -> {args.out_dir}"
    )
    console.print(
        f"Total: {total_in:.0f} MB -> {total_out:.0f} MB "
        f"({100 * (1 - total_out / total_in):.0f}% reduction)"
    )


if __name__ == "__main__":
    main()
