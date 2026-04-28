"""
Re-save GBIF rasters with faster zstd compression (no reordering).

Input:  data/1km/raw/targets/gbif/*.tif
Output: data/1km/targets/gbif/*.tif

Reads and writes one tile at a time — constant memory regardless of raster size.

Usage:
    python recompress_gbif.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

os.environ.setdefault("GDAL_NUM_THREADS", "1")

IN_DIR = Path("/scratch3/plant-traits-v2/data/1km/raw/targets/gbif")
OUT_DIR = Path("/scratch3/plant-traits-v2/data/1km/targets/gbif")

N_WORKERS = 8
ZSTD_LEVEL = 1

console = Console()


def process_tif(tif: Path) -> tuple[str, float]:
    out_path = OUT_DIR / tif.name

    with rasterio.open(tif) as src:
        profile = src.profile.copy()
        profile.update(zstd_level=ZSTD_LEVEL)
        bands = src.descriptions

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = bands
            for _, window in src.block_windows(1):
                for band_idx in range(1, src.count + 1):
                    dst.write(
                        src.read(band_idx, window=window), band_idx, window=window
                    )

    return tif.stem, out_path.stat().st_size / 1e6


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tifs = sorted(IN_DIR.glob("*.tif"))
    console.print(
        f"Found {len(tifs)} GBIF rasters — processing with {N_WORKERS} workers\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids = {
            tif.stem: progress.add_task(f"{tif.stem} ...", total=None) for tif in tifs
        }

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(process_tif, tif): tif for tif in tifs}
            for fut in as_completed(futures):
                stem, size_mb = fut.result()
                progress.update(
                    task_ids[stem],
                    description=f"[green]✓[/green] {stem:<25} {size_mb:.1f} MB",
                    completed=1,
                    total=1,
                )

    console.print(f"\n[bold green]Done![/bold green] Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
