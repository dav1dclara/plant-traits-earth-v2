"""
Re-save sPlot rasters with band order matching GBIF (mean, std, median, q05, q95, count).

Input:  data/1km/raw/targets/splot/*.tif  (band order: mean, count, std, median, q05, q95)
Output: data/1km/targets/splot/*.tif      (band order: mean, std, median, q05, q95, count)

Reads and writes one tile at a time — constant memory regardless of raster size.

Usage:
    python reorder_splot_bands.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

os.environ.setdefault("GDAL_NUM_THREADS", "1")

IN_DIR = Path("/scratch3/plant-traits-v2/data/1km/raw/targets/splot")
OUT_DIR = Path("/scratch3/plant-traits-v2/data/1km/targets/splot")
GBIF_REF = Path("/scratch3/plant-traits-v2/data/1km/raw/targets/gbif/X13.tif")

N_WORKERS = 8
ZSTD_LEVEL = 1

console = Console()


def process_tif(tif: Path, target_bands: tuple) -> tuple[str, float]:
    out_path = OUT_DIR / tif.name

    with rasterio.open(tif) as src:
        src_bands = src.descriptions
        band_order = [src_bands.index(b) + 1 for b in target_bands]
        profile = src.profile.copy()
        profile.update(zstd_level=ZSTD_LEVEL)
        windows = list(src.block_windows(1))

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = target_bands
            for _, window in windows:
                for out_idx, src_idx in enumerate(band_order, start=1):
                    tile = src.read(src_idx, window=window)
                    dst.write(tile, out_idx, window=window)

    return tif.stem, out_path.stat().st_size / 1e6


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(GBIF_REF) as ref:
        target_bands = ref.descriptions

    console.print(f"Target band order: {target_bands}")

    tifs = sorted(IN_DIR.glob("*.tif"))
    console.print(
        f"Found {len(tifs)} sPlot rasters — processing with {N_WORKERS} workers\n"
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
            futures = {pool.submit(process_tif, tif, target_bands): tif for tif in tifs}
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
