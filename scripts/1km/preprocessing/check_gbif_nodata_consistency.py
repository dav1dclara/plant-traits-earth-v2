"""
Check that all GBIF and sPlot target rasters share the same nodata pixel locations.

Scans rasters window by window to stay memory-efficient regardless of raster size.
A pixel is flagged as inconsistent if it is nodata in some rasters but not others.

Usage:
    python check_gbif_nodata_consistency.py
"""

from pathlib import Path

import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def check_dir(tif_dir: Path, label: str) -> None:
    tifs = sorted(tif_dir.glob("*.tif"))

    if not tifs:
        console.print(f"[red]No rasters found in {tif_dir}[/red]")
        return

    console.rule(f"[bold blue]{label} nodata consistency check")
    console.print(f"Directory: [cyan]{tif_dir}[/cyan]")
    console.print(f"Rasters:   [cyan]{len(tifs)}[/cyan]\n")

    with rasterio.open(tifs[0]) as ref:
        profile = ref.profile
        windows = [w for _, w in ref.block_windows(1)]
        nodata_val = ref.nodata

    console.print(f"Nodata value: [cyan]{nodata_val}[/cyan]")
    console.print(
        f"Grid: {profile['height']} x {profile['width']}, {len(windows)} blocks\n"
    )

    srcs = [rasterio.open(t) for t in tifs]
    n = len(tifs)
    total_inconsistent = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning windows ...", total=len(windows))

            for window in windows:
                masks = np.stack(
                    [src.read(1, window=window) == nodata_val for src in srcs]
                )
                nodata_count = masks.sum(axis=0)
                inconsistent = (nodata_count > 0) & (nodata_count < n)
                total_inconsistent += int(inconsistent.sum())
                progress.advance(task)

    finally:
        for src in srcs:
            src.close()

    console.print()
    if total_inconsistent == 0:
        console.print(
            f"[bold green]✓ All {n} rasters share identical nodata masks.[/bold green]"
        )
    else:
        console.print(
            f"[bold red]✗ {total_inconsistent:,} pixels have inconsistent nodata "
            f"across the {n} rasters.[/bold red]"
        )


@hydra.main(
    version_base=None,
    config_path="../../../config/1km",
    config_name="preprocessing",
)
def main(cfg: DictConfig) -> None:
    check_dir(Path(cfg.targets.splot.out_dir), "sPlot")


if __name__ == "__main__":
    main()
