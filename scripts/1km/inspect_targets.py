"""
Inspect target rasters (gbif + splot):
  - Verify all files share the same transform and CRS.
  - Verify band ordering is consistent within and across sources.

Usage:
    python inspect_targets.py
"""

from pathlib import Path

import rasterio
from rich.console import Console
from rich.table import Table

RAW_ROOT = Path("/scratch3/plant-traits-v2/data/1km/raw/targets")
TARGETS_ROOT = Path("/scratch3/plant-traits-v2/data/1km/targets")
SOURCES = {
    "GBIF": TARGETS_ROOT / "gbif",
    "sPlot": TARGETS_ROOT / "splot",
}

console = Console()


def inspect_source(
    source_name: str, source_dir: Path, ref_transform, ref_crs, ref_bands
) -> tuple:
    """Returns (transform, crs, bands) from the first file; prints a table of consistency checks."""
    tifs = sorted(source_dir.glob("*.tif"))
    console.rule(f"[bold blue]{source_name}  ({len(tifs)} traits)")

    if not tifs:
        console.print("[red]No .tif files found.")
        return ref_transform, ref_crs, ref_bands

    ref_count = len(list((TARGETS_ROOT / "gbif").glob("*.tif")))
    count_str = (
        f"[green]{len(tifs)}[/green]"
        if len(tifs) == ref_count
        else f"[red]{len(tifs)} (expected {ref_count})[/red]"
    )
    console.print(f"Files: {count_str}")

    with rasterio.open(tifs[0]) as ds:
        first_transform = ds.transform
        first_crs = ds.crs
        first_bands = ds.descriptions
        h, w = ds.shape
        res_x, res_y = abs(ds.transform.a), abs(ds.transform.e)
        console.print(
            f"CRS: {first_crs}  |  Grid: {h} x {w}  |  Resolution: {res_x:.0f} x {res_y:.0f} m"
        )
        console.print(f"Bands ({ds.count}): {first_bands}\n")

    table = Table(show_lines=False, header_style="bold cyan")
    table.add_column("Trait", style="cyan", no_wrap=True)
    table.add_column("Transform", justify="center")
    table.add_column("CRS", justify="center")
    table.add_column("Bands", justify="center")
    table.add_column("Size (MB)", justify="right")

    for tif in tifs:
        size_mb = tif.stat().st_size / 1e6
        with rasterio.open(tif) as ds:
            t_ok = ds.transform == first_transform
            crs_ok = ds.crs == first_crs
            bands_ok = ds.descriptions == first_bands

        # Also check against cross-source reference if provided
        ref_t_ok = ref_transform is None or first_transform == ref_transform
        ref_crs_ok = ref_crs is None or first_crs == ref_crs
        ref_bands_ok = ref_bands is None or first_bands == ref_bands

        t_str = "[green]OK[/green]" if t_ok else "[red]MISMATCH[/red]"
        crs_str = "[green]OK[/green]" if crs_ok else "[red]MISMATCH[/red]"
        bands_str = "[green]OK[/green]" if bands_ok else "[red]MISMATCH[/red]"
        table.add_row(tif.stem, t_str, crs_str, bands_str, f"{size_mb:.1f}")

    console.print(table)

    # Cross-source summary
    if ref_transform is not None:
        cross_t = (
            "[green]OK[/green]"
            if first_transform == ref_transform
            else "[red]MISMATCH vs previous source[/red]"
        )
        cross_crs = (
            "[green]OK[/green]"
            if first_crs == ref_crs
            else "[red]MISMATCH vs previous source[/red]"
        )
        cross_bands = (
            "[green]OK[/green]"
            if first_bands == ref_bands
            else "[red]MISMATCH vs previous source[/red]"
        )
        console.print(
            f"Cross-source — transform: {cross_t}  CRS: {cross_crs}  bands: {cross_bands}"
        )
        if first_bands != ref_bands:
            console.print(f"  previous: {ref_bands}")
            console.print(f"  this:     {first_bands}")

    return first_transform, first_crs, first_bands


def main() -> None:
    ref_transform, ref_crs, ref_bands = None, None, None
    for name, path in SOURCES.items():
        ref_transform, ref_crs, ref_bands = inspect_source(
            name, path, ref_transform, ref_crs, ref_bands
        )
        console.print()


if __name__ == "__main__":
    main()
