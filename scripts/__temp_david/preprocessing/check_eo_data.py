"""
Check all GeoTIFF files in the 1km EO data directory for corruption or read errors.
"""

import csv
from pathlib import Path

import rasterio
from rich.console import Console

EO_DATA_DIR = Path("/scratch3/plant-traits-v2/data/1km/eo_data")
OUT_FILE = Path(__file__).parent / "eo_data_1km.csv"

console = Console()

tif_files = sorted(EO_DATA_DIR.rglob("*.tif"))
console.print(
    f"Found [cyan]{len(tif_files)}[/cyan] files in [cyan]{EO_DATA_DIR}[/cyan]\n"
)

corrupted = []
rows = []

for path in tif_files:
    rel = path.relative_to(EO_DATA_DIR)
    try:
        with rasterio.open(path) as src:
            # Read all bands to catch truncated or unreadable data
            data = src.read()
            if data.size == 0:
                raise ValueError("File opened but contains no data")
        console.print(f"[green]OK[/green]     {rel}")
        rows.append({"filename": str(rel), "status": "OK", "error": ""})
    except Exception as e:
        corrupted.append((path, e))
        console.print(f"[red]FAILED[/red] {rel}: {e}")
        rows.append({"filename": str(rel), "status": "FAILED", "error": str(e)})

console.rule()
if corrupted:
    console.print(
        f"[bold red]{len(corrupted)} corrupted / unreadable file(s).[/bold red]"
    )
else:
    console.print(
        f"[bold green]All {len(tif_files)} files read successfully.[/bold green]"
    )

with OUT_FILE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "status", "error"])
    writer.writeheader()
    writer.writerows(rows)

console.print(f"\nResults written to [cyan]{OUT_FILE}[/cyan]")
