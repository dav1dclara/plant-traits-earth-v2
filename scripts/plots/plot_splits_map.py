"""Plot H3 split cells on a world map.

Usage:
    python scripts/plots/plot_splits_map.py --resolution 1
    python scripts/plots/plot_splits_map.py --resolution 22
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from rich.console import Console

DATA_DIR = Path("/scratch3/plant-traits-v2/data")
OUT_DIR = Path(__file__).parents[2] / "viz" / "data"

H3_RESOLUTION_BY_RES = {"1": 2, "22": 1}

SPLIT_COLORS = {
    "train": "#2196F3",
    "val": "#FF9800",
    "test": "#4CAF50",
    "none": "#9E9E9E",
}

plt.rcParams["font.family"] = "monospace"
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H3 split cells on a world map.")
    parser.add_argument(
        "--resolution",
        choices=["1", "22"],
        required=True,
        help="Data resolution in km.",
    )
    args = parser.parse_args()

    res = args.resolution
    h3_resolution = H3_RESOLUTION_BY_RES[res]
    splits_file = (
        DATA_DIR / f"{res}km" / "splits" / f"h3_splits_res{h3_resolution}_{res}km.gpkg"
    )
    out_path = OUT_DIR / f"{res}km" / "splits" / "split_map.png"

    console.rule(f"[bold]SPLIT MAP — {res}km[/bold]")
    console.print(f"Splits file:  [cyan]{splits_file}[/cyan]")
    console.print(f"Output:       [cyan]{out_path}[/cyan]")

    with console.status("Loading split file..."):
        gdf = gpd.read_file(splits_file).to_crs("EPSG:4326")
    console.print(f"Loaded [cyan]{len(gdf):,}[/cyan] H3 cells")

    proj = ccrs.EqualEarth()
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555")

    for split, color in SPLIT_COLORS.items():
        sub = gdf[gdf["split"] == split]
        if not len(sub):
            continue
        sub.plot(
            ax=ax,
            color=color,
            edgecolor=color,
            linewidth=0.5,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )

    handles = [
        mpatches.Patch(
            color=color, label=f"{split} ({(gdf['split'] == split).sum():,})"
        )
        for split, color in SPLIT_COLORS.items()
        if (gdf["split"] == split).sum() > 0
    ]
    ax.set_global()
    ax.spines["geo"].set_linewidth(0.5)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles),
        frameon=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compress_level": 9, "optimize": True},
    )
    console.print(f"[green]Saved[/green] [cyan]{out_path}[/cyan]")


if __name__ == "__main__":
    main()
