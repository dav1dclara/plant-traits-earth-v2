"""Plot per-trait value distributions for train/val/test splits at 1km.

Uses pre-aggregated histograms (single raster pass) instead of per-cell pixel
extraction, which is required at 1km resolution due to the data volume.

Usage:
    python plot_split_distributions.py
    python plot_split_distributions.py --n-traits 5
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ptev2.data.splitting import build_cell_index, build_cell_labels

DATA_DIR = Path("/scratch3/plant-traits-v2/data/1km")
OUT_DIR = Path(__file__).parents[3] / "viz/data/splitting/1km"

H3_RESOLUTION = 2
N_BINS = 50
BAND = 1  # mean band

SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}

plt.rcParams["font.family"] = "monospace"
console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--n-traits", type=int, default=None)
args = parser.parse_args()

# Load splits (train/val/test only — exclude predictor-only "none" cells)
split_file = DATA_DIR / "splits" / f"h3_splits_res{H3_RESOLUTION}_1km.gpkg"
gdf = gpd.read_file(split_file)
gdf = gdf[gdf["split"].isin(SPLIT_COLORS)].reset_index(drop=True)
n_cells = len(gdf)
console.print(f"Cells with labels: [cyan]{n_cells:,}[/cyan]")

# Rasters
splot_rasters = sorted((DATA_DIR / "targets" / "splot").glob("*.tif"))
gbif_rasters = sorted((DATA_DIR / "targets" / "gbif").glob("*.tif"))
if args.n_traits:
    splot_rasters = splot_rasters[: args.n_traits]
    gbif_rasters = gbif_rasters[: args.n_traits]
traits = [r.stem for r in splot_rasters]
console.print(f"Traits: [cyan]{len(traits)}[/cyan]")

with rasterio.open(splot_rasters[0]) as src:
    raster_crs = src.crs.to_string()
    ref_shape = src.shape
    ref_transform = src.transform

# Build cell index from the split GDF polygons
console.print("Building cell index...")
gdf_raster = gdf.to_crs(raster_crs)
cell_labels = build_cell_labels(list(gdf_raster.geometry), ref_shape, ref_transform)
cell_index = build_cell_index(cell_labels, n_cells)
del cell_labels

splits = gdf["split"].values


def build_split_histograms(
    rasters: list[Path],
) -> tuple[dict[str, np.ndarray], list]:
    """Build per-split summed histograms for all traits in a single raster pass."""
    from ptev2.data.splitting import build_histograms_from_rasters

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        hists, bin_edges, _, _, _, _ = build_histograms_from_rasters(
            rasters, [BAND], cell_index, n_cells, N_BINS, progress=progress
        )

    # hists: (n_cells, n_traits, n_bins) — sum by split
    split_hists = {
        split: hists[splits == split].sum(axis=0)  # (n_traits, n_bins)
        for split in SPLIT_COLORS
    }
    return split_hists, bin_edges


console.rule("sPlot histograms")
splot_hists, splot_edges = build_split_histograms(splot_rasters)

console.rule("GBIF histograms")
gbif_hists, gbif_edges = build_split_histograms(gbif_rasters)

# Plot
n_rows = len(traits)
n_cols = 2
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
)

for i, trait in enumerate(traits):
    for col, (hists, edges, source) in enumerate(
        [(splot_hists, splot_edges, "sPlot"), (gbif_hists, gbif_edges, "GBIF")]
    ):
        ax = axes[i, col]
        edge = edges[i]
        if edge is None:
            ax.set_visible(False)
            continue
        centers = (edge[:-1] + edge[1:]) / 2
        for split, color in SPLIT_COLORS.items():
            h = hists[split][i]
            total = h.sum()
            if total == 0:
                continue
            y = h / total
            ax.step(centers, y, color=color, label=split, linewidth=1.2, where="mid")
            ax.fill_between(centers, y, alpha=0.15, color=color, step="mid")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("value")
        ax.set_ylabel("density")

for ax, source in zip(axes[0], ["sPlot", "GBIF"]):
    ax.set_title(source, fontsize=10)

axes[0, 0].legend(frameon=False, fontsize=8)

# Trait labels on the left
fig.tight_layout(rect=[0.04, 0, 1, 1])
for i, trait in enumerate(traits):
    bbox = axes[i, 0].get_position()
    y_center = (bbox.y0 + bbox.y1) / 2
    fig.text(0.01, y_center, trait, va="center", ha="left", fontsize=9, rotation=90)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"split_distributions_res{H3_RESOLUTION}_1km.png"
fig.savefig(out_path, dpi=150, pil_kwargs={"compress_level": 9, "optimize": True})
console.print(f"Saved: [cyan]{out_path}[/cyan]")
