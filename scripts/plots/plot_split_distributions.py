"""Plot per-trait value distributions for train/val/test splits.

Uses pre-aggregated histograms (single raster pass) for memory efficiency.
Plots two rows (sPlot / GBIF) and one column per trait, with the value axis
on the y-axis and density on the x-axis (horizontal histogram).

Usage:
    python scripts/plots/plot_split_distributions.py --resolution 1
    python scripts/plots/plot_split_distributions.py --resolution 22
    python scripts/plots/plot_split_distributions.py --resolution 22 --n-traits 5
"""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ptev2.data.splitting import build_cell_index, build_cell_labels

DATA_DIR = Path("/scratch3/plant-traits-v2/data")
OUT_DIR = Path(__file__).parents[2] / "viz" / "data"

H3_RESOLUTION_BY_RES = {"1": 2, "22": 1}

SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
N_BINS = 50
BAND = 1  # mean band
TRAITS_PER_ROW = 5

plt.rcParams["font.family"] = "monospace"
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split distributions.")
    parser.add_argument(
        "--resolution",
        choices=["1", "22"],
        required=True,
        help="Data resolution in km.",
    )
    parser.add_argument(
        "--n-traits",
        type=int,
        default=None,
        help="Only plot the first N traits (default: all).",
    )
    args = parser.parse_args()

    res = args.resolution
    h3_resolution = H3_RESOLUTION_BY_RES[res]
    res_dir = DATA_DIR / f"{res}km"
    splot_dir = res_dir / "targets" / "splot"
    gbif_dir = res_dir / "targets" / "gbif"
    splits_file = res_dir / "splits" / f"h3_splits_res{h3_resolution}_{res}km.gpkg"

    console.rule(f"[bold]SPLIT DISTRIBUTIONS — {res}km[/bold]")
    console.print(f"Splits file:  [cyan]{splits_file}[/cyan]")
    console.print(f"sPlot dir:    [cyan]{splot_dir}[/cyan]")
    console.print(f"GBIF dir:     [cyan]{gbif_dir}[/cyan]")

    # Load splits (train/val/test only — exclude predictor-only "none" cells)
    gdf = gpd.read_file(splits_file)
    gdf = gdf[gdf["split"].isin(SPLIT_COLORS)].reset_index(drop=True)
    n_cells = len(gdf)
    console.print(f"Cells with labels: [cyan]{n_cells:,}[/cyan]")

    splot_rasters = sorted(splot_dir.glob("*.tif"))
    gbif_rasters = sorted(gbif_dir.glob("*.tif"))
    if args.n_traits:
        splot_rasters = splot_rasters[: args.n_traits]
        gbif_rasters = gbif_rasters[: args.n_traits]
    traits = [r.stem for r in splot_rasters]
    console.print(f"Traits: [cyan]{len(traits)}[/cyan]")

    with rasterio.open(splot_rasters[0]) as src:
        raster_crs = src.crs.to_string()
        ref_shape = src.shape
        ref_transform = src.transform

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

    for source, hists, label in [
        ("sPlot", splot_hists, "sPlot pixels per trait per split"),
        ("GBIF", gbif_hists, "GBIF pixels per trait per split"),
    ]:
        table = Table(title=label, show_footer=True)
        table.add_column("Trait", style="bold")
        totals = {split: 0 for split in SPLIT_COLORS}
        for split, color in SPLIT_COLORS.items():
            per_trait = hists[split].sum(axis=1).astype(int)  # (n_traits,)
            totals[split] = int(per_trait.sum())
            table.add_column(
                f"[{color}]{split}[/{color}]",
                justify="right",
                footer=f"{totals[split]:,}",
            )
        for i, trait in enumerate(traits):
            table.add_row(
                trait,
                *[f"{int(hists[split][i].sum()):,}" for split in SPLIT_COLORS],
            )
        console.print(table)

    # Layout: for each batch of TRAITS_PER_ROW traits, one sPlot row then one GBIF row.
    # row = batch * 2 + source_idx  (source_idx: 0=sPlot, 1=GBIF)
    n_trait_rows = math.ceil(len(traits) / TRAITS_PER_ROW)
    n_rows = 2 * n_trait_rows
    n_cols = min(TRAITS_PER_ROW, len(traits))
    cell_size = 3  # inches per subplot (square)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_size * n_cols, cell_size * n_rows),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_aspect("auto")
        ax.set_box_aspect(1)

    sources = [
        ("sPlot", splot_hists, splot_edges),
        ("GBIF", gbif_hists, gbif_edges),
    ]

    for source_idx, (source, hists, edges) in enumerate(sources):
        for trait_idx, trait in enumerate(traits):
            batch = trait_idx // TRAITS_PER_ROW
            col = trait_idx % TRAITS_PER_ROW
            row = batch * 2 + source_idx
            ax = axes[row, col]
            edge = edges[trait_idx]
            if edge is None:
                ax.set_visible(False)
                continue
            centers = (edge[:-1] + edge[1:]) / 2
            bin_width = edge[1] - edge[0]
            for split, color in SPLIT_COLORS.items():
                h = hists[split][trait_idx]
                total = h.sum()
                if total == 0:
                    continue
                y = h / (total * bin_width)
                ax.step(
                    centers, y, color=color, label=split, linewidth=1.2, where="mid"
                )
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("value")
            ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Hide unused subplots in the last batch
    n_last = len(traits) - (n_trait_rows - 1) * TRAITS_PER_ROW
    for source_idx in range(2):
        for col in range(n_last, n_cols):
            axes[(n_trait_rows - 1) * 2 + source_idx, col].set_visible(False)

    # Uniform y-axis limits across all visible subplots
    global_ymax = max(ax.get_ylim()[1] for ax in axes.flat if ax.get_visible())
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_ylim(0, global_ymax)

    # Uniform x-axis limits, centred at 0
    all_edges = [e for e in splot_edges + gbif_edges if e is not None]
    global_xmax = max(np.abs(e).max() for e in all_edges)
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlim(-global_xmax, global_xmax)

    # Trait names on the sPlot row of each batch
    for batch in range(n_trait_rows):
        for col in range(n_cols):
            trait_idx = batch * TRAITS_PER_ROW + col
            if trait_idx < len(traits):
                axes[batch * 2, col].set_title(
                    traits[trait_idx], fontsize=9, fontweight="bold"
                )

    # Source labels on the left of every sPlot/GBIF row pair
    for batch in range(n_trait_rows):
        for source_idx, (source, _, _) in enumerate(sources):
            axes[batch * 2 + source_idx, 0].annotate(
                source,
                xy=(0, 0.5),
                xycoords="axes fraction",
                xytext=(-0.45, 0.5),
                textcoords="axes fraction",
                fontsize=10,
                fontweight="bold",
                va="center",
                ha="right",
            )

    fig.tight_layout(h_pad=3.0)

    # Horizontal separator lines between batch pairs (drawn after layout so positions are final)
    for batch in range(n_trait_rows - 1):
        y_bottom = axes[batch * 2 + 1, 0].get_position().y0
        y_top = axes[(batch + 1) * 2, 0].get_position().y1
        y_mid = (y_bottom + y_top) / 2
        fig.add_artist(
            plt.Line2D(
                [0.02, 0.98],
                [y_mid, y_mid],
                transform=fig.transFigure,
                color="#aaaaaa",
                linewidth=1.0,
                linestyle="--",
            )
        )

    out_dir = OUT_DIR / f"{res}km" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"split_distributions_res{h3_resolution}_{res}km.png"
    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
        pil_kwargs={"compress_level": 9, "optimize": True},
    )
    console.print(f"Saved: [cyan]{out_path}[/cyan]")


if __name__ == "__main__":
    main()
