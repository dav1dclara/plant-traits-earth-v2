"""
Plot per-trait value distributions for train, val, and test splits.

For each trait, overlays the three distributions on a single axis.
Reads split assignments from the split GeoPackage and samples values from target rasters.

Usage:
    python plot_split_distributions.py --resolution 22   # default
    python plot_split_distributions.py --resolution 1
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rich.progress import track
from scipy.stats import gaussian_kde

from ptev2.data.splitting import build_cell_index, build_cell_labels

DATA_DIR = Path("/scratch3/plant-traits-v2/data")
OUT_DIR = Path(__file__).parents[2] / "viz" / "splits"

BANDS_BY_RES = {
    "22": ["mean", "std", "median", "q05", "q95", "source"],
    "1": ["mean", "source"],
}

H3_RESOLUTION_BY_RES = {"22": 2, "1": 1}
SOURCE = "comb"
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}

plt.rcParams["font.family"] = "monospace"

parser = argparse.ArgumentParser(description="Plot split distributions.")
parser.add_argument(
    "--resolution",
    choices=["1", "22"],
    default="22",
    help="Data resolution in km (default: 22).",
)
parser.add_argument(
    "--n-traits",
    type=int,
    default=None,
    help="Only plot the first N traits (default: all).",
)
args = parser.parse_args()

RES = args.resolution
MAX_TRAITS = args.n_traits
H3_RESOLUTION = H3_RESOLUTION_BY_RES[RES]
BANDS = BANDS_BY_RES[RES]
COMB_DIR = DATA_DIR / f"{RES}km" / "targets" / SOURCE
SPLITS_FILE = (
    DATA_DIR
    / f"{RES}km"
    / "splits"
    / f"h3_splits_res{H3_RESOLUTION}_{SOURCE}_{RES}km.gpkg"
)


def extract_cell_values_all_bands(
    raster_path: Path,
    cell_polygons_raster_crs: list,
    cell_index: tuple[np.ndarray, np.ndarray],
) -> dict[str, list[np.ndarray]]:
    """Returns {band_name: [array_of_values_per_cell]} using pre-computed sort index."""
    n_cells = len(cell_polygons_raster_crs)
    order, boundaries = cell_index
    with rasterio.open(raster_path) as src:
        descriptions = [d.lower() for d in src.descriptions]
        band_data = {}
        for band_name in BANDS:
            band_idx = descriptions.index(band_name) + 1
            data = src.read(band_idx).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            sorted_band = data.ravel()[order]
            cell_arrays = []
            for idx in range(n_cells):
                v = sorted_band[boundaries[idx + 1] : boundaries[idx + 2]]
                cell_arrays.append(v[np.isfinite(v)])
            band_data[band_name] = cell_arrays
    return band_data


def plot_trait_row(
    axes: np.ndarray,
    trait: str,
    gdf: gpd.GeoDataFrame,
    cell_polys: list,
    cell_index: tuple[np.ndarray, np.ndarray],
) -> None:
    raster_path = COMB_DIR / f"{trait}.tif"
    band_cell_values = extract_cell_values_all_bands(
        raster_path, cell_polys, cell_index
    )

    for ax, band_name in zip(axes, BANDS):
        split_values: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
        for vals, split in zip(band_cell_values[band_name], gdf["split"]):
            if split in split_values:
                split_values[split].append(vals)

        pooled = {s: np.concatenate(vs) for s, vs in split_values.items() if vs}

        if band_name == "source":
            splits = ["train", "val", "test"]
            x = np.arange(2)
            width = 0.25
            for i, split in enumerate(splits):
                vals = pooled.get(split, np.array([]))
                if len(vals) == 0:
                    continue
                gbif_frac = np.mean(vals == 1)
                splot_frac = np.mean(vals == 2)
                ax.bar(
                    x + i * width,
                    [gbif_frac, splot_frac],
                    width,
                    color=SPLIT_COLORS[split],
                    alpha=0.8,
                    label=split,
                )
            ax.set_xticks(x + width)
            ax.set_xticklabels(["GBIF", "sPlot"])
            ax.set_ylabel("fraction")
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
            continue

        for split, vals in pooled.items():
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals, bw_method="scott")
            x = np.linspace(vals.min(), vals.max(), 500)
            y = kde(x)
            color = SPLIT_COLORS[split]
            ax.plot(
                x, y, color=color, linewidth=1.5, label=f"{split} (n={len(vals):,})"
            )
            ax.fill_between(x, y, alpha=0.15, color=color)

        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8, ncol=1, loc="upper left")


def main() -> None:
    gdf = gpd.read_file(SPLITS_FILE)
    traits = sorted(p.stem for p in COMB_DIR.glob("*.tif"))[:MAX_TRAITS]

    with rasterio.open(COMB_DIR / f"{traits[0]}.tif") as src:
        raster_crs = src.crs.to_string()
        ref_shape = src.shape
        ref_transform = src.transform

    gdf_raster = gdf.to_crs(raster_crs)
    cell_polys = list(gdf_raster.geometry)

    print("Pre-computing cell index...")
    cell_labels = build_cell_labels(cell_polys, ref_shape, ref_transform)
    cell_index = build_cell_index(cell_labels, len(cell_polys))
    del cell_labels

    n_rows = len(traits)
    n_cols = len(BANDS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    for row, trait in enumerate(track(traits, description="Plotting traits...")):
        plot_trait_row(axes[row], trait, gdf, cell_polys, cell_index)

    for ax, band_name in zip(axes[0], BANDS):
        ax.set_title(band_name, fontsize=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"split_distributions_res{H3_RESOLUTION}_{SOURCE}_{RES}km.png"
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    for row, trait in enumerate(traits):
        bbox = axes[row, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.01, y_center, trait, va="center", ha="left", fontsize=11, rotation=90
        )

    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
