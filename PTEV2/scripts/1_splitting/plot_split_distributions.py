"""
Plot per-trait value distributions for train, val, and test splits.

For each trait, overlays the three distributions on a single axis.
Reads split assignments from h3_unified.gpkg and samples values from GBIF rasters.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
from rich.progress import track
from scipy.stats import gaussian_kde

SPLITS_DIR = Path("/scratch3/plant-traits-v2/data/22km/splits")
COMB_DIR = Path("/scratch3/plant-traits-v2/data/22km/targets/comb")
OUT_DIR = Path(__file__).parents[2] / "viz" / "splits"

H3_RESOLUTION = 2
SOURCE = "comb"
DATA_RES = "22km"
SPLITS_FILE = SPLITS_DIR / f"h3_splits_res{H3_RESOLUTION}_{SOURCE}_{DATA_RES}.gpkg"
BANDS = ["mean", "std", "median", "q05", "q95", "source"]
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}

MAX_TRAITS = None  # TODO: remove to plot all traits

plt.rcParams["font.family"] = "monospace"


def extract_cell_values_all_bands(
    raster_path: Path, cell_polygons_raster_crs: list
) -> dict[str, list[np.ndarray]]:
    """Returns {band_name: [array_of_values_per_cell]}."""
    with rasterio.open(raster_path) as src:
        cell_labels = rasterio.features.rasterize(
            [
                (geom.__geo_interface__, idx + 1)
                for idx, geom in enumerate(cell_polygons_raster_crs)
            ],
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype=np.int32,
        )
        descriptions = [d.lower() for d in src.descriptions]
        band_data = {}
        for band_name in BANDS:
            band_idx = descriptions.index(band_name) + 1
            data = src.read(band_idx).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            band_data[band_name] = [
                (lambda v: v[np.isfinite(v)])(data[cell_labels == idx + 1])
                for idx in range(len(cell_polygons_raster_crs))
            ]
    return band_data


def plot_trait_row(
    axes: np.ndarray,
    trait: str,
    gdf: gpd.GeoDataFrame,
    cell_polys: list,
) -> None:
    raster_path = COMB_DIR / f"{trait}.tif"
    band_cell_values = extract_cell_values_all_bands(raster_path, cell_polys)

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
            ax.set_xticklabels(["GBIF", "SPLOT"])
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

    # Legend in the mean subplot (first), one item per row
    axes[0].legend(frameon=False, fontsize=8, ncol=1, loc="upper left")


def main() -> None:
    gdf = gpd.read_file(SPLITS_FILE)
    traits = sorted(p.stem for p in COMB_DIR.glob("*.tif"))[:MAX_TRAITS]

    with rasterio.open(COMB_DIR / f"{traits[0]}.tif") as src:
        raster_crs = src.crs.to_string()

    gdf_raster = gdf.to_crs(raster_crs)
    cell_polys = list(gdf_raster.geometry)

    n_rows = len(traits)
    n_cols = len(BANDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    for row, trait in enumerate(
        track(traits, description="Extracting pixel values...")
    ):
        plot_trait_row(axes[row], trait, gdf, cell_polys)

    # Column titles (band names) on top row only
    for ax, band_name in zip(axes[0], BANDS):
        ax.set_title(band_name, fontsize=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        OUT_DIR / f"split_distributions_res{H3_RESOLUTION}_{SOURCE}_{DATA_RES}.png"
    )
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    # Add row labels after tight_layout so positions are finalised
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
