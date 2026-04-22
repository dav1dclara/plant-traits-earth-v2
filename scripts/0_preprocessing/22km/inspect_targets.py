"""
Inspect target rasters from GBIF and SPLOT sources.

Prints count and percentage of valid (non-NaN) pixels for each source,
and saves a distribution plot (one subplot per band) to plots/<TRAIT>_dist.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# If True, use the original (un-normalized) traits from gbif_original / splot_original.
#   These files have a single band (mean only, no descriptions) and are named
#   <trait>_original.tif. Outputs are suffixed with _original.
# If False, use the normalized traits from gbif / splot (all bands, with descriptions).
ORIGINAL = True

BANDS = ["mean", "std", "median", "q05", "q95"]
ALL_BANDS = BANDS + ["count"]

_suffix = "_original" if ORIGINAL else ""
_gbif_dir = "gbif_original" if ORIGINAL else "gbif"
_splot_dir = "splot_original" if ORIGINAL else "splot"

GBIF_DIR = Path(f"/scratch3/plant-traits-v2/data/22km/{_gbif_dir}")
SPLOT_DIR = Path(f"/scratch3/plant-traits-v2/data/22km/{_splot_dir}")
PLOTS_DIR = Path(__file__).parent / "plots"
DIFF_DIR = Path(f"/scratch3/plant-traits-v2/data/22km/splot_gbif_diff")


def read_all_bands(
    path: Path, descriptions: list[str] | None = None
) -> tuple[dict[str, np.ndarray], rasterio.transform.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as src:
        descs = descriptions if descriptions is not None else list(src.descriptions)
        bands = {
            name: src.read(descs.index(name) + 1, masked=False).astype(float)
            for name in ALL_BANDS
        }
        transform = src.transform
        crs = src.crs
    return bands, transform, crs


def save_diff(
    trait: str,
    gbif_bands: dict[str, np.ndarray],
    splot_bands: dict[str, np.ndarray],
    splot_mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: rasterio.crs.CRS,
) -> None:
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    # SPLOT - GBIF where both valid, NaN elsewhere
    diff_bands = {
        b: np.where(splot_mask == 1, splot_bands[b] - gbif_bands[b], np.nan)
        for b in BANDS
    }

    out_path = DIFF_DIR / f"{trait}{_suffix}_diff.tif"
    height, width = next(iter(diff_bands.values())).shape

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(BANDS),
        dtype="float64",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        for i, band in enumerate(BANDS, start=1):
            dst.write(diff_bands[band], i)
        dst.descriptions = tuple(BANDS)

    print(f"Diff raster saved to {out_path}")


def count_valid(data: np.ndarray) -> tuple[int, int]:
    total = data.size
    valid = int(np.sum(~np.isnan(data)))
    return valid, total


def report(source: str, data: np.ndarray) -> None:
    valid, total = count_valid(data)
    pct = 100 * valid / total
    print(f"{source:>15}  valid pixels: {valid:>8,} / {total:,}  ({pct:.2f}%)")


def line_hist(
    ax: plt.Axes,
    datasets: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    bins: int = 50,
) -> None:
    edges = np.histogram_bin_edges(np.concatenate(datasets), bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    for vals, label, color in zip(datasets, labels, colors):
        counts, _ = np.histogram(vals, bins=edges, density=True)
        ax.plot(bin_centers, counts, label=label, color=color, linewidth=1.5)


def plot_distributions(
    trait: str,
    gbif_bands: dict[str, np.ndarray],
    splot_bands: dict[str, np.ndarray],
    gbif_masked: dict[str, np.ndarray],
    splot_masked: dict[str, np.ndarray],
    combined_bands: dict[str, np.ndarray],
    n_valid: int,
    pct_valid: float,
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "monospace"

    n_bands = len(BANDS)
    fig, axes = plt.subplots(
        2, n_bands, figsize=(4 * n_bands, 8), sharey=False, squeeze=False
    )
    fig.suptitle(trait, fontsize=13, fontweight="bold")

    # Row 1: co-located pixels (both GBIF and SPLOT valid)
    for ax, band in zip(axes[0], BANDS):
        gbif_vals = gbif_masked[band][~np.isnan(gbif_masked[band])].ravel()
        splot_vals = splot_masked[band][~np.isnan(splot_masked[band])].ravel()

        line_hist(
            ax, [gbif_vals, splot_vals], ["GBIF", "SPLOT"], ["steelblue", "darkorange"]
        )
        ax.set_title(
            f"{band}\nco-located (n={n_valid:,}, {pct_valid:.2f}%)", fontsize=8
        )
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=7)

    # Row 2: full distributions + combined
    for ax, band in zip(axes[1], BANDS):
        gbif_vals = gbif_bands[band][~np.isnan(gbif_bands[band])].ravel()
        splot_vals = splot_bands[band][~np.isnan(splot_bands[band])].ravel()
        combined_vals = combined_bands[band][~np.isnan(combined_bands[band])].ravel()

        line_hist(
            ax,
            [gbif_vals, splot_vals, combined_vals],
            [
                f"GBIF (n={len(gbif_vals):,})",
                f"SPLOT (n={len(splot_vals):,})",
                f"Combined (n={len(combined_vals):,})",
            ],
            ["steelblue", "darkorange", "seagreen"],
        )
        ax.set_title(f"{band}\nall pixels", fontsize=8)
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=7)

    fig.tight_layout()

    out_path = PLOTS_DIR / f"{trait}{_suffix}_dist.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


def process_trait(trait: str) -> None:
    print(f"Trait: {trait}")

    # For original files, descriptions are missing — read them from the normalized counterparts
    norm_gbif_dir = Path("/scratch3/plant-traits-v2/data/22km/gbif")
    norm_splot_dir = Path("/scratch3/plant-traits-v2/data/22km/splot")

    def get_descriptions(norm_dir: Path) -> list[str]:
        with rasterio.open(norm_dir / f"{trait}.tif") as src:
            return list(src.descriptions)

    gbif_descs = get_descriptions(norm_gbif_dir) if ORIGINAL else None
    splot_descs = get_descriptions(norm_splot_dir) if ORIGINAL else None

    gbif_bands, gbif_transform, _ = read_all_bands(
        GBIF_DIR / f"{trait}{_suffix}.tif", gbif_descs
    )
    splot_bands, splot_transform, crs = read_all_bands(
        SPLOT_DIR / f"{trait}{_suffix}.tif", splot_descs
    )

    assert gbif_transform == splot_transform, (
        f"Transform mismatch:\n  GBIF:  {gbif_transform}\n  SPLOT: {splot_transform}"
    )

    splot_mask = (
        ~np.isnan(splot_bands["mean"]) & ~np.isnan(gbif_bands["mean"])
    ).astype(np.uint8)
    gbif_masked = {
        b: np.where(splot_mask == 1, gbif_bands[b], np.nan) for b in ALL_BANDS
    }
    splot_masked = {
        b: np.where(splot_mask == 1, splot_bands[b], np.nan) for b in ALL_BANDS
    }

    combined_bands = {
        b: np.where(~np.isnan(splot_bands[b]), splot_bands[b], gbif_bands[b])
        for b in ALL_BANDS
    }

    report("GBIF (all)", gbif_bands["mean"])
    report("GBIF (masked)", gbif_masked["mean"])
    report("SPLOT (masked)", splot_masked["mean"])

    n_valid = int(splot_mask.sum())
    pct_valid = 100 * n_valid / splot_mask.size
    plot_distributions(
        trait,
        gbif_bands,
        splot_bands,
        gbif_masked,
        splot_masked,
        combined_bands,
        n_valid,
        pct_valid,
    )
    save_diff(trait, gbif_bands, splot_bands, splot_mask, splot_transform, crs)
    print()


def main() -> None:
    # Trait names are the file stems stripped of the optional _original suffix
    traits = sorted(
        p.stem.removesuffix("_original")
        for p in GBIF_DIR.glob("*.tif")
        if (SPLOT_DIR / p.name).exists()
    )
    print(f"Processing {len(traits)} traits...\n")
    for trait in traits:
        process_trait(trait)


if __name__ == "__main__":
    main()
