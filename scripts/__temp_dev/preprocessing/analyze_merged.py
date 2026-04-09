#!/usr/bin/env python3
"""
Analysis script for merged multi-band trait TIFFs.

Each merged TIFF has 6 bands: mean, std, median, q05, q95, count.
This script loads a specified trait's merged file, computes statistics,
and generates visualizations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Band names
BAND_NAMES = ["mean", "std", "median", "q05", "q95", "count"]


def load_merged_tif(trait_path):
    """Load multi-band merged TIFF."""
    with rasterio.open(trait_path) as src:
        data = src.read()  # Shape: (6, H, W)
        profile = src.profile
    # Convert nodata to NaN
    nodata = profile.get("nodata", np.nan)
    if not np.isnan(nodata):
        data = data.astype(float)
        data[data == nodata] = np.nan
    return data, profile


def analyze_trait(trait):
    """Analyze and visualize a trait's merged data."""
    merged_dir = Path("/scratch3/plant-traits-v2/data/22km/merged")
    trait_path = merged_dir / f"{trait}.tif"

    if not trait_path.exists():
        print(f"Error: Merged file for trait '{trait}' not found at {trait_path}")
        return

    print(f"Analyzing trait: {trait}")
    print(f"File: {trait_path}")

    # Load data
    data, profile = load_merged_tif(trait_path)
    n_bands, h, w = data.shape
    print(f"Shape: {n_bands} bands x {h} x {w} pixels")

    # Statistics for each band
    print("\nBand Statistics:")
    print("Band     | Min      | Max      | Mean     | Std      | Valid Pixels")
    print("-" * 70)

    for i, band_name in enumerate(BAND_NAMES):
        band_data = data[i]
        valid = ~np.isnan(band_data)
        n_valid = np.sum(valid)
        if n_valid > 0:
            min_val = np.nanmin(band_data)
            max_val = np.nanmax(band_data)
            mean_val = np.nanmean(band_data)
            std_val = np.nanstd(band_data)
            print(
                f"{band_name:8} | {min_val:8.3f} | {max_val:8.3f} | {mean_val:8.3f} | {std_val:8.3f} | {n_valid:>12,}"
            )
        else:
            print(f"{band_name:8} | No valid data")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Merged Trait Analysis: {trait}", fontsize=16, fontweight="bold")

    for i, (ax, band_name) in enumerate(zip(axes.flat, BAND_NAMES)):
        band_data = data[i]

        # Plot the spatial map
        im = ax.imshow(band_data, cmap="RdYlGn", interpolation="nearest", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{band_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"merged_analysis_{trait}_spatial.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Value Distributions: {trait}", fontsize=16, fontweight="bold")

    for i, (ax, band_name) in enumerate(zip(axes.flat, BAND_NAMES)):
        band_data = data[i]
        valid_data = band_data[~np.isnan(band_data)].ravel()

        if len(valid_data) > 0:
            ax.hist(valid_data, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
            ax.set_title(
                f"{band_name.capitalize()}\n(n={len(valid_data):,})", fontsize=12
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{band_name.capitalize()}\n(No data)", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"merged_analysis_{trait}_histograms.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"\nSaved plots: merged_analysis_{trait}_spatial.png, merged_analysis_{trait}_histograms.png"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze merged multi-band trait TIFFs"
    )
    parser.add_argument("trait", help="Trait name (e.g., X1080)")
    args = parser.parse_args()

    analyze_trait(args.trait)


if __name__ == "__main__":
    main()
