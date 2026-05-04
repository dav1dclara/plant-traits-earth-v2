"""
Reproject a GEE-exported embedding GeoTIFF (EPSG:4326) to the project reference
grid (EPSG:6933, 1km, aligned to existing predictors).

Usage:
    python reproject_to_reference.py <input.tif> [--output <output.tif>]
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

# Reference grid from existing 1km predictors (e.g. ETH canopy height)
REFERENCE_CRS = CRS.from_epsg(6933)
REFERENCE_TRANSFORM = Affine(
    1000.030543281014,
    0,
    -17367530.445161372,
    0,
    -1000.167580032292,
    7342230.205017056,
)
REFERENCE_SHAPE = (14682, 34734)  # (height, width)


def reproject_embedding(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        n_bands = src.count
        dst_data = np.zeros((n_bands, *REFERENCE_SHAPE), dtype=np.float32)

        for i in range(1, n_bands + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=dst_data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=REFERENCE_TRANSFORM,
                dst_crs=REFERENCE_CRS,
                resampling=Resampling.bilinear,
            )

        profile = src.profile.copy()
        profile.update(
            crs=REFERENCE_CRS,
            transform=REFERENCE_TRANSFORM,
            width=REFERENCE_SHAPE[1],
            height=REFERENCE_SHAPE[0],
            dtype="float32",
            driver="GTiff",
            tiled=True,
            compress="deflate",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dst_data)

    print(f"Written: {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.input.with_name(
        args.input.stem.replace("_wgs84", "") + "_6933.tif"
    )
    reproject_embedding(args.input, output)


if __name__ == "__main__":
    main()
