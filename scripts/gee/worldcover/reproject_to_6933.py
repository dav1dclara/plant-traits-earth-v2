"""
Reproject the ESA WorldCover land mask to the exact grid of a reference raster
using nearest-neighbour resampling (appropriate for a categorical mask).

The output will have the same CRS, transform, width, and height as the reference
file. Nodata (255) is preserved.

Usage:
    python reproject_to_6933.py
    python reproject_to_6933.py \
        --input data/1km/predictors_new/worldcover/worldcover_land_mask.tif \
        --reference data/1km/targets/gbif/X4.tif \
        --output data/1km/predictors_new/worldcover/worldcover_land_mask_6933.tif

Requires: rasterio, numpy
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

INPUT_DEFAULT = Path("data/1km/predictors_new/worldcover/worldcover_land_mask.tif")
REFERENCE_DEFAULT = Path("data/1km/targets/gbif/X4.tif")
NODATA = 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.input.parent / (args.input.stem + "_6933.tif")

    with rasterio.open(args.reference) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    print(f"Reference grid:")
    print(f"  CRS:       {dst_crs}")
    print(f"  Transform: {dst_transform}")
    print(f"  Size:      {dst_width} x {dst_height}")

    with rasterio.open(args.input) as src:
        dst_data = np.full((dst_height, dst_width), NODATA, dtype=np.uint8)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=NODATA,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=NODATA,
            resampling=Resampling.nearest,
        )

        profile = src.profile | {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "dtype": "uint8",
            "nodata": NODATA,
            "compress": "deflate",
            "tiled": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(dst_data[np.newaxis])

    print(f"Written: {output}")


if __name__ == "__main__":
    main()
