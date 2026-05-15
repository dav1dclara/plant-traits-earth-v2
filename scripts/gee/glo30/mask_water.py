"""
Mask permanent water bodies out of the Copernicus GLO-30 DEM using the
ESA WorldCover land mask.

Both inputs must already be in the same CRS and on the same pixel grid (reproject
each with the corresponding reproject_to_6933.py script first). Pixels where the
land mask is 0 (water) or 255 (nodata) are set to NaN in the output.

The script verifies grid alignment and aborts with a clear error if it fails.

Usage:
    python mask_water.py
    python mask_water.py \
        --dem data/1km/predictors_new/glo30/copernicus_glo30_6933.tif \
        --land-mask data/1km/predictors_new/worldcover/worldcover_land_mask_6933.tif \
        --output data/1km/predictors_new/glo30/copernicus_glo30_6933_masked.tif

Requires: rasterio, numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DEM_DEFAULT = Path("data/1km/predictors_new/glo30/copernicus_glo30_6933.tif")
MASK_DEFAULT = Path("data/1km/predictors_new/worldcover/worldcover_land_mask_6933.tif")


def check_alignment(dem_ds, mask_ds):
    """Assert both datasets share CRS, pixel size, and grid alignment.

    Returns (col_off, row_off): DEM pixel (0, 0) expressed in mask pixel coordinates.
    Raises ValueError with a descriptive message on any mismatch.
    """
    if dem_ds.crs != mask_ds.crs:
        raise ValueError(f"CRS mismatch:\n  DEM:  {dem_ds.crs}\n  mask: {mask_ds.crs}")

    dt, mt = dem_ds.transform, mask_ds.transform

    if not (abs(dt.a - mt.a) < 1e-9 and abs(dt.e - mt.e) < 1e-9):
        raise ValueError(
            f"Pixel size mismatch:\n"
            f"  DEM:  dx={dt.a:.10f}  dy={dt.e:.10f}\n"
            f"  mask: dx={mt.a:.10f}  dy={mt.e:.10f}"
        )

    # Origin offset in fractional pixels — must be integers for the grids to align.
    dx = (dt.c - mt.c) / dt.a
    dy = (dt.f - mt.f) / dt.e
    if abs(dx - round(dx)) > 1e-3 or abs(dy - round(dy)) > 1e-3:
        raise ValueError(
            f"Pixel misalignment: DEM origin is ({dx:.6f}, {dy:.6f}) pixels from "
            f"mask origin — must be exact integers."
        )

    return round(dx), round(dy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, default=DEM_DEFAULT)
    parser.add_argument("--land-mask", type=Path, default=MASK_DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.dem.parent / (args.dem.stem + "_masked.tif")

    with rasterio.open(args.dem) as dem_ds, rasterio.open(args.land_mask) as mask_ds:
        try:
            col_off, row_off = check_alignment(dem_ds, mask_ds)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Grid alignment OK")
        print(f"  CRS:        {dem_ds.crs}")
        print(f"  Pixel size: {dem_ds.transform.a:.10f}°")
        print(f"  DEM origin is ({col_off:+d}, {row_off:+d}) px from mask origin")

        dem_data = dem_ds.read(1).astype(np.float32)

        # Find the mask window that corresponds to the DEM extent.
        mask_col0 = col_off
        mask_row0 = row_off
        mask_col1 = col_off + dem_ds.width
        mask_row1 = row_off + dem_ds.height

        # Clip to the mask's actual extent (both files may not cover the full globe).
        rc0 = max(0, mask_col0)
        rr0 = max(0, mask_row0)
        rc1 = min(mask_ds.width, mask_col1)
        rr1 = min(mask_ds.height, mask_row1)

        if rc1 <= rc0 or rr1 <= rr0:
            print("WARNING: DEM and land mask do not overlap — no masking applied.")
        else:
            window = Window(rc0, rr0, rc1 - rc0, rr1 - rr0)
            land_mask = mask_ds.read(1, window=window)

            # Corresponding slice in the DEM array.
            dc0 = rc0 - mask_col0
            dr0 = rr0 - mask_row0
            patch = dem_data[
                dr0 : dr0 + land_mask.shape[0], dc0 : dc0 + land_mask.shape[1]
            ]

            water_px = int((land_mask == 0).sum())
            nodata_px = int((land_mask == 255).sum())
            patch[(land_mask == 0) | (land_mask == 255)] = np.nan
            print(
                f"  Masked {water_px:,} water pixels and {nodata_px:,} nodata pixels → NaN"
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        profile = dem_ds.profile | {"dtype": "float32", "nodata": np.nan}
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(dem_data[np.newaxis])

    print(f"Written: {output}")


if __name__ == "__main__":
    main()
