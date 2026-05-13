"""
Merge all copernicus_glo30_*.tif grid cells into a single copernicus_glo30.tif.
Skips .tmp.tif files (incomplete). Non-NaN values from each part are written
into the output in-place; existing pixels are not overwritten.

Usage:
    python merge_glo30.py
    python merge_glo30.py --input-dir data/1km/predictors_new/glo30 --output copernicus_glo30.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

REF_CRS = CRS.from_epsg(6933)
REF_TRANSFORM = Affine(
    1000.030543281014, 0, -17367530.445161372, 0, -1000.167580032292, 7342230.205017056
)
REF_WIDTH = 34734
REF_HEIGHT = 14682


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/1km/predictors_new/glo30")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parts = sorted(
        p for p in args.input_dir.glob("copernicus_glo30_*.tif") if ".tmp" not in p.name
    )
    if not parts:
        print("No part files found.")
        return

    output = args.output or args.input_dir / "copernicus_glo30.tif"
    print(f"Merging {len(parts)} files → {output}")

    # Create empty output file
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=REF_HEIGHT,
        width=REF_WIDTH,
        count=1,
        dtype="float32",
        crs=REF_CRS,
        transform=REF_TRANSFORM,
        compress="deflate",
        tiled=True,
        nodata=np.nan,
    ):
        pass

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
    with progress, rasterio.open(output, "r+") as dst:
        task = progress.add_task("Merging", total=len(parts))
        for part in parts:
            with rasterio.open(part) as src:
                data = src.read(1)
                mask = ~np.isnan(data)
                if not mask.any():
                    progress.advance(task)
                    continue
                col_off = round((src.transform.c - REF_TRANSFORM.c) / REF_TRANSFORM.a)
                row_off = round((src.transform.f - REF_TRANSFORM.f) / REF_TRANSFORM.e)
                window = rasterio.windows.Window(
                    col_off, row_off, src.width, src.height
                )
                existing = dst.read(1, window=window)
                existing[mask] = data[mask]
                dst.write(existing[np.newaxis], window=window)
            progress.advance(task)

    print(f"Written: {output}")


if __name__ == "__main__":
    main()
