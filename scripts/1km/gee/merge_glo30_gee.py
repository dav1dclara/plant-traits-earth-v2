"""
Merge all copernicus_glo30_*.tif tiles into a single global mosaic.

Tiles must be aligned to the global reference grid defined in download_glo30_gee.py
(EPSG:4326, ~0.008982°/pixel, origin at -180°/90°). Skips .tmp.tif files.
Existing valid pixels (non-NaN) take priority over nodata (NaN).

Usage:
    python merge_glo30_gee.py
    python merge_glo30_gee.py --input-dir data/1km/predictors_new/glo30_gee --output copernicus_glo30.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

GLOBAL_RES = 15.0 / 1670
GLOBAL_TRANSFORM = Affine(GLOBAL_RES, 0, -180.0, 0, -GLOBAL_RES, 90.0)
GLOBAL_WIDTH = round(360.0 / GLOBAL_RES)  # 40080
GLOBAL_HEIGHT = round(180.0 / GLOBAL_RES)  # 20040


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/1km/predictors_new/glo30_gee")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parts = sorted(
        p for p in args.input_dir.glob("copernicus_glo30_*.tif") if ".tmp" not in p.name
    )
    if not parts:
        print("No tile files found.")
        return

    col_mins, col_maxs, row_mins, row_maxs = [], [], [], []
    for p in parts:
        with rasterio.open(p) as src:
            c0 = round((src.transform.c - GLOBAL_TRANSFORM.c) / GLOBAL_TRANSFORM.a)
            r0 = round((src.transform.f - GLOBAL_TRANSFORM.f) / GLOBAL_TRANSFORM.e)
            col_mins.append(c0)
            row_mins.append(r0)
            col_maxs.append(c0 + src.width)
            row_maxs.append(r0 + src.height)

    out_col0 = max(0, min(col_mins))
    out_row0 = max(0, min(row_mins))
    out_width = min(GLOBAL_WIDTH, max(col_maxs)) - out_col0
    out_height = min(GLOBAL_HEIGHT, max(row_maxs)) - out_row0
    out_transform = Affine(
        GLOBAL_RES,
        0,
        GLOBAL_TRANSFORM.c + out_col0 * GLOBAL_RES,
        0,
        -GLOBAL_RES,
        GLOBAL_TRANSFORM.f - out_row0 * GLOBAL_RES,
    )

    output = args.output or args.input_dir / "copernicus_glo30.tif"
    print(f"Merging {len(parts)} tiles → {output}")
    print(
        f"Output: {out_width}×{out_height} px, EPSG:4326, pixel size {GLOBAL_RES:.6f}°"
    )

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=out_transform,
        compress="deflate",
        tiled=True,
        nodata=np.nan,
    ) as dst:
        dst.write(np.full((1, out_height, out_width), np.nan, dtype=np.float32))

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
                c0 = round((src.transform.c - GLOBAL_TRANSFORM.c) / GLOBAL_TRANSFORM.a)
                r0 = round((src.transform.f - GLOBAL_TRANSFORM.f) / GLOBAL_TRANSFORM.e)

            col_off = c0 - out_col0
            row_off = r0 - out_row0
            col_start = max(0, col_off)
            row_start = max(0, row_off)
            col_end = min(out_width, col_off + data.shape[1])
            row_end = min(out_height, row_off + data.shape[0])
            dc0 = col_start - col_off
            dr0 = row_start - row_off
            w = col_end - col_start
            h = row_end - row_start

            window = rasterio.windows.Window(col_start, row_start, w, h)
            existing = dst.read(1, window=window)
            patch = data[dr0 : dr0 + h, dc0 : dc0 + w]
            valid = ~np.isnan(patch)
            existing[valid] = patch[valid]
            dst.write(existing[np.newaxis], window=window)
            progress.advance(task)

    print(f"Written: {output}")


if __name__ == "__main__":
    main()
