"""
Merge all satellite_embedding_*.tif tiles into a single mosaic.

Tiles must be aligned to the global reference grid defined in download_embeddings.py
(EPSG:4326, ~0.008982°/pixel, origin at -180°/90°). Skips .tmp.tif files.
Existing valid pixels (non-NaN) take priority over nodata (NaN).

The output covers only the bounding box of the available tiles. NaN fill is
written in row chunks so memory usage stays bounded regardless of output size.

Usage:
    python merge_embeddings.py
    python merge_embeddings.py --year 2023
    python merge_embeddings.py --input-dir data/1km/predictors_new/embeddings/tiles_2024 --output merged.tif

Requires: rasterio, numpy, rich
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

# Rows written per chunk during NaN initialisation (bounds peak memory use)
INIT_CHUNK_ROWS = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing tile .tif files (default: data/1km/predictors_new/embeddings/tiles_{year})",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir or Path(
        f"data/1km/predictors_new/embeddings/tiles_{args.year}"
    )
    parts = sorted(
        p for p in input_dir.glob("satellite_embedding_*.tif") if ".tmp" not in p.name
    )
    if not parts:
        print(f"No tile files found in {input_dir}")
        return

    # ── Compute output bounding box from tile extents ──────────────────────────
    col_mins, col_maxs, row_mins, row_maxs = [], [], [], []
    n_bands = None
    for p in parts:
        with rasterio.open(p) as src:
            c0 = round((src.transform.c - GLOBAL_TRANSFORM.c) / GLOBAL_TRANSFORM.a)
            r0 = round((src.transform.f - GLOBAL_TRANSFORM.f) / GLOBAL_TRANSFORM.e)
            col_mins.append(c0)
            row_mins.append(r0)
            col_maxs.append(c0 + src.width)
            row_maxs.append(r0 + src.height)
            if n_bands is None:
                n_bands = src.count

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

    output = args.output or input_dir.parent / f"satellite_embedding_{args.year}.tif"
    print(f"Merging {len(parts)} tiles → {output}")
    print(
        f"Output: {out_width}×{out_height} px, {n_bands} bands, "
        f"EPSG:4326, pixel size {GLOBAL_RES:.6f}°"
    )

    # ── Create output file and initialise to NaN in row chunks ────────────────
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=n_bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=out_transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=np.nan,
    ) as dst:
        nan_chunk = np.full(
            (n_bands, INIT_CHUNK_ROWS, out_width), np.nan, dtype=np.float32
        )
        for row_start in range(0, out_height, INIT_CHUNK_ROWS):
            h = min(INIT_CHUNK_ROWS, out_height - row_start)
            window = rasterio.windows.Window(0, row_start, out_width, h)
            dst.write(nan_chunk[:, :h, :], window=window)

    # ── Patch in each tile ────────────────────────────────────────────────────
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
                data = src.read().astype(np.float32)  # (bands, h, w)
                c0 = round((src.transform.c - GLOBAL_TRANSFORM.c) / GLOBAL_TRANSFORM.a)
                r0 = round((src.transform.f - GLOBAL_TRANSFORM.f) / GLOBAL_TRANSFORM.e)

            col_off = c0 - out_col0
            row_off = r0 - out_row0
            col_start = max(0, col_off)
            row_start = max(0, row_off)
            col_end = min(out_width, col_off + data.shape[2])
            row_end = min(out_height, row_off + data.shape[1])
            dc0 = col_start - col_off
            dr0 = row_start - row_off
            w = col_end - col_start
            h = row_end - row_start

            window = rasterio.windows.Window(col_start, row_start, w, h)
            existing = dst.read(window=window).astype(np.float32)  # (bands, h, w)
            patch = data[:, dr0 : dr0 + h, dc0 : dc0 + w]
            valid = ~np.isnan(patch)
            existing[valid] = patch[valid]
            dst.write(existing, window=window)
            progress.advance(task)

    print(f"Written: {output}")


if __name__ == "__main__":
    main()
