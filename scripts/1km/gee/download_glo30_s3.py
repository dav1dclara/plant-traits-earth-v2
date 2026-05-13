"""
Build Copernicus DEM GLO-30 at ~1km resolution directly from AWS S3 COG tiles.
Tiles are streamed one by one via /vsicurl/ and reprojected directly into the
output file (EPSG:6933, 1km) using average resampling. Memory usage is constant
regardless of region size.

No GEE, no Drive, no intermediate files.

Usage:
    # Test with Switzerland:
    python download_glo30_s3.py

    # Full global coverage:
    python download_glo30_s3.py --global

Requires: rasterio, pyproj
"""

import argparse
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Project reference grid (EPSG:6933, matches existing 1km predictors)
REF_CRS = CRS.from_epsg(6933)
REF_TRANSFORM = Affine(
    1000.030543281014, 0, -17367530.445161372, 0, -1000.167580032292, 7342230.205017056
)
REF_WIDTH = 34734
REF_HEIGHT = 14682

BASE_URL = "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com"

SWITZERLAND_BBOX = (5, 45, 11, 48)  # (lon_min, lat_min, lon_max, lat_max)
EUROPE_BBOX = (-25, 34, 45, 72)


_TILE_RE = re.compile(r"Copernicus_DSM_COG_10_(N|S)(\d+)_00_(E|W)(\d+)_00_DEM/")


def available_tiles() -> set[tuple[int, int]]:
    """Return the set of (lat, lon) pairs that have a tile on S3."""
    tiles, token = set(), None
    while True:
        params = {
            "list-type": "2",
            "prefix": "Copernicus_DSM_COG_10_",
            "delimiter": "/",
            "max-keys": "1000",
        }
        if token:
            params["continuation-token"] = token
        r = requests.get(
            "https://copernicus-dem-30m.s3.amazonaws.com/", params=params, timeout=30
        )
        r.raise_for_status()
        for ns, lat, ew, lon in _TILE_RE.findall(r.text):
            tiles.add(
                (
                    (-1 if ns == "S" else 1) * int(lat),
                    (-1 if ew == "W" else 1) * int(lon),
                )
            )
        m = re.search(r"<NextContinuationToken>(.+?)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return tiles


def tile_url(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"{BASE_URL}/{name}/{name}.tif"


def bbox_to_ref_subgrid(lon_min, lat_min, lon_max, lat_max):
    """Return (transform, width, height) for the ref grid subregion covering the bbox."""
    t = Transformer.from_crs("EPSG:4326", REF_CRS, always_xy=True)
    xs, ys = t.transform(
        [lon_min, lon_min, lon_max, lon_max], [lat_min, lat_max, lat_min, lat_max]
    )
    col_min = max(0, math.floor((min(xs) - REF_TRANSFORM.c) / REF_TRANSFORM.a))
    col_max = min(REF_WIDTH, math.ceil((max(xs) - REF_TRANSFORM.c) / REF_TRANSFORM.a))
    row_min = max(0, math.floor((max(ys) - REF_TRANSFORM.f) / REF_TRANSFORM.e))
    row_max = min(REF_HEIGHT, math.ceil((min(ys) - REF_TRANSFORM.f) / REF_TRANSFORM.e))

    transform = Affine(
        REF_TRANSFORM.a,
        0,
        REF_TRANSFORM.c + col_min * REF_TRANSFORM.a,
        0,
        REF_TRANSFORM.e,
        REF_TRANSFORM.f + row_min * REF_TRANSFORM.e,
    )
    return transform, col_max - col_min, row_max - row_min, col_min, row_min


def build(
    lon_min,
    lat_min,
    lon_max,
    lat_max,
    output: Path,
    out_transform: Affine,
    out_width: int,
    out_height: int,
    col_offset: int,
    row_offset: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching tile index from S3...")
    valid = available_tiles()

    lats = range(math.floor(lat_min), math.ceil(lat_max))
    lons = range(math.floor(lon_min), math.ceil(lon_max))
    tiles = [(lat, lon) for lat in lats for lon in lons if (lat, lon) in valid]
    total = len(tiles)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    to_ref = Transformer.from_crs("EPSG:4326", REF_CRS, always_xy=True)
    write_lock = threading.Lock()

    # Compute output windows for all tiles
    def tile_window(lat, lon):
        xs, ys = to_ref.transform(
            [lon, lon, lon + 1, lon + 1], [lat, lat + 1, lat, lat + 1]
        )
        c0 = max(
            0, math.floor((min(xs) - REF_TRANSFORM.c) / REF_TRANSFORM.a) - col_offset
        )
        c1 = min(
            out_width,
            math.ceil((max(xs) - REF_TRANSFORM.c) / REF_TRANSFORM.a) - col_offset,
        )
        r0 = max(
            0, math.floor((max(ys) - REF_TRANSFORM.f) / REF_TRANSFORM.e) - row_offset
        )
        r1 = min(
            out_height,
            math.ceil((min(ys) - REF_TRANSFORM.f) / REF_TRANSFORM.e) - row_offset,
        )
        return c0, r0, c1 - c0, r1 - r0

    def process_tile(lat, lon, c0, r0, w, h):
        tile_transform = Affine(
            REF_TRANSFORM.a,
            0,
            REF_TRANSFORM.c + (c0 + col_offset) * REF_TRANSFORM.a,
            0,
            REF_TRANSFORM.e,
            REF_TRANSFORM.f + (r0 + row_offset) * REF_TRANSFORM.e,
        )
        src = rasterio.open(tile_url(lat, lon))
        if src.crs is None:
            src.close()
            return None
        dst_data = np.full((h, w), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tile_transform,
            dst_crs=REF_CRS,
            resampling=Resampling.average,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        src.close()
        return c0, r0, w, h, dst_data

    pending = [(lat, lon, *tile_window(lat, lon)) for lat, lon in tiles]
    pending = [
        (lat, lon, c0, r0, w, h)
        for lat, lon, c0, r0, w, h in pending
        if w > 0 and h > 0
    ]

    with (
        progress,
        rasterio.open(
            output,
            "w",
            driver="GTiff",
            height=out_height,
            width=out_width,
            count=1,
            dtype="float32",
            crs=REF_CRS,
            transform=out_transform,
            compress="deflate",
            tiled=True,
            nodata=np.nan,
        ) as dst,
    ):
        task = progress.add_task("Processing tiles", total=len(pending))
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {
                pool.submit(process_tile, lat, lon, c0, r0, w, h): (lat, lon)
                for lat, lon, c0, r0, w, h in pending
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    c0, r0, w, h, dst_data = result
                    with write_lock:
                        dst.write(
                            dst_data[np.newaxis],
                            window=rasterio.windows.Window(c0, r0, w, h),
                        )
                progress.advance(task)

    print(f"Written: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region", choices=["switzerland", "europe", "global"], default="switzerland"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.region == "global":
        bbox = (-180, -90, 180, 90)
        out_transform, out_width, out_height = REF_TRANSFORM, REF_WIDTH, REF_HEIGHT
        col_offset, row_offset = 0, 0
        output = args.output or Path(
            "data/1km/predictors_new/glo30/copernicus_glo30.tif"
        )
    else:
        bbox = SWITZERLAND_BBOX if args.region == "switzerland" else EUROPE_BBOX
        out_transform, out_width, out_height, col_offset, row_offset = (
            bbox_to_ref_subgrid(*bbox)
        )
        output = args.output or Path(
            f"data/1km/predictors_new/glo30/copernicus_glo30_{args.region}.tif"
        )

    build(*bbox, output, out_transform, out_width, out_height, col_offset, row_offset)


if __name__ == "__main__":
    main()
