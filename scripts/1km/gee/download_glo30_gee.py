"""
Download Copernicus DEM GLO-30 at ~1km resolution from Google Earth Engine.

Tiles are downloaded in EPSG:4326 aligned to a fixed global reference grid so
that merging is trivial. Average resampling is used to aggregate from ~30m to ~1km.
Pixels outside GLO-30 coverage are nodata (NaN).

Global reference grid (EPSG:4326):
    resolution : 15/1670 ≈ 0.008982° per pixel (~1 km at equator)
    origin     : (-180°, 90°)
    size       : 40080 × 20040 pixels (global)

Usage:
    # Test with Switzerland:
    python download_glo30_gee.py

    # Full global coverage:
    python download_glo30_gee.py --region global

Requires: earthengine-api, geemap, rasterio, numpy
"""

import argparse
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import Affine

# Fixed global EPSG:4326 reference grid (same as WorldCover download script)
PIXELS_PER_DEG_BLOCK = 1670
DEG_BLOCK = 15
GLOBAL_RES = DEG_BLOCK / PIXELS_PER_DEG_BLOCK  # ≈ 0.008982°
GLOBAL_TRANSFORM = Affine(GLOBAL_RES, 0, -180.0, 0, -GLOBAL_RES, 90.0)
GLOBAL_WIDTH = round(360.0 / GLOBAL_RES)  # 40080
GLOBAL_HEIGHT = round(180.0 / GLOBAL_RES)  # 20040

SWITZERLAND_BBOX = (5, 45, 11, 48)  # (lon_min, lat_min, lon_max, lat_max)
EUROPE_BBOX = (-25, 34, 45, 72)


def build_dem() -> ee.Image:
    """Load GLO-30 and aggregate to ~1km using average resampling."""
    collection = ee.ImageCollection("COPERNICUS/DEM/GLO30")
    dem = collection.select("DEM").mosaic()
    proj = collection.first().select("DEM").projection()
    return (
        dem.setDefaultProjection(proj)
        .reduceResolution(reducer=ee.Reducer.mean(), bestEffort=True)
        .float()
    )


def snap_to_grid(lon_min, lat_min, lon_max, lat_max):
    """Snap a bbox to the nearest pixels of the global reference grid.

    Returns (tile_left, tile_top, crs_transform_list) for GEE's crs_transform.
    """
    col = round((lon_min - GLOBAL_TRANSFORM.c) / GLOBAL_TRANSFORM.a)
    row = round((lat_max - GLOBAL_TRANSFORM.f) / GLOBAL_TRANSFORM.e)
    tile_left = GLOBAL_TRANSFORM.c + col * GLOBAL_TRANSFORM.a
    tile_top = GLOBAL_TRANSFORM.f + row * GLOBAL_TRANSFORM.e
    return tile_left, tile_top, [GLOBAL_RES, 0, tile_left, 0, -GLOBAL_RES, tile_top]


def download_tile(
    dem: ee.Image,
    lon_min,
    lat_min,
    lon_max,
    lat_max,
    output: Path,
) -> None:
    if output.exists():
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    region = ee.Geometry.BBox(lon_min, lat_min, lon_max, lat_max)
    _, _, crs_transform = snap_to_grid(lon_min, lat_min, lon_max, lat_max)

    geemap.ee_export_image(
        dem,
        filename=str(output),
        crs="EPSG:4326",
        crs_transform=crs_transform,
        region=region,
        file_per_band=False,
        verbose=False,
    )

    if not output.exists():
        return

    # Normalize nodata: geemap may write any sentinel; replace with NaN.
    # Discard tiles that are entirely nodata (outside GLO-30 coverage).
    tmp = output.with_suffix(".tmp.tif")
    with rasterio.open(output) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        if np.all(np.isnan(data)):
            output.unlink()
            return
        profile = src.profile | {"nodata": np.nan, "dtype": "float32"}
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data[np.newaxis])
    tmp.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["switzerland", "europe", "global"])
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEG_BLOCK,
        help=f"Grid cell size in degrees for global mode (default: {DEG_BLOCK})",
    )
    args = parser.parse_args()

    ee.Initialize(project="plant-traits-earth-v2")
    dem = build_dem()

    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = args.bbox
        ns = "N" if lat_min >= 0 else "S"
        ew = "E" if lon_min >= 0 else "W"
        output = args.output or Path(
            f"data/1km/predictors_new/glo30_gee/copernicus_glo30_{ns}{abs(int(lat_min)):02d}_{ew}{abs(int(lon_min)):03d}.tif"
        )
        download_tile(dem, lon_min, lat_min, lon_max, lat_max, output)

    elif args.region == "global":
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        out_dir = args.output or Path("data/1km/predictors_new/glo30_gee")
        out_dir.mkdir(parents=True, exist_ok=True)
        grid = args.grid_size

        cells = sorted(
            [
                (lat_s, lon_w)
                for lat_s in range(-90, 90, grid)
                for lon_w in range(-180, 180, grid)
            ],
            key=lambda c: c[0] ** 2 + c[1] ** 2,
        )

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task("Downloading tiles", total=len(cells))
            for lat_s, lon_w in cells:
                lat_n = min(90, lat_s + grid)
                lon_e = min(180, lon_w + grid)
                ns = "N" if lat_s >= 0 else "S"
                ew = "E" if lon_w >= 0 else "W"
                name = f"copernicus_glo30_{ns}{abs(lat_s):02d}_{ew}{abs(lon_w):03d}.tif"
                download_tile(dem, lon_w, lat_s, lon_e, lat_n, out_dir / name)
                progress.advance(task)

    else:
        region = args.region or "switzerland"
        bbox = SWITZERLAND_BBOX if region == "switzerland" else EUROPE_BBOX
        output = args.output or Path(
            f"data/1km/predictors_new/glo30_gee/copernicus_glo30_{region}.tif"
        )
        download_tile(dem, *bbox, output)


if __name__ == "__main__":
    main()
