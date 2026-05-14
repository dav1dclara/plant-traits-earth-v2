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

    # Full global coverage (5° tiles, land mask pre-filtering):
    python download_glo30_gee.py --region global

    # Full global coverage with larger tiles (15°, faster but less reliable):
    python download_glo30_gee.py --region global --grid-size 15

    # Custom bbox:
    python download_glo30_gee.py --bbox 5 45 11 48

    # Custom land mask (default: data/1km/predictors_new/worldcover/worldcover_land_mask.tif):
    python download_glo30_gee.py --region global --land-mask path/to/worldcover_land_mask.tif

Requires: earthengine-api, geemap, rasterio, numpy
"""

import argparse
import contextlib
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        .unmask(-9999)
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
    # Avoid passing exactly -180 to BBox: GEE treats -180≡+180 and interprets
    # BBox(+180, ..., -165, ...) as crossing the antimeridian, producing wrong output.
    bbox_lon_min = max(lon_min, -179.9999)
    region = ee.Geometry.BBox(bbox_lon_min, lat_min, lon_max, lat_max)
    _, _, crs_transform = snap_to_grid(lon_min, lat_min, lon_max, lat_max)

    for attempt in range(3):
        try:
            geemap.ee_export_image(
                dem,
                filename=str(output),
                crs="EPSG:4326",
                crs_transform=crs_transform,
                region=region,
                file_per_band=False,
                verbose=False,
            )
        except Exception:
            pass
        if output.exists():
            break
        if attempt < 2:
            time.sleep(5 * (attempt + 1))

    if not output.exists():
        return

    # Normalize nodata: geemap may write any sentinel; replace with NaN.
    # Discard tiles that are entirely nodata (outside GLO-30 coverage).
    tmp = output.with_suffix(".tmp.tif")
    with rasterio.open(output) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        data[data == -9999] = np.nan
        if np.all(np.isnan(data)):
            output.unlink()
            return
        # GEE canonicalizes lon=-180 → +180; fix so the merge places the tile correctly.
        transform = src.transform
        if abs(transform.c - 180.0) < 1e-6 and lon_min < -179.9:
            transform = Affine(
                transform.a,
                transform.b,
                transform.c - 360.0,
                transform.d,
                transform.e,
                transform.f,
            )
        profile = src.profile | {
            "nodata": np.nan,
            "dtype": "float32",
            "transform": transform,
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data[np.newaxis])
    tmp.replace(output)


def filter_cells_by_land_mask(cells, grid, land_mask_path: Path):
    """Return only cells that overlap at least one valid WorldCover pixel (0 or 1)."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    valid = []
    with rasterio.open(land_mask_path) as src:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        with progress:
            task = progress.add_task("Filtering cells", total=len(cells))
            for lat_s, lon_w in cells:
                lat_n = min(90, lat_s + grid)
                lon_e = min(180, lon_w + grid)
                r0, c0 = src.index(lon_w, lat_n)
                r1, c1 = src.index(lon_e, lat_s)
                r0, c0 = max(0, r0), max(0, c0)
                r1, c1 = min(src.height, r1), min(src.width, c1)
                if r1 > r0 and c1 > c0:
                    window = rasterio.windows.Window.from_slices((r0, r1), (c0, c1))
                    data = src.read(1, window=window)
                    if np.any(data != 255):
                        valid.append((lat_s, lon_w))
                else:
                    valid.append((lat_s, lon_w))  # outside WorldCover extent, keep
                progress.advance(task)
    return valid


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
        default=5,
        help="Grid cell size in degrees for global mode (default: 5)",
    )
    parser.add_argument(
        "--land-mask",
        type=Path,
        default=Path("data/1km/predictors_new/worldcover/worldcover_land_mask.tif"),
        help="WorldCover land mask for pre-filtering ocean-only tiles",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
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

        if args.land_mask.exists():
            cells = filter_cells_by_land_mask(cells, grid, args.land_mask)
            print(f"After land mask filter: {len(cells)} cells to download")
        else:
            print(f"Land mask not found, downloading all {len(cells)} cells")

        already_done = len(cells) - sum(
            1
            for lat_s, lon_w in cells
            if not (
                out_dir
                / f"copernicus_glo30_{'N' if lat_s >= 0 else 'S'}{abs(lat_s):02d}_{'E' if lon_w >= 0 else 'W'}{abs(lon_w):03d}.tif"
            ).exists()
        )
        print(
            f"{already_done}/{len(cells)} tiles already downloaded, {len(cells) - already_done} remaining."
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
        pending = [
            (lat_s, lon_w)
            for lat_s, lon_w in cells
            if not (
                out_dir
                / f"copernicus_glo30_{'N' if lat_s >= 0 else 'S'}{abs(lat_s):02d}_{'E' if lon_w >= 0 else 'W'}{abs(lon_w):03d}.tif"
            ).exists()
        ]

        def _download(lat_s, lon_w):
            lat_n = min(90, lat_s + grid)
            lon_e = min(180, lon_w + grid)
            ns = "N" if lat_s >= 0 else "S"
            ew = "E" if lon_w >= 0 else "W"
            name = f"copernicus_glo30_{ns}{abs(lat_s):02d}_{ew}{abs(lon_w):03d}.tif"
            with contextlib.redirect_stdout(io.StringIO()):
                download_tile(dem, lon_w, lat_s, lon_e, lat_n, out_dir / name)

        with progress:
            task = progress.add_task("Downloading tiles", total=len(pending))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_download, lat_s, lon_w): (lat_s, lon_w)
                    for lat_s, lon_w in pending
                }
                for future in as_completed(futures):
                    future.result()
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
