"""
Download ESA WorldCover v200 as a binary land/water mask at ~1km resolution.

A pixel is 0 (water) only if ALL native 10m pixels within the 1km output pixel
are permanent water bodies (WorldCover class 80). Otherwise the pixel is 1 (land).
Pixels outside WorldCover coverage are nodata (255).

Tiles are downloaded in EPSG:4326 aligned to a fixed global reference grid so
that merging is trivial. Reprojection to the project reference grid (EPSG:6933)
is handled by a separate script.

Global reference grid (EPSG:4326):
    resolution : 15/1670 ≈ 0.008982° per pixel (~1 km at equator)
    origin     : (-180°, 90°)
    size       : 40080 × 20040 pixels (global)

Usage:
    # Test with Switzerland:
    python download_worldcover.py

    # Full global coverage:
    python download_worldcover.py --region global

Requires: earthengine-api, geemap, rasterio
"""

import argparse
from pathlib import Path

import ee
import geemap
import rasterio
from rasterio.transform import Affine

NODATA = 255

# Fixed global EPSG:4326 reference grid.
# 1670 pixels per 15° → tiles align perfectly with no overlap or gap.
PIXELS_PER_DEG_BLOCK = 1670
DEG_BLOCK = 15
GLOBAL_RES = DEG_BLOCK / PIXELS_PER_DEG_BLOCK  # ≈ 0.008982°
GLOBAL_TRANSFORM = Affine(GLOBAL_RES, 0, -180.0, 0, -GLOBAL_RES, 90.0)
GLOBAL_WIDTH = round(360.0 / GLOBAL_RES)  # 40080
GLOBAL_HEIGHT = round(180.0 / GLOBAL_RES)  # 20040

SWITZERLAND_BBOX = (5, 45, 11, 48)  # (lon_min, lat_min, lon_max, lat_max)
EUROPE_BBOX = (-25, 34, 45, 72)


def build_land_mask() -> ee.Image:
    """Load WorldCover v200 and aggregate to binary land/water mask.

    Pixel is 1 (land) unless ALL native 10m pixels are permanent water (class 80).
    Pixels outside WorldCover coverage get nodata (255).
    """
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    return (
        wc.neq(80)
        .setDefaultProjection(wc.projection())
        .reduceResolution(reducer=ee.Reducer.max(), bestEffort=True)
        .rename("land_mask")
        .uint8()
        .unmask(NODATA)
    )


def snap_to_grid(lon_min, lat_min, lon_max, lat_max):
    """Snap a bbox to the nearest pixels of the global reference grid.

    Returns (tile_left, tile_top, crs_transform_list) where crs_transform_list
    is the 6-element list for GEE's crs_transform parameter.
    """
    import math

    col = round((lon_min - GLOBAL_TRANSFORM.c) / GLOBAL_TRANSFORM.a)
    row = round((lat_max - GLOBAL_TRANSFORM.f) / GLOBAL_TRANSFORM.e)
    tile_left = GLOBAL_TRANSFORM.c + col * GLOBAL_TRANSFORM.a
    tile_top = GLOBAL_TRANSFORM.f + row * GLOBAL_TRANSFORM.e
    return tile_left, tile_top, [GLOBAL_RES, 0, tile_left, 0, -GLOBAL_RES, tile_top]


def download_tile(
    land_mask: ee.Image,
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

    geemap.ee_export_image(
        land_mask,
        filename=str(output),
        crs="EPSG:4326",
        crs_transform=crs_transform,
        region=region,
        file_per_band=False,
        verbose=False,
    )

    if not output.exists():
        return

    # geemap writes nodata=0; fix to 255 so water pixels (0) aren't treated as nodata.
    # If all pixels are nodata (outside WorldCover coverage), discard the file.
    tmp = output.with_suffix(".tmp.tif")
    with rasterio.open(output) as src:
        data = src.read()
        if not (data != NODATA).any():
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
        profile = src.profile | {"nodata": NODATA, "transform": transform}
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
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
    land_mask = build_land_mask()

    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = args.bbox
        ns = "N" if lat_min >= 0 else "S"
        ew = "E" if lon_min >= 0 else "W"
        output = args.output or Path(
            f"data/1km/predictors_new/worldcover/worldcover_land_mask_{ns}{abs(int(lat_min)):02d}_{ew}{abs(int(lon_min)):03d}.tif"
        )
        download_tile(land_mask, lon_min, lat_min, lon_max, lat_max, output)

    elif args.region == "global":
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        out_dir = args.output or Path("data/1km/predictors_new/worldcover")
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

        already_done = sum(
            1
            for lat_s, lon_w in cells
            if (
                out_dir
                / f"worldcover_land_mask_{'N' if lat_s >= 0 else 'S'}{abs(lat_s):02d}_{'E' if lon_w >= 0 else 'W'}{abs(lon_w):03d}.tif"
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
        with progress:
            task = progress.add_task(
                "Downloading tiles", total=len(cells) - already_done
            )
            for lat_s, lon_w in cells:
                lat_n = min(90, lat_s + grid)
                lon_e = min(180, lon_w + grid)
                ns = "N" if lat_s >= 0 else "S"
                ew = "E" if lon_w >= 0 else "W"
                name = f"worldcover_land_mask_{ns}{abs(lat_s):02d}_{ew}{abs(lon_w):03d}.tif"
                if (out_dir / name).exists():
                    continue
                download_tile(land_mask, lon_w, lat_s, lon_e, lat_n, out_dir / name)
                progress.advance(task)

    else:
        region = args.region or "switzerland"
        bbox = SWITZERLAND_BBOX if region == "switzerland" else EUROPE_BBOX
        output = args.output or Path(
            f"data/1km/predictors_new/worldcover/worldcover_land_mask_{region}.tif"
        )
        download_tile(land_mask, *bbox, output)


if __name__ == "__main__":
    main()
