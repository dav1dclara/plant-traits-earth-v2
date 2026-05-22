"""
Download Google AlphaEarth Foundations Satellite Embeddings at ~1km resolution
from Google Earth Engine.

Tiles are downloaded in EPSG:4326 aligned to a fixed global reference grid so
that merging is trivial. Average resampling is used to aggregate from 10m to ~1km.
The output has 64 float32 bands (A00–A63) representing the embedding vector.

Global reference grid (EPSG:4326):
    resolution : 15/1670 ≈ 0.008982° per pixel (~1 km at equator)
    origin     : (-180°, 90°)
    size       : 40080 × 20040 pixels (global)

Tiles are written to a tiles_{year}/ subfolder; named-region outputs are merged
into a single file alongside that folder.

Usage:
    # Test with a small region in Switzerland (default):
    python download_embeddings.py

    # Specific year:
    python download_embeddings.py --year 2023

    # Full global coverage (5° tiles, land mask pre-filtering):
    python download_embeddings.py --region global

    # Custom bbox:
    python download_embeddings.py --bbox 7 46 8 47

    # Custom land mask (default: data/1km/predictors_new/worldcover/worldcover_land_mask.tif):
    python download_embeddings.py --region global --land-mask path/to/worldcover_land_mask.tif

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

# Fixed global EPSG:4326 reference grid (same as GLO-30 and WorldCover scripts)
PIXELS_PER_DEG_BLOCK = 1670
DEG_BLOCK = 15
GLOBAL_RES = DEG_BLOCK / PIXELS_PER_DEG_BLOCK  # ≈ 0.008982°
GLOBAL_TRANSFORM = Affine(GLOBAL_RES, 0, -180.0, 0, -GLOBAL_RES, 90.0)
GLOBAL_WIDTH = round(360.0 / GLOBAL_RES)  # 40080
GLOBAL_HEIGHT = round(180.0 / GLOBAL_RES)  # 20040

# Small test bbox inside Switzerland (Bern/Zurich area)
TEST_BBOX = (7, 46, 8, 47)
SWITZERLAND_BBOX = (5, 45, 11, 48)
EUROPE_BBOX = (-25, 34, 45, 72)

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DEFAULT_YEAR = 2024
N_BANDS = 64
NODATA = -9999.0


def build_embeddings(year: int) -> ee.Image:
    """Load annual AlphaEarth embeddings and aggregate to ~1km using mean resampling.

    Each tile in the collection lives in its own UTM projection. We apply
    reduceResolution per tile (so the mean is computed correctly in each tile's
    native projection), then mosaic. This properly averages the ~10 000 native
    10m pixels that fall within each 1km output pixel, rather than point-sampling
    as mosaic().resample("bilinear") would do.
    """
    filtered = ee.ImageCollection(COLLECTION).filter(
        ee.Filter.calendarRange(year, year, "year")
    )
    aggregated = filtered.map(
        lambda img: img.setDefaultProjection(img.projection())
        .reduceResolution(reducer=ee.Reducer.mean(), bestEffort=True)
        .float()
    )
    return aggregated.mosaic().unmask(NODATA)


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
    embeddings: ee.Image,
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
                embeddings,
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

    # Normalize nodata: replace sentinel with NaN; discard all-nodata tiles.
    tmp = output.with_suffix(".tmp.tif")
    with rasterio.open(output) as src:
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        data[data == NODATA] = np.nan
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
            dst.write(data)
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


def tile_name(year: int, lat_s: int, lon_w: int) -> str:
    ns = "N" if lat_s >= 0 else "S"
    ew = "E" if lon_w >= 0 else "W"
    return f"satellite_embedding_{year}_{ns}{abs(lat_s):02d}_{ew}{abs(lon_w):03d}.tif"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["test", "switzerland", "europe", "global"])
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Year of the annual embedding to download (default: {DEFAULT_YEAR})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=3,
        help="Grid cell size in degrees (default: 3; 64 bands × 3°×3° ≈ 36 MB, within GEE's 50 MB limit)",
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
    embeddings = build_embeddings(args.year)

    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = args.bbox
        ns = "N" if lat_min >= 0 else "S"
        ew = "E" if lon_min >= 0 else "W"
        output = args.output or Path(
            f"data/1km/predictors_new/embeddings/{tile_name(args.year, int(lat_min), int(lon_min))}"
        )
        download_tile(embeddings, lon_min, lat_min, lon_max, lat_max, output)
        if output.exists():
            with rasterio.open(output) as src:
                print(
                    f"Downloaded: {output}  shape={src.read().shape}  dtype={src.dtypes[0]}"
                )

    elif args.region == "global":
        import sys

        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        # Capture the real stderr now, before any worker thread can swap it.
        progress_console = Console(file=sys.stderr)

        out_dir = (
            args.output or Path("data/1km/predictors_new/embeddings")
        ) / f"tiles_{args.year}"
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

        pending = [
            (lat_s, lon_w)
            for lat_s, lon_w in cells
            if not (out_dir / tile_name(args.year, lat_s, lon_w)).exists()
        ]
        print(
            f"{len(cells) - len(pending)}/{len(cells)} tiles already downloaded, "
            f"{len(pending)} remaining."
        )

        def _download(lat_s, lon_w):
            lat_n = min(90, lat_s + grid)
            lon_e = min(180, lon_w + grid)
            name = tile_name(args.year, lat_s, lon_w)
            # Only redirect stdout (geemap's error print). Do NOT redirect stderr —
            # contextlib.redirect_stderr is not thread-safe and would swap the
            # global sys.stderr that Rich's Progress refresh thread writes to.
            with contextlib.redirect_stdout(io.StringIO()):
                download_tile(embeddings, lon_w, lat_s, lon_e, lat_n, out_dir / name)

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=progress_console,
        )
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
        from rasterio.merge import merge as rasterio_merge

        region = args.region or "test"
        if region == "test":
            bbox = TEST_BBOX
        elif region == "switzerland":
            bbox = SWITZERLAND_BBOX
        else:
            bbox = EUROPE_BBOX

        lon_min, lat_min, lon_max, lat_max = bbox
        output = args.output or Path(
            f"data/1km/predictors_new/embeddings/satellite_embedding_{args.year}_{region}.tif"
        )

        # Build a grid of tiles covering the bbox, then merge.
        # GEE enforces a 50 MB per-request limit; with 64 bands a single large
        # bbox easily exceeds it, so we always tile even for small regions.
        grid = args.grid_size
        lon_starts = [
            l for l in range(-180, 180, grid) if l < lon_max and l + grid > lon_min
        ]
        lat_starts = [
            l for l in range(-90, 90, grid) if l < lat_max and l + grid > lat_min
        ]
        cells = [(lat_s, lon_w) for lat_s in lat_starts for lon_w in lon_starts]

        tile_dir = output.parent / f"tiles_{args.year}"
        tile_dir.mkdir(parents=True, exist_ok=True)

        pending = [
            (lat_s, lon_w)
            for lat_s, lon_w in cells
            if not (tile_dir / tile_name(args.year, lat_s, lon_w)).exists()
        ]
        print(
            f"{len(cells) - len(pending)}/{len(cells)} tiles already downloaded, "
            f"{len(pending)} remaining."
        )

        def _download(lat_s, lon_w):
            lat_n = min(lat_max, lat_s + grid)
            lon_e = min(lon_max, lon_w + grid)
            name = tile_name(args.year, lat_s, lon_w)
            with contextlib.redirect_stdout(io.StringIO()):
                download_tile(embeddings, lon_w, lat_s, lon_e, lat_n, tile_dir / name)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_download, lat_s, lon_w): (lat_s, lon_w)
                for lat_s, lon_w in pending
            }
            total = len(pending)
            for i, future in enumerate(as_completed(futures), 1):
                lat_s, lon_w = futures[future]
                future.result()
                print(f"  [{i}/{total}] tile ({lat_s}, {lon_w}) done")

        tile_paths = sorted(tile_dir.glob("*.tif"))
        if not tile_paths:
            print("No tiles downloaded.")
            return

        if not output.exists():
            print(f"Merging {len(tile_paths)} tiles → {output.name} …")
            datasets = [rasterio.open(p) for p in tile_paths]
            mosaic, transform = rasterio_merge(datasets)
            profile = datasets[0].profile | {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
            }
            for ds in datasets:
                ds.close()
            with rasterio.open(output, "w", **profile) as dst:
                dst.write(mosaic)

        with rasterio.open(output) as src:
            print(
                f"Downloaded: {output}  shape={src.read().shape}  dtype={src.dtypes[0]}"
            )


if __name__ == "__main__":
    main()
