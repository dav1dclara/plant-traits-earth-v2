#!/usr/bin/env python
"""Convert a single-band raster (e.g. the GLO-30 DEM) to a raster PMTiles archive.

The source is reprojected on the fly to Web Mercator (EPSG:3857) via a clamped
WarpedVRT, sliced into the standard XYZ tile pyramid, colour-mapped to RGBA
(NoData -> transparent), and packed into one ``.pmtiles`` file ready to upload to
Cloudflare R2 and point the viewer at (``VITE_DEM_PMTILES_URL``).

Pure Python: rio-tiler reads each reprojected tile, the mask is derived from
``isfinite`` (the GLO-30 DEM uses NaN NoData, which rio-tiler's own masking does
not handle), tiles are colour-mapped with a matplotlib LUT and encoded by Pillow,
and pmtiles writes the archive. No GDAL CLI or go-pmtiles binary required.

Examples
--------
    # Default: masked GLO-30 DEM -> dem.pmtiles, zooms 0-8, viridis, 0-6000 m, WEBP
    python viewer/scripts/dem_to_pmtiles.py

    # Any raster, custom zooms / colormap / value range / lossless PNG
    python viewer/scripts/dem_to_pmtiles.py path/to/raster.tif -o out.pmtiles \
        --max-zoom 7 --colormap terrain --vmin 0 --vmax 4000 --format png
"""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib
import morecantile
import numpy as np
import rasterio
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from tqdm import tqdm

# Web Mercator limits.
MERC_LAT = 85.05112878
MERC_MAX = 20037508.342789244

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = (
    REPO_ROOT / "data/1km/predictors_new/glo30/copernicus_glo30_6933_masked.tif"
)


def colormap_lut(name: str) -> np.ndarray:
    """256x4 uint8 RGBA lookup table from a matplotlib colormap."""
    return (matplotlib.colormaps[name](np.arange(256) / 255.0) * 255).astype(np.uint8)


def auto_value_range(dataset, lo: float, hi: float) -> tuple[float, float]:
    """Robust min/max from a decimated read, ignoring NoData."""
    arr = dataset.read(1, out_shape=(1, 360, 720), masked=True).compressed()
    if arr.size == 0:
        raise SystemExit("Source has no valid (non-NoData) pixels.")
    return tuple(float(v) for v in np.percentile(arr, [lo, hi]))


def ensure_overviews(src: Path) -> tuple[Path, Path | None]:
    """rio-tiler decimates the whole raster for every low-zoom tile, which is very
    slow when the source has no overviews. If it has none, copy it to a temp file
    and build them. Returns (path_to_read, temp_dir_to_remove)."""
    with rasterio.open(src) as ds:
        if ds.overviews(1):
            return src, None
    tmpdir = Path(tempfile.mkdtemp(prefix="dem_ovr_"))
    tmp = tmpdir / src.name
    print(f"building overviews (one-off) -> {tmp} …")
    shutil.copy(src, tmp)
    with rasterio.open(tmp, "r+") as ds:
        ds.build_overviews([2, 4, 8, 16, 32, 64, 128], Resampling.average)
    return tmp, tmpdir


def render_tile(
    data: np.ndarray, lut: np.ndarray, vmin: float, vmax: float, fmt: str, quality: int
) -> bytes | None:
    """Colour-map one float tile to RGBA bytes; None if it has no valid pixels."""
    finite = np.isfinite(data)
    if not finite.any():
        return None
    norm = np.clip((np.nan_to_num(data) - vmin) / (vmax - vmin), 0, 1)
    rgba = lut[(norm * 255).astype(np.uint8)]
    rgba[..., 3] = np.where(finite, 255, 0)
    buf = io.BytesIO()
    img = Image.fromarray(rgba, "RGBA")
    if fmt == "webp":
        img.save(buf, "WEBP", quality=quality)
    else:
        img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "src",
        nargs="?",
        type=Path,
        default=DEFAULT_SRC,
        help="input raster (default: masked GLO-30 DEM)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("dem.pmtiles"),
        help="output .pmtiles (default: dem.pmtiles)",
    )
    p.add_argument("--min-zoom", type=int, default=0)
    p.add_argument("--max-zoom", type=int, default=8)
    p.add_argument("--tilesize", type=int, default=256)
    p.add_argument(
        "--colormap",
        default="viridis",
        help="matplotlib colormap name (default: viridis)",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="value mapped to colormap min (default: 0)",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=6000.0,
        help="value mapped to colormap max (default: 6000)",
    )
    p.add_argument(
        "--format",
        choices=("webp", "png"),
        default="webp",
        help="tile image format (default: webp)",
    )
    p.add_argument(
        "--quality", type=int, default=80, help="WEBP quality 1-100 (default: 80)"
    )
    p.add_argument(
        "--resampling", default="bilinear", help="warp resampling (default: bilinear)"
    )
    args = p.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Input not found: {args.src}")

    warnings.filterwarnings("ignore")  # silence rio-tiler's no-overviews note
    tms = morecantile.tms.get("WebMercatorQuad")
    lut = colormap_lut(args.colormap)
    resampling = Resampling[args.resampling]
    tile_type = TileType.WEBP if args.format == "webp" else TileType.PNG
    # VRT resolution sized to the deepest zoom, so the warp never upsamples.
    res = (2 * MERC_MAX) / (args.tilesize * 2**args.max_zoom)

    work_src, tmpdir = ensure_overviews(args.src)
    with rasterio.open(work_src) as ds:
        vmin, vmax = args.vmin, args.vmax
        if vmin is None or vmax is None:
            p2, p98 = auto_value_range(ds, 2, 98)
            vmin = p2 if vmin is None else vmin
            vmax = p98 if vmax is None else vmax

        # Data footprint in WGS84, clamped to the Mercator latitude limit.
        w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
        s, n = max(s, -MERC_LAT), min(n, MERC_LAT)
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", "EPSG:3857", w, s, e, n, densify_pts=21
        )
        width, height = round((right - left) / res), round((top - bottom) / res)
        transform = rasterio.transform.from_origin(left, top, res, res)

        print(f"src       {args.src}")
        print(f"crs       {ds.crs} -> EPSG:3857   vrt {width}x{height} px")
        print(f"value map [{vmin:.1f}, {vmax:.1f}] -> {args.colormap}")
        print(f"zooms     {args.min_zoom}-{args.max_zoom}   format {args.format}")

        # Tiles covering the footprint, sorted by tile id so the archive is clustered.
        jobs = sorted(
            (
                (zxy_to_tileid(t.z, t.x, t.y), t)
                for z in range(args.min_zoom, args.max_zoom + 1)
                for t in tms.tiles(w, s, e, n, [z])
            ),
            key=lambda j: j[0],
        )

        with (
            WarpedVRT(
                ds,
                crs="EPSG:3857",
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
            ) as vrt,
            Reader(str(work_src), dataset=vrt) as reader,
            open(args.out, "wb") as fh,
        ):
            writer = Writer(fh)
            written = 0
            for tid, t in tqdm(jobs, desc="tiling", unit="tile"):
                try:
                    data = reader.tile(t.x, t.y, t.z, tilesize=args.tilesize).data[0]
                except TileOutsideBounds:
                    continue
                png = render_tile(data, lut, vmin, vmax, args.format, args.quality)
                if png is None:  # no valid pixels (e.g. open ocean) -> skip
                    continue
                writer.write_tile(tid, png)
                written += 1

            if written == 0:
                raise SystemExit(
                    "No non-empty tiles produced — check the value range / bounds."
                )

            writer.finalize(
                {
                    "tile_type": tile_type,
                    "tile_compression": Compression.NONE,  # WEBP/PNG are already compressed
                    "min_lon_e7": int(w * 1e7),
                    "min_lat_e7": int(s * 1e7),
                    "max_lon_e7": int(e * 1e7),
                    "max_lat_e7": int(n * 1e7),
                    "center_zoom": args.min_zoom,
                    "center_lon_e7": int((w + e) / 2 * 1e7),
                    "center_lat_e7": int((s + n) / 2 * 1e7),
                },
                {"attribution": "Copernicus GLO-30", "source": args.src.name},
            )

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {written} tiles -> {args.out}  ({size_mb:.1f} MB)")
    print("next: upload to R2 and set VITE_DEM_PMTILES_URL in viewer/.env")


if __name__ == "__main__":
    main()
