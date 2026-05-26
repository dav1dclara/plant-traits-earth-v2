#!/usr/bin/env python
"""Convert prediction rasters (data/1km/predictions/float16/*.tif) to PMTiles archives.

Mirrors ``dem_to_pmtiles.py``: each source is reprojected on the fly to Web Mercator
(EPSG:3857) via a clamped WarpedVRT, sliced into the standard XYZ tile pyramid,
colour-mapped to RGBA (NoData -> transparent), and packed into a ``.pmtiles`` file.

Prediction rasters share the DEM's projection (EPSG:6933) and NaN NoData, but each
trait has its own value range — so by default we auto-detect vmin/vmax from the
2nd/98th percentiles of a decimated read. Pass ``--vmin``/``--vmax`` to override
(only meaningful with a single input).

After each successful tiling the script also (re)writes ``viewer/tiles/ranges.json``
mapping ``trait_id -> {vmin, vmax, colormap}`` so the viewer can label colorbars
correctly. Outputs already on disk are skipped (resume-safe), but their ranges
are still refreshed from each archive's own header — re-running rebuilds a
complete manifest without re-tiling.

Examples
--------
    # Batch all traits in data/1km/predictions/float16/ -> viewer/tiles/<stem>.pmtiles
    python viewer/scripts/predictor_to_pmtiles.py

    # Same, but tile 8 traits in parallel
    python viewer/scripts/predictor_to_pmtiles.py -j 8

    # Single trait, custom output path, fixed range / different colormap
    python viewer/scripts/predictor_to_pmtiles.py data/1km/predictions/float16/4.tif \
        -o 4.pmtiles --vmin 0 --vmax 1 --colormap magma

    # Re-tile a subset, overwriting existing outputs
    python viewer/scripts/predictor_to_pmtiles.py --force \
        data/1km/predictions/float16/{4,13,18}.tif
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import morecantile
import numpy as np
import rasterio
from PIL import Image
from pmtiles.reader import MmapSource
from pmtiles.reader import Reader as PMTilesReader
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
DEFAULT_SRC_DIR = REPO_ROOT / "data/1km/predictions/float16"
DEFAULT_OUT_DIR = REPO_ROOT / "viewer/tiles"


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
    tmpdir = Path(tempfile.mkdtemp(prefix="pred_ovr_"))
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


def read_existing_range(path: Path) -> tuple[float, float, str] | None:
    """Pull the (vmin, vmax, colormap) we baked into a previously-written PMTiles
    header. Returns None if the file is unreadable or missing the fields."""
    try:
        with open(path, "rb") as f:
            meta = PMTilesReader(MmapSource(f)).metadata()
        return float(meta["vmin"]), float(meta["vmax"]), meta.get("colormap", "viridis")
    except Exception as e:
        print(f"warning: could not read metadata from {path}: {e}")
        return None


def update_ranges(
    out_dir: Path, trait_id: str, vmin: float, vmax: float, colormap: str
) -> None:
    """Merge this trait's encoding params into viewer/tiles/ranges.json (atomic).

    The viewer fetches this sidecar to label colorbars with the right vmin/vmax/
    colormap for each trait, so we keep it in sync with what's actually baked
    into the .pmtiles archives."""
    path = out_dir / "ranges.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"warning: {path} was unreadable, recreating from scratch")
    data[trait_id] = {"vmin": vmin, "vmax": vmax, "colormap": colormap}
    sorted_data = dict(sorted(data.items(), key=lambda kv: int(kv[0].lstrip("X") or 0)))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted_data, indent=2) + "\n")
    tmp.replace(path)


def process(src: Path, out: Path, args) -> tuple[float, float]:
    """Tile one raster to one PMTiles archive. Writes to <out>.tmp and renames on
    success, so a half-finished file from a Ctrl-C never gets mistaken for done.
    Returns the (vmin, vmax) actually used (auto-detected if not supplied)."""
    # With multiple workers the per-tile progress bars interleave into noise — the
    # main process prints a single "done" line per trait instead.
    quiet = getattr(args, "workers", 1) > 1
    tms = morecantile.tms.get("WebMercatorQuad")
    lut = colormap_lut(args.colormap)
    resampling = Resampling[args.resampling]
    tile_type = TileType.WEBP if args.format == "webp" else TileType.PNG
    # VRT resolution sized to the deepest zoom, so the warp never upsamples.
    res = (2 * MERC_MAX) / (args.tilesize * 2**args.max_zoom)

    tmp_out = out.with_suffix(out.suffix + ".tmp")
    work_src, tmpdir = ensure_overviews(src)
    try:
        with rasterio.open(work_src) as ds:
            vmin, vmax = args.vmin, args.vmax
            if vmin is None or vmax is None:
                p2, p98 = auto_value_range(ds, 2, 98)
                vmin = p2 if vmin is None else vmin
                vmax = p98 if vmax is None else vmax

            # Data footprint in WGS84, clamped to the Mercator latitude limit.
            w, s, e, n = transform_bounds(
                ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21
            )
            s, n = max(s, -MERC_LAT), min(n, MERC_LAT)
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", "EPSG:3857", w, s, e, n, densify_pts=21
            )
            width, height = round((right - left) / res), round((top - bottom) / res)
            transform = rasterio.transform.from_origin(left, top, res, res)

            if not quiet:
                print(f"src       {src}")
                print(f"crs       {ds.crs} -> EPSG:3857   vrt {width}x{height} px")
                print(f"value map [{vmin:.4g}, {vmax:.4g}] -> {args.colormap}")
                print(
                    f"zooms     {args.min_zoom}-{args.max_zoom}   format {args.format}"
                )

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
                open(tmp_out, "wb") as fh,
            ):
                writer = Writer(fh)
                written = 0
                for tid, t in tqdm(jobs, desc="tiling", unit="tile", disable=quiet):
                    try:
                        data = reader.tile(t.x, t.y, t.z, tilesize=args.tilesize).data[
                            0
                        ]
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
                    {
                        "source": src.name,
                        "vmin": f"{vmin:.6g}",
                        "vmax": f"{vmax:.6g}",
                        "colormap": args.colormap,
                    },
                )
    except BaseException:
        # Any failure (including Ctrl-C) leaves the .tmp file behind and *no*
        # final file — the next run will re-attempt this trait.
        tmp_out.unlink(missing_ok=True)
        raise
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Atomic rename: from here on the final path's existence == "fully written".
    tmp_out.replace(out)
    if not quiet:
        size_mb = out.stat().st_size / 1e6
        print(f"wrote {written} tiles -> {out}  ({size_mb:.1f} MB)")
    return vmin, vmax


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "src",
        nargs="*",
        type=Path,
        default=[],
        help=f"input rasters (default: every *.tif in {DEFAULT_SRC_DIR.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .pmtiles path; only valid with a single input "
        "(default: viewer/tiles/<src stem>.pmtiles)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing outputs (default: skip)",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="parallel traits to tile (default: 1; try cpu_count for batch runs)",
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
        default=None,
        help="value mapped to colormap min (default: 2nd percentile)",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="value mapped to colormap max (default: 98th percentile)",
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

    srcs = args.src or sorted(DEFAULT_SRC_DIR.glob("*.tif"))
    if not srcs:
        raise SystemExit(f"No inputs given and none found in {DEFAULT_SRC_DIR}")
    if args.out and len(srcs) > 1:
        raise SystemExit("-o/--out is only valid with a single input")
    for src in srcs:
        if not src.exists():
            raise SystemExit(f"Input not found: {src}")

    warnings.filterwarnings("ignore")  # silence rio-tiler's no-overviews note

    # Split inputs: those already tiled (refresh ranges.json from their headers)
    # vs. those still to do (queue up for the pool / sequential loop).
    to_process: list[tuple[Path, Path]] = []
    for src in srcs:
        out = args.out or (DEFAULT_OUT_DIR / f"{src.stem}.pmtiles")
        if out.exists() and not args.force:
            print(f"skip {out.name} (exists) — refreshing ranges")
            existing = read_existing_range(out)
            if existing is not None:
                vmin, vmax, colormap = existing
                update_ranges(out.parent, f"X{src.stem}", vmin, vmax, colormap)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        to_process.append((src, out))

    if not to_process:
        print("\nnothing to tile — ranges.json refreshed for existing outputs.")
        return

    workers = max(1, min(args.workers, len(to_process)))
    print(f"\ntiling {len(to_process)} trait(s) with {workers} worker(s)")

    if workers == 1:
        for i, (src, out) in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}] {src.name}")
            vmin, vmax = process(src, out, args)
            # Filenames are numeric trait ids (e.g. 4.tif -> X4); keep ranges.json in sync.
            update_ranges(out.parent, f"X{src.stem}", vmin, vmax, args.colormap)
        return

    # Parallel: each worker tiles one trait end-to-end. ranges.json is only ever
    # updated from this (main) process, so there are no file write races.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process, src, out, args): (src, out) for src, out in to_process
        }
        for done_i, fut in enumerate(as_completed(futures), 1):
            src, out = futures[fut]
            try:
                vmin, vmax = fut.result()
            except Exception as e:
                print(f"[{done_i}/{len(to_process)}] FAILED {src.name}: {e}")
                continue
            update_ranges(out.parent, f"X{src.stem}", vmin, vmax, args.colormap)
            size_mb = out.stat().st_size / 1e6
            print(f"[{done_i}/{len(to_process)}] done {src.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
