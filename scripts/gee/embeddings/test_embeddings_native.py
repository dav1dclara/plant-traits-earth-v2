"""
Verification script: download AlphaEarth embeddings at native 10m resolution
as a grid of neighboring tiles around Lake Zurich, then merge into a mosaic.

Uses the same Lake Zurich area as test_worldcover_aggregation.py.

Each tile is 0.02° × 0.02° (~222×222 px, ~12 MB) to stay within GEE's 50 MB
per-request limit (64 bands × 222 × 222 × 4 bytes ≈ 12 MB).

Saves to data/1km/predictors_new/embeddings/native_grid/:
  tile_r{row}_c{col}.tif         — individual tiles
  emb_lake_zurich_native_10m.tif — merged mosaic (all tiles)
"""

import contextlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
from rasterio.merge import merge

OUT_DIR = Path("data/1km/predictors_new/embeddings")
TILE_DIR = OUT_DIR / "native_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2024
TILE_DEG = 0.02  # 0.02° ≈ 2.2 km at Swiss latitudes → ~222×222 px at 10 m

# 6-column × 4-row grid covering the middle of Lake Zurich
# lon: 8.58 → 8.70  (6 tiles)
# lat: 47.20 → 47.28 (4 tiles)
GRID_LON0 = 8.58
GRID_LAT0 = 47.20
GRID_COLS = 6
GRID_ROWS = 4

ee.Initialize(project="plant-traits-earth-v2")

filtered = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filter(
    ee.Filter.calendarRange(YEAR, YEAR, "year")
)
embeddings = filtered.mosaic().float()


def download_tile(row: int, col: int) -> Path:
    lon_min = GRID_LON0 + col * TILE_DEG
    lat_min = GRID_LAT0 + row * TILE_DEG
    lon_max = lon_min + TILE_DEG
    lat_max = lat_min + TILE_DEG
    out = TILE_DIR / f"tile_r{row:02d}_c{col:02d}.tif"
    if out.exists():
        return out
    region = ee.Geometry.BBox(lon_min, lat_min, lon_max, lat_max)
    with contextlib.redirect_stdout(io.StringIO()):
        geemap.ee_export_image(
            embeddings,
            filename=str(out),
            crs="EPSG:4326",
            scale=10,
            region=region,
            file_per_band=False,
            verbose=False,
        )
    return out


# ── Download grid in parallel ─────────────────────────────────────────────────
tiles = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
total = len(tiles)
done = sum(1 for r, c in tiles if (TILE_DIR / f"tile_r{r:02d}_c{c:02d}.tif").exists())
print(
    f"Downloading {total - done}/{total} tiles ({GRID_COLS}×{GRID_ROWS} grid, {TILE_DEG}°/tile) …"
)

with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(download_tile, r, c): (r, c) for r, c in tiles}
    for i, future in enumerate(as_completed(futures), 1):
        r, c = futures[future]
        path = future.result()
        status = "ok" if path.exists() else "FAILED"
        print(f"  [{i}/{total}] r={r} c={c} → {status}")

# ── Merge into a single mosaic ────────────────────────────────────────────────
tile_paths = sorted(TILE_DIR.glob("tile_r*.tif"))
mosaic_out = OUT_DIR / "emb_lake_zurich_native_10m.tif"

if not mosaic_out.exists() and tile_paths:
    print(f"\nMerging {len(tile_paths)} tiles …")
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile | {
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
    }
    with rasterio.open(mosaic_out, "w", **profile) as dst:
        dst.write(mosaic)
    for ds in datasets:
        ds.close()
    print(f"Written: {mosaic_out.name}")
else:
    print(f"\nExists: {mosaic_out.name}")

# ── Stats ─────────────────────────────────────────────────────────────────────
print("\n── Results ──────────────────────────────────────────────────────────")
with rasterio.open(mosaic_out) as src:
    data = src.read().astype(np.float32)
    print(f"  shape : {data.shape}  ({src.width}×{src.height} px, {src.count} bands)")
    print(f"  res   : {src.res[0]:.6f}° × {src.res[1]:.6f}°")
    print(
        f"  extent: lon [{src.bounds.left:.4f}, {src.bounds.right:.4f}]  "
        f"lat [{src.bounds.bottom:.4f}, {src.bounds.top:.4f}]"
    )
    print(f"  min   : {np.nanmin(data):.4f}")
    print(f"  max   : {np.nanmax(data):.4f}")
    print(f"  mean  : {np.nanmean(data):.4f}")
