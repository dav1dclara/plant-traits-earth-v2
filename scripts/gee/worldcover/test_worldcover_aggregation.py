"""
Verification script: download a tiny area at native 10m and 1km resolution
in EPSG:4326 and compare to check the reduceResolution land mask is correct.

Saves three files to data/1km/predictors_new/worldcover/:
  wc_test_native_10m.tif   — raw WorldCover at ~10m (EPSG:4326)
  wc_test_mask_default.tif — land mask, bestEffort=True, no maxPixels
  wc_test_mask_65536.tif   — land mask, bestEffort=True, maxPixels=65536
"""

import math
import zipfile
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
import requests

OUT_DIR = Path("data/1km/predictors_new/worldcover")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Small box on the south shore of Lake Zurich — guaranteed mixed land/water pixels
BBOX = (8.68, 47.21, 8.74, 47.27)  # lon_min, lat_min, lon_max, lat_max

ee.Initialize(project="plant-traits-earth-v2")
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
region = ee.Geometry.BBox(*BBOX)


def download_raw(out: Path):
    if out.exists():
        print(f"Exists: {out.name}")
        return
    url = wc.getDownloadURL(
        {
            "name": out.stem,
            "filePerBand": False,
            "crs": "EPSG:4326",
            "scale": 10,
            "region": region,
        }
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    zip_path = out.with_suffix(".zip")
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(OUT_DIR)
    zip_path.unlink()
    print(f"Written: {out.name}")


def download_mask(out: Path, **reduce_kwargs):
    if out.exists():
        print(f"Exists: {out.name}")
        return
    img = (
        wc.neq(80)
        .setDefaultProjection(wc.projection())
        .reduceResolution(reducer=ee.Reducer.max(), **reduce_kwargs)
        .uint8()
        .unmask(255)
    )
    geemap.ee_export_image(
        img,
        filename=str(out),
        crs="EPSG:4326",
        scale=1000,
        region=region,
        file_per_band=False,
    )
    # geemap writes nodata=0; fix to 255 so water pixels (0) aren't treated as nodata
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(out) as src:
        profile = src.profile | {"nodata": 255}
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(src.read())
    tmp.replace(out)


download_raw(OUT_DIR / "wc_test_native_10m.tif")
download_mask(OUT_DIR / "wc_test_mask_default.tif", bestEffort=True)
download_mask(OUT_DIR / "wc_test_mask_65536.tif", bestEffort=True, maxPixels=65536)

# ── Analysis ──────────────────────────────────────────────────────────────────
print("\n── Results ──────────────────────────────────────────────────────────")

with rasterio.open(OUT_DIR / "wc_test_native_10m.tif") as src:
    native = src.read(1).astype(np.int16)
    native_transform = src.transform
    print(f"Native 10m shape: {native.shape}")
    vals, counts = np.unique(native, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  WorldCover class {v:3d}: {c} px ({100 * c / native.size:.1f}%)")

with rasterio.open(OUT_DIR / "wc_test_mask_default.tif") as src:
    m_default = src.read(1)
    mask_transform = src.transform
    print(f"\nMask shape: {m_default.shape}")

with rasterio.open(OUT_DIR / "wc_test_mask_65536.tif") as src:
    m_65536 = src.read(1)

for v, label in [(0, "water"), (1, "land"), (255, "nodata")]:
    print(
        f"  {label:6s} ({v}): default={(m_default == v).sum()}  65536={(m_65536 == v).sum()}"
    )

disagree = (m_default != m_65536) & (m_default != 255) & (m_65536 != 255)
print(f"\nPixels where default ≠ explicit 65536: {disagree.sum()} / {m_default.size}")

print("\nPer 1km pixel: fraction of native 10m pixels that are water (class 80):")
for row in range(m_default.shape[0]):
    for col in range(m_default.shape[1]):
        x0 = mask_transform.c + col * mask_transform.a
        y0 = mask_transform.f + row * mask_transform.e
        x1, y1 = x0 + mask_transform.a, y0 + mask_transform.e
        # Map to native pixel indices (both in EPSG:4326, no transform needed)
        # y0 is the top (north) edge, y1 is the bottom (south) edge (y1 < y0)
        c0 = max(0, math.floor((x0 - native_transform.c) / native_transform.a))
        c1 = min(
            native.shape[1], math.ceil((x1 - native_transform.c) / native_transform.a)
        )
        r0 = max(0, math.floor((y0 - native_transform.f) / native_transform.e))
        r1 = min(
            native.shape[0], math.ceil((y1 - native_transform.f) / native_transform.e)
        )
        if c1 <= c0 or r1 <= r0:
            continue
        chunk = native[r0:r1, c0:c1]
        frac_water = (chunk == 80).mean()
        all_water = bool((chunk == 80).all())
        correct = 0 if all_water else 1
        print(
            f"  [{row},{col}]: n={chunk.size:5d}  frac_water={frac_water:.2f}"
            f"  all_water={all_water}  → correct={correct}"
            f"  default={m_default[row, col]}  65536={m_65536[row, col]}"
        )
