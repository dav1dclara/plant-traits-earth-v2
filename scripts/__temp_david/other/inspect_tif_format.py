"""Inspect the format of source GeoTIFF files (tiling, compression, band count)."""

from pathlib import Path

import rasterio

DATA_ROOT = Path("/scratch3/plant-traits-v2/data/22km/eo_data")

DATASETS = ["canopy_height", "modis", "soilgrids", "vodca", "worldclim"]


def inspect_dataset(name: str, path: Path) -> None:
    tifs = sorted(path.glob("*.tif"))
    if not tifs:
        print(f"  {name}: no .tif files found at {path}")
        return

    print(f"  {name}: {len(tifs)} file(s)")
    src = rasterio.open(tifs[0])
    profile = src.profile
    print(f"    file:        {tifs[0].name}")
    print(f"    size:        {src.width} × {src.height} px")
    print(f"    bands:       {src.count}")
    print(f"    dtype:       {profile['dtype']}")
    print(f"    driver:      {profile['driver']}")
    print(f"    compress:    {profile.get('compress', 'none')}")
    block_w, block_h = src.block_shapes[0] if src.block_shapes else (None, None)
    tiled = profile.get("tiled", False)
    print(f"    tiled:       {tiled}")
    print(f"    block shape: {block_w} × {block_h} px")
    src.close()


if __name__ == "__main__":
    print("--- TIFF FORMAT INSPECTION ---\n")
    for name in DATASETS:
        inspect_dataset(name, DATA_ROOT / name)
        print()
