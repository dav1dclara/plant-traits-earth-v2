"""
Rasterize Ecoregions2017 polygons into a binary land mask TIFF.

Output:
  /scratch3/plant-traits-v2/data/other/land_mask.tif
    - Band 1: uint8, 1 inside any polygon, 0 outside (nodata = 255)
    - Same CRS, transform, and shape as the trait TIFFs

Usage:
    python create_land_mask.py
"""

from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.features import rasterize

ECOREGIONS_PATH = Path(
    "/scratch3/plant-traits-v2/data/other/Ecoregions2017/Ecoregions2017.shp"
)
OUTPUT_PATH = Path("/scratch3/plant-traits-v2/data/other/land_mask.tif")


@hydra.main(
    version_base=None,
    config_path="../../../config/preprocessing",
    config_name="1km",
)
def main(cfg: DictConfig) -> None:
    targets_dir = Path(cfg.base_dir) / "targets" / cfg.targets.source
    ref_tif = sorted(targets_dir.glob("*.tif"))[0]
    print(f"Reference TIFF: {ref_tif}")

    with rasterio.open(ref_tif) as src:
        crs = src.crs
        transform = src.transform
        n_rows, n_cols = src.shape

    print(f"Grid: {n_rows} x {n_cols}  CRS: {crs}")
    print("Loading and reprojecting ecoregions ...")

    gdf = gpd.read_file(ECOREGIONS_PATH).to_crs(crs)
    shapes = [
        (geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty
    ]

    print("Rasterizing ...")
    mask = rasterize(
        shapes,
        out_shape=(n_rows, n_cols),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="zstd",
        predictor=2,
        blockxsize=256,
        blockysize=256,
        tiled=True,
    ) as dst:
        dst.write(mask, 1)

    print(f"Saved: {OUTPUT_PATH}")
    n_land = int(mask.sum())
    total = n_rows * n_cols
    print(f"Land pixels: {n_land:,} / {total:,}  ({n_land / total:.1%})")


if __name__ == "__main__":
    main()
