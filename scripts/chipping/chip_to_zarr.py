"""Script to chip predictor and target rasters into per-split zarr stores."""

import math
from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.features import rasterize
from shapely.geometry import Point

from ptev2.data.chipping import chip_to_zarr

H3_SPLITS_DIR = Path("/scratch3/plant-traits-v2/data/temp/h3_splits")
SPLIT_ENCODING = {"train": 0, "val": 1, "test": 2}  # -1 = unknown


def compute_split_labels(
    ref_tif: Path, patch_size: int, stride: int, h3_gdf: gpd.GeoDataFrame
) -> np.ndarray:
    """Return an int8 array (n_chips,) with split encoding per chip center."""
    with rasterio.open(ref_tif) as src:
        transform = src.transform
        height, width = src.height, src.width

    n_cols = math.ceil((width - patch_size) / stride) + 1
    n_rows = math.ceil((height - patch_size) / stride) + 1

    # Chip center coordinates in the raster CRS
    xs, ys = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            cx = transform.c + (col * stride + patch_size / 2) * transform.a
            cy = transform.f + (row * stride + patch_size / 2) * transform.e
            xs.append(cx)
            ys.append(cy)

    centers = gpd.GeoDataFrame(
        {"chip_idx": range(len(xs))},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=h3_gdf.crs,
    )
    joined = gpd.sjoin(
        centers, h3_gdf[["split", "geometry"]], how="left", predicate="within"
    )

    labels = np.full(len(xs), -1, dtype=np.int8)
    matched = joined["split"].notna()
    labels[joined.loc[matched, "chip_idx"].values] = (
        joined.loc[matched, "split"]
        .map(SPLIT_ENCODING)
        .fillna(-1)
        .astype(np.int8)
        .values
    )

    for name, code in SPLIT_ENCODING.items():
        print(f"  {name}: {(labels == code).sum():,} chips")
    print(f"  unknown: {(labels == -1).sum():,} chips")
    return labels


def compute_pixel_split_mask(ref_tif: Path, h3_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Rasterize H3 split grid onto the reference raster grid.

    Returns an int8 array (height, width) with SPLIT_ENCODING values,
    -1 where no H3 cell covers the pixel.
    """
    with rasterio.open(ref_tif) as src:
        transform = src.transform
        height, width = src.height, src.width
        h3_proj = h3_gdf.to_crs(src.crs)

    split_codes = h3_proj["split"].map(SPLIT_ENCODING).fillna(-1).astype(int)
    shapes = [
        (geom, int(code))
        for geom, code in zip(h3_proj.geometry, split_codes)
        if geom is not None
    ]
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype=np.int8,
    )


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    zarr_dir = Path(
        "/scratch3/plant-traits-v2/data/chips/22km/"
    )  # TODO: define in config later

    print("--- CHIPPING DATA ---")
    chipping_cfg = cfg.chipping
    patch_size = chipping_cfg.patch_size
    stride = chipping_cfg.stride
    used_predictors = [
        name for name, enabled in chipping_cfg.data.predictors.items() if enabled
    ]

    data_cfg = cfg.data
    data_root_dir = Path(data_cfg.root_dir)
    predictor_paths = {
        name: sorted((data_root_dir / data_cfg[name].path).glob("*.tif"))
        for name in used_predictors
    }

    target_paths = {}
    for name, target_cfg in chipping_cfg.data.targets.items():
        # target_cfg is either False or a dict with {enabled, traits}
        if not target_cfg or not target_cfg.get("enabled", False):
            continue
        tif_dir = data_root_dir / data_cfg[name].path
        traits = target_cfg.get("traits", None)
        if traits:
            files = sorted(tif_dir / f"X{t}.tif" for t in traits)
        else:
            files = sorted(tif_dir.glob("*.tif"))
        target_paths[name] = files

    print(f"Patch size: {patch_size} px")
    print(f"Stride:     {stride} px")
    print("Predictors:")
    for name, path in predictor_paths.items():
        print(f"  - {name}: '{path}'")
    print("Targets:")
    for name, path in target_paths.items():
        print(f"  - {name}: '{path}'")

    print()

    # --- Compute split labels and pixel mask from H3 grid ---
    h3_file = H3_SPLITS_DIR / "h3_res1_X1080_mean.gpkg"
    print(f"Loading H3 split grid from {h3_file.name}...")
    h3_gdf = gpd.read_file(h3_file)
    ref_tif = next(iter(predictor_paths.values()))[0]
    print("Computing chip-level split labels...")
    split_labels = compute_split_labels(ref_tif, patch_size, stride, h3_gdf)
    print("Rasterizing H3 split grid to pixel mask...")
    pixel_split_mask = compute_pixel_split_mask(ref_tif, h3_gdf)

    # train.zarr / val.zarr / test.zarr will be written to output_dir
    output_dir = zarr_dir / f"patch{patch_size}_stride{stride}"

    chip_to_zarr(
        predictors=predictor_paths,
        targets=target_paths,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        split_labels=split_labels,
        split_encoding=SPLIT_ENCODING,
        pixel_split_mask=pixel_split_mask,
    )


if __name__ == "__main__":
    main()
