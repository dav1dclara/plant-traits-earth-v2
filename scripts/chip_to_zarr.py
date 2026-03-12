"""Chip all EO and trait rasters into a single zarr store."""

import math
from pathlib import Path

import hydra
import numpy as np
import rasterio
import rasterio.windows
import zarr
from omegaconf import DictConfig

DATA_ROOT = Path("/scratch3/plant-traits-v2/data/22km")

DATASET_PATHS = {
    "canopy_height": DATA_ROOT / "eo_data/canopy_height",
    "modis": DATA_ROOT / "eo_data/modis",
    "soilgrids": DATA_ROOT / "eo_data/soilgrids",
    "vodca": DATA_ROOT / "eo_data/vodca",
    "worldclim": DATA_ROOT / "eo_data/worldclim",
    "gbif": DATA_ROOT / "gbif",
    "splot": DATA_ROOT / "splot",
}


def open_dataset(path: Path) -> list[rasterio.DatasetReader]:
    files = sorted(path.glob("*.tif"))
    assert files, f"No .tif files found in {path}"
    return [rasterio.open(f) for f in files]


def chip_to_zarr(cfg: DictConfig) -> None:
    patch_size = cfg.patch_size
    stride = cfg.stride if cfg.stride > 0 else patch_size
    output_path = Path(cfg.output_path)

    datasets = {name for name, enabled in cfg.datasets.items() if enabled}

    print("Opening datasets...")
    all_srcs = {name: open_dataset(DATASET_PATHS[name]) for name in datasets}

    # Use first file of first dataset as the reference grid
    ref = next(iter(all_srcs.values()))[0]
    height, width, crs, transform = ref.height, ref.width, ref.crs, ref.transform
    print(f"Reference grid: {height}×{width}, CRS={crs.to_epsg()}")

    # Verify all datasets share the same grid
    for name, srcs in all_srcs.items():
        for src in srcs:
            assert src.height == height and src.width == width, (
                f"{name}: shape mismatch ({src.height}×{src.width} vs {height}×{width})"
            )
            assert src.crs == crs, f"{name}: CRS mismatch"

    n_cols = math.ceil((width - patch_size) / stride) + 1
    n_rows = math.ceil((height - patch_size) / stride) + 1
    n_chips = n_rows * n_cols
    print(f"Chips: {n_rows} rows × {n_cols} cols = {n_chips} total\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(output_path), mode="w")
    store.attrs["crs"] = crs.to_epsg()
    store.attrs["transform"] = list(transform)
    store.attrs["patch_size"] = patch_size
    store.attrs["stride"] = stride

    # Pre-allocate one array per dataset
    arrays = {}
    for name, srcs in all_srcs.items():
        n_bands = len(srcs)
        arrays[name] = store.create_array(
            name,
            shape=(n_chips, n_bands, patch_size, patch_size),
            chunks=(1, n_bands, patch_size, patch_size),
            dtype="f4",
        )
        print(f"  {name}: {n_chips} × {n_bands} bands → {arrays[name].shape}")

    bounds_arr = store.create_array(
        "bounds", shape=(n_chips, 4), chunks=(1024, 4), dtype="f8"
    )

    print("\nChipping...")
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            y, x = row * stride, col * stride
            window = rasterio.windows.Window(x, y, patch_size, patch_size)

            for name, srcs in all_srcs.items():
                chip = np.stack(
                    [
                        src.read(1, window=window, boundless=True, fill_value=0).astype(
                            np.float32
                        )
                        for src in srcs
                    ],
                    axis=0,
                )
                arrays[name][idx] = chip

            win_transform = rasterio.windows.transform(window, transform)
            min_x = win_transform.c
            max_y = win_transform.f
            max_x = min_x + patch_size * transform.a
            min_y = max_y + patch_size * transform.e
            bounds_arr[idx] = [min_x, min_y, max_x, max_y]

            idx += 1

        print(f"  row {row + 1}/{n_rows} — {idx} chips")

    for srcs in all_srcs.values():
        for src in srcs:
            src.close()

    print(f"\nDone. {idx} chips saved to {output_path}")


@hydra.main(
    version_base=None, config_path="../config/preprocess", config_name="default"
)
def main(cfg: DictConfig) -> None:
    chip_to_zarr(cfg)


if __name__ == "__main__":
    main()
