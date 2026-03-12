import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import zarr
from tqdm import tqdm


def print_zarr_summary(store: zarr.Group) -> None:
    """Print the structure and attributes of a zarr store."""
    print("--- ZARR SUMMARY ---")
    print()
    print("Attributes:")
    for k, v in store.attrs.items():
        print(f"  - {k}: {v}")
    print()

    print("\nArrays:")
    for group_name, group in store.groups():
        for arr_name, arr in group.arrays():
            print(f"{group_name}/{arr_name}")
            print(f"  - shape={arr.shape}")
            print(f"  - dtype={arr.dtype}")
            print(f"  - chunks={arr.chunks}")

            if "files" in arr.attrs:
                print("  - files:")
                for f in arr.attrs["files"]:
                    print(f"      - {f}")

    print("\nBounds:")
    bounds = store["bounds"]
    print(f"  - shape={bounds.shape}")
    print(f"  - dtype={bounds.dtype}")


def chip_to_zarr(
    predictors: dict[str, Path],
    targets: dict[str, Path],
    output_path: Path,
    patch_size: int,
    stride: int,
) -> None:
    print("Opening datasets...")
    all_srcs = {
        name: [rasterio.open(f) for f in sorted(path.glob("*.tif"))]
        for name, path in {**predictors, **targets}.items()
    }

    # Use first file of first dataset as the reference grid
    ref = next(iter(all_srcs.values()))[0]
    height, width, crs, transform = ref.height, ref.width, ref.crs, ref.transform
    print(f"Reference grid: {height}×{width}, CRS=EPSG:{crs.to_epsg()}")

    # Verify all datasets share the same grid
    for name, srcs in all_srcs.items():
        for src in srcs:
            assert src.height == height and src.width == width, (
                f"{name}: shape mismatch ({src.height}x{src.width} vs {height}x{width})"
            )
            assert src.crs == crs, f"{name}: CRS mismatch"
            assert src.transform == transform, f"{name}: transform mismatch"
            assert src.count == 1, (
                f"{name}: expected 1 band per file, got {src.count} in {src.name}"
            )

    # Calculate number of chips
    n_cols = math.ceil((width - patch_size) / stride) + 1
    n_rows = math.ceil((height - patch_size) / stride) + 1
    n_chips = n_rows * n_cols
    print(f"Chips: {n_rows} rows × {n_cols} cols = {n_chips} total\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize zarr store and metadata
    store = zarr.open_group(str(output_path), mode="w")

    store.attrs["split"] = output_path.stem
    store.attrs["epsg_crs"] = crs.to_epsg()
    store.attrs["res_km"] = [round(transform.a / 1000), round(abs(transform.e) / 1000)]
    store.attrs["transform"] = list(transform)
    store.attrs["patch_size"] = patch_size
    store.attrs["stride"] = stride
    store.attrs["creation_date"] = datetime.now().isoformat(timespec="seconds")

    # Pre-allocate one array per dataset, organised into subgroups
    predictors_group = store.require_group("predictors")
    targets_group = store.require_group("targets")

    arrays = {}
    for name in predictors.keys():
        srcs = all_srcs[name]
        n_bands = len(srcs)
        arrays[name] = predictors_group.create_array(
            name,
            shape=(n_chips, n_bands, patch_size, patch_size),
            chunks=(1, n_bands, patch_size, patch_size),
            dtype="f4",
        )
        arrays[name].attrs["files"] = [Path(src.name).name for src in srcs]
        print(
            f"  predictors/{name}: {n_chips} × {n_bands} bands -> {arrays[name].shape}"
        )

    for name in targets.keys():
        srcs = all_srcs[name]
        n_bands = len(srcs)
        arrays[name] = targets_group.create_array(
            name,
            shape=(n_chips, n_bands, patch_size, patch_size),
            chunks=(1, n_bands, patch_size, patch_size),
            dtype="f4",
        )
        arrays[name].attrs["files"] = [Path(src.name).name for src in srcs]
        print(f"  targets/{name}: {n_chips} × {n_bands} bands -> {arrays[name].shape}")

    bounds_arr = store.create_array(
        "bounds",
        shape=(n_chips, 4),
        chunks=(1024, 4),
        dtype="f8",
    )

    print("\nChipping...")
    idx = 0
    for row in tqdm(range(n_rows), desc="Rows"):
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

    for srcs in all_srcs.values():
        for src in srcs:
            src.close()

    print(f"\nDone. {idx} chips saved to {output_path}")
    print_zarr_summary(store)
