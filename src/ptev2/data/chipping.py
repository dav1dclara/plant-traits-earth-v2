import math
from datetime import datetime
from pathlib import Path

BUFFER_SIZE = 512

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


def _init_zarr_store(
    path: Path,
    n_chips: int,
    split_name: str,
    predictors: dict,
    targets: dict,
    all_srcs: dict,
    patch_size: int,
    stride: int,
    crs,
    transform,
) -> tuple[zarr.Group, dict, zarr.Array]:
    """Create and pre-allocate a zarr store for one split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.attrs["split"] = split_name
    store.attrs["epsg_crs"] = crs.to_epsg()
    store.attrs["res_km"] = [transform.a, abs(transform.e)]
    store.attrs["transform"] = list(transform)
    store.attrs["patch_size"] = patch_size
    store.attrs["stride"] = stride
    store.attrs["creation_date"] = datetime.now().isoformat(timespec="seconds")

    arrays = {}
    pred_group = store.require_group("predictors")
    tgt_group = store.require_group("targets")

    for name, group, group_name in [
        (n, pred_group, "predictors") for n in predictors
    ] + [(n, tgt_group, "targets") for n in targets]:
        srcs = all_srcs[name]
        n_bands = sum(src.count for src in srcs)
        arrays[name] = group.create_array(
            name,
            shape=(n_chips, n_bands, patch_size, patch_size),
            chunks=(1, n_bands, patch_size, patch_size),
            dtype="f4",
        )
        arrays[name].attrs["files"] = [Path(src.name).name for src in srcs]
        print(
            f"  {group_name}/{name}: {n_chips} × {n_bands} bands -> {arrays[name].shape}"
        )

    bounds_arr = store.create_array(
        "bounds", shape=(n_chips, 4), chunks=(1024, 4), dtype="f8"
    )

    return store, arrays, bounds_arr


def chip_to_zarr(
    predictors: dict[str, list[Path]],
    targets: dict[str, list[Path]],
    output_dir: Path,
    patch_size: int,
    stride: int,
    split_labels: np.ndarray,
    split_encoding: dict[str, int],
    pixel_split_mask: np.ndarray | None = None,
) -> None:
    """
    Chip rasters into one zarr store per split in output_dir.

    Args:
        split_labels:    int8 array (n_chips,) mapping each chip to a split code.
        split_encoding:  e.g. {"train": 0, "val": 1, "test": 2}; -1 = unknown/skip.
        pixel_split_mask: int8 array (height, width) with the same encoding;
                         pixels outside a chip's own split are set to NaN.
    """
    print("Opening datasets...")
    all_srcs = {
        name: [rasterio.open(f) for f in files]
        for name, files in {**predictors, **targets}.items()
    }

    ref = next(iter(all_srcs.values()))[0]
    height, width, crs, transform = ref.height, ref.width, ref.crs, ref.transform
    print(f"Reference grid: {height}×{width}, CRS=EPSG:{crs.to_epsg()}")

    for name, srcs in all_srcs.items():
        for src in srcs:
            assert src.height == height and src.width == width, (
                f"{name}: shape mismatch ({src.height}x{src.width} vs {height}x{width})"
            )
            assert src.crs == crs, f"{name}: CRS mismatch"
            assert src.transform == transform, f"{name}: transform mismatch"

    n_cols = math.ceil((width - patch_size) / stride) + 1
    n_rows = math.ceil((height - patch_size) / stride) + 1
    n_chips = n_rows * n_cols
    print(f"Chips: {n_rows} rows × {n_cols} cols = {n_chips} total\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    code_to_name = {code: name for name, code in split_encoding.items()}
    n_per_split = {
        name: int((split_labels == code).sum()) for name, code in split_encoding.items()
    }
    print("Chips per split:")
    for name, n in n_per_split.items():
        print(f"  {name}: {n:,}")
    unknown = int((split_labels == -1).sum())
    if unknown:
        print(f"  unknown (skipped): {unknown:,}")
    print()

    # Pre-allocate one zarr store per split
    split_arrays = {}
    split_bounds_arrs = {}
    for split_name, n_split in n_per_split.items():
        print(f"Initialising {split_name}.zarr ({n_split:,} chips)...")
        _, split_arrays[split_name], split_bounds_arrs[split_name] = _init_zarr_store(
            path=output_dir / f"{split_name}.zarr",
            n_chips=n_split,
            split_name=split_name,
            predictors=predictors,
            targets=targets,
            all_srcs=all_srcs,
            patch_size=patch_size,
            stride=stride,
            crs=crs,
            transform=transform,
        )

    # Per-split buffers
    split_bufs = {
        split_name: {
            name: np.empty(
                (
                    BUFFER_SIZE,
                    sum(s.count for s in all_srcs[name]),
                    patch_size,
                    patch_size,
                ),
                dtype=np.float32,
            )
            for name in all_srcs
        }
        for split_name in split_encoding
    }
    split_bounds_bufs = {
        split_name: np.empty((BUFFER_SIZE, 4), dtype=np.float64)
        for split_name in split_encoding
    }
    split_buf_pos = {split_name: 0 for split_name in split_encoding}
    split_buf_start = {split_name: 0 for split_name in split_encoding}

    def flush(split_name: str, count: int) -> None:
        start = split_buf_start[split_name]
        for name in all_srcs:
            split_arrays[split_name][name][start : start + count] = split_bufs[
                split_name
            ][name][:count]
        split_bounds_arrs[split_name][start : start + count] = split_bounds_bufs[
            split_name
        ][:count]

    print("\nChipping...")
    chip_global_idx = 0
    for row in tqdm(range(n_rows), desc="Rows"):
        for col in range(n_cols):
            split_code = int(split_labels[chip_global_idx])
            chip_global_idx += 1

            if split_code not in code_to_name:
                continue  # unknown — skip

            split_name = code_to_name[split_code]
            y_px, x_px = row * stride, col * stride
            window = rasterio.windows.Window(x_px, y_px, patch_size, patch_size)

            # Read chip
            chip_data = {}
            for name, srcs in all_srcs.items():
                chip_data[name] = np.concatenate(
                    [
                        src.read(window=window, boundless=True, fill_value=0).astype(
                            np.float32
                        )
                        for src in srcs
                    ],
                    axis=0,
                )

            # Mask pixels that belong to a different split
            if pixel_split_mask is not None:
                y_end = min(y_px + patch_size, height)
                x_end = min(x_px + patch_size, width)
                mask_chip = np.full((patch_size, patch_size), -1, dtype=np.int8)
                mask_chip[: y_end - y_px, : x_end - x_px] = pixel_split_mask[
                    y_px:y_end, x_px:x_end
                ]
                outside = mask_chip != split_code  # (patch_size, patch_size)
                for name in chip_data:
                    chip_data[name][:, outside] = np.nan

            buf_pos = split_buf_pos[split_name]
            for name in all_srcs:
                split_bufs[split_name][name][buf_pos] = chip_data[name]

            win_t = rasterio.windows.transform(window, transform)
            split_bounds_bufs[split_name][buf_pos] = [
                win_t.c,
                win_t.f + patch_size * transform.e,
                win_t.c + patch_size * transform.a,
                win_t.f,
            ]

            split_buf_pos[split_name] += 1
            if split_buf_pos[split_name] == BUFFER_SIZE:
                flush(split_name, BUFFER_SIZE)
                split_buf_start[split_name] += BUFFER_SIZE
                split_buf_pos[split_name] = 0

    # Final flush
    for split_name in split_encoding:
        pos = split_buf_pos[split_name]
        if pos > 0:
            flush(split_name, pos)

    for srcs in all_srcs.values():
        for src in srcs:
            src.close()

    print(f"\nDone. Splits written to {output_dir}")
    for split_name, n_split in n_per_split.items():
        print(f"  {split_name}.zarr: {n_split:,} chips")
