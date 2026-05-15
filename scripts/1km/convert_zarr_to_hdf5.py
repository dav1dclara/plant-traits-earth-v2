"""One-time conversion of .zarr.zip chip stores to HDF5 (.h5).

Usage:
    python scripts/1km/convert_zarr_to_hdf5.py
    python scripts/1km/convert_zarr_to_hdf5.py --splits train val  # specific splits
    python scripts/1km/convert_zarr_to_hdf5.py --compress           # gzip level 1

Output: {chips_dir}/{split}.h5 alongside the existing .zarr.zip files.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ptev2.data.dataloader import _open_zarr

BATCH = 512  # chips copied per iteration


def convert_split(zarr_path: Path, h5_path: Path, compress: bool) -> None:
    print(f"\n{zarr_path.name} → {h5_path.name}")
    store = _open_zarr(zarr_path)

    compression = "gzip" if compress else None
    compression_opts = 1 if compress else None

    t0 = time.perf_counter()
    with h5py.File(h5_path, "w") as f:
        for group_name in ("predictors", "targets"):
            grp = store[group_name]
            h5_grp = f.require_group(group_name)

            # copy group attrs (e.g. band_names on targets)
            for k, v in grp.attrs.items():
                h5_grp.attrs[k] = v

            for arr_name, arr in grp.arrays():
                N, *rest = arr.shape
                chunk_shape = (1, *rest)
                print(f"  {group_name}/{arr_name}: shape={arr.shape} dtype={arr.dtype}")
                ds = h5_grp.create_dataset(
                    arr_name,
                    shape=arr.shape,
                    dtype=np.float32,
                    chunks=chunk_shape,
                    compression=compression,
                    compression_opts=compression_opts,
                )
                for start in tqdm(range(0, N, BATCH), unit="batch", leave=False):
                    end = min(start + BATCH, N)
                    ds[start:end] = arr[start:end]

    elapsed = time.perf_counter() - t0
    size_gb = h5_path.stat().st_size / 1e9
    print(f"  Done in {elapsed:.0f}s  →  {size_gb:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chips-dir",
        type=Path,
        default=Path("data/1km/chips/patch128_stride64"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Apply gzip level-1 compression (smaller files, slightly slower reads)",
    )
    args = parser.parse_args()

    chips_dir: Path = args.chips_dir

    for split in args.splits:
        zarr_path = chips_dir / f"{split}.zarr.zip"
        if not zarr_path.exists():
            zarr_path = chips_dir / f"{split}.zarr"
        if not zarr_path.exists():
            print(f"Skipping {split}: no zarr store found at {chips_dir}")
            continue

        h5_path = chips_dir / f"{split}.h5"
        if h5_path.exists():
            print(f"Skipping {split}: {h5_path} already exists (delete to reconvert)")
            continue

        convert_split(zarr_path, h5_path, compress=args.compress)

    print("\nAll done.")


if __name__ == "__main__":
    main()
