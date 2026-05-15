"""
Pack train/val/test Zarr directory stores into ZipStores for efficient transfer to Euler.
Output .zarr.zip files can be read directly by Zarr without extraction.

Usage:
    python pack_zarr_zip.py
    python pack_zarr_zip.py --chips-dir data/1km/chips/patch128_stride64
"""

import argparse
from pathlib import Path

import zarr


def pack(src_path: Path) -> None:
    dst_path = src_path.with_suffix(".zarr.zip")
    if dst_path.exists():
        print(f"Skipping {src_path.name} (already exists: {dst_path})")
        return

    print(f"Packing {src_path} → {dst_path}")
    src = zarr.open(str(src_path), mode="r")
    with zarr.storage.ZipStore(str(dst_path), mode="w") as store:
        dst = zarr.open(store, mode="w")
        zarr.copy_all(src, dst, log=print)
    print(f"Done: {dst_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chips-dir", type=Path, default=Path("data/1km/chips/patch128_stride64")
    )
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        src = args.chips_dir / f"{split}.zarr"
        if not src.exists():
            print(f"Skipping {split} (not found: {src})")
            continue
        pack(src)


if __name__ == "__main__":
    main()
