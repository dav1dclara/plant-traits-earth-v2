"""
Pack train/val/test Zarr directory stores into ZipStores for efficient transfer to Euler.
Output .zarr.zip files can be read directly by Zarr without extraction.

Usage:
    python pack_zarr_zip.py
    python pack_zarr_zip.py --chips-dir data/1km/chips/patch128_stride64
"""

import argparse
<<<<<<< HEAD
from pathlib import Path

import zarr

=======
import zipfile
from pathlib import Path

>>>>>>> 72f9b53 (...)

def pack(src_path: Path) -> None:
    dst_path = src_path.with_suffix(".zarr.zip")
    if dst_path.exists():
        print(f"Skipping {src_path.name} (already exists: {dst_path})")
        return

    print(f"Packing {src_path} → {dst_path}")
<<<<<<< HEAD
    src = zarr.open(str(src_path), mode="r")
    with zarr.storage.ZipStore(str(dst_path), mode="w") as store:
        dst = zarr.open(store, mode="w")
        zarr.copy_all(src, dst, log=print)
=======
    files = sorted(
        f for f in src_path.rglob("*") if f.is_file() and f.name != ".DS_Store"
    )
    with zipfile.ZipFile(
        str(dst_path), mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for i, file_path in enumerate(files):
            arcname = file_path.relative_to(src_path)
            zf.write(file_path, arcname)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} files written...")
>>>>>>> 72f9b53 (...)
    print(f"Done: {dst_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chips-dir", type=Path, default=Path("data/1km/chips/patch128_stride64")
    )
<<<<<<< HEAD
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
=======
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    for split in args.splits:
>>>>>>> 72f9b53 (...)
        src = args.chips_dir / f"{split}.zarr"
        if not src.exists():
            print(f"Skipping {split} (not found: {src})")
            continue
        pack(src)


if __name__ == "__main__":
    main()
