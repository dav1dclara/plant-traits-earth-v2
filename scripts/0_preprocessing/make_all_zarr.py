"""Merge train.zarr + val.zarr + test.zarr → all.zarr for global inference.

Creates all.zarr in the same directory as the three split zarrs, concatenating
every array along axis 0.  A uint8 `split_id` array records provenance
(0=train, 1=val, 2=test) so chips can be re-split later.

Works for any chips_centered patch size (patch3_stride1, patch5_stride1,
patch7_stride1). Patch size is auto-detected from the zarr attrs.

Usage
-----
    # 3×3 patches (default)
    python scripts/0_preprocessing/make_all_zarr.py

    # 5×5 patches
    python scripts/0_preprocessing/make_all_zarr.py --patch_size 5

    # 7×7 patches
    python scripts/0_preprocessing/make_all_zarr.py --patch_size 7

    # Custom path + overwrite
    python scripts/0_preprocessing/make_all_zarr.py \\
        --zarr_dir /scratch3/plant-traits-v2/data/22km/chips_centered/patch5_stride1 \\
        --overwrite
"""

import argparse
from pathlib import Path

import numpy as np
import zarr

SPLITS = ["train", "val", "test"]

PREDICTOR_NAMES = ["canopy_height", "modis", "soil_grids", "vodca", "worldclim"]

CENTER_NAMES = [
    "splot_mean",
    "splot_std",
    "splot_count",
    "splot_valid",
    "splot_q05",
    "splot_q95",
    "gbif_mean",
    "gbif_valid",
]

TARGET_NAMES = ["supervision_splot_only", "supervision_gbif_only", "supervision"]

FLAT_ARRAYS = ["bounds", "row", "col"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split zarrs into all.zarr")
    parser.add_argument(
        "--zarr_dir",
        default=None,
        help=(
            "Directory containing train.zarr, val.zarr, test.zarr. "
            "If omitted, auto-built from --patch_size."
        ),
    )
    parser.add_argument(
        "--patch_size",
        default=3,
        type=int,
        choices=[3, 5, 7],
        help="Patch size (3, 5, or 7). Used to locate zarr_dir when --zarr_dir is omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate all.zarr if it already exists.",
    )
    args = parser.parse_args()

    if args.zarr_dir:
        zarr_dir = Path(args.zarr_dir)
    else:
        zarr_dir = Path(
            f"/scratch3/plant-traits-v2/data/22km/chips_centered/patch{args.patch_size}_stride1"
        )
    out_path = zarr_dir / "all.zarr"

    if out_path.exists():
        if args.overwrite:
            import shutil

            print(f"Removing existing {out_path} ...")
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(
                f"all.zarr already exists at {out_path}.\n"
                "Use --overwrite to replace it."
            )

    # ------------------------------------------------------------------
    # Open split stores
    # ------------------------------------------------------------------
    stores: list = []
    split_sizes: list[int] = []

    for split in SPLITS:
        p = zarr_dir / f"{split}.zarr"
        if not p.exists():
            raise FileNotFoundError(f"{split}.zarr not found at {p}")
        s = zarr.open_group(str(p), mode="r")
        n = int(s["bounds"].shape[0])
        stores.append(s)
        split_sizes.append(n)
        print(f"  {split:5s}.zarr : {n:,} chips")

    N_total = sum(split_sizes)
    print(
        f"\nTotal chips: {N_total:,}  (train={split_sizes[0]:,}  val={split_sizes[1]:,}  test={split_sizes[2]:,})"
    )

    # ------------------------------------------------------------------
    # Build attrs from train.zarr template
    # ------------------------------------------------------------------
    ref_attrs = dict(stores[0].attrs)
    new_attrs = {k: v for k, v in ref_attrs.items() if k != "split"}
    new_attrs.update(
        {
            "split": "all",
            "n_train": split_sizes[0],
            "n_val": split_sizes[1],
            "n_test": split_sizes[2],
            "n_total": N_total,
            "split_id_map": {"0": "train", "1": "val", "2": "test"},
        }
    )

    # ------------------------------------------------------------------
    # Create output zarr
    # ------------------------------------------------------------------
    print(f"\nWriting {out_path} ...")
    out = zarr.open_group(str(out_path), mode="w")
    out.attrs.update(new_attrs)

    def _write(group, name: str, data: np.ndarray, chunks: tuple) -> None:
        """Create a zarr array and populate it (zarr v3 compatible)."""
        arr = group.require_array(
            name, shape=data.shape, chunks=chunks, dtype=data.dtype
        )
        arr[:] = data

    # ---- flat arrays (bounds, row, col) --------------------------------
    for name in FLAT_ARRAYS:
        print(f"  {name} ...", end="  ", flush=True)
        data = np.concatenate([s[name][:] for s in stores], axis=0)
        chunk0 = min(4096, N_total)
        chunks = (chunk0,) + data.shape[1:]
        _write(out, name, data, chunks)
        print(data.shape)

    # ---- split provenance (0=train, 1=val, 2=test) --------------------
    split_id = np.concatenate(
        [np.full(n, i, dtype=np.uint8) for i, n in enumerate(split_sizes)]
    )
    _write(out, "split_id", split_id, (min(4096, N_total),))
    print(f"  split_id ...  {split_id.shape}")

    # ---- predictors ---------------------------------------------------
    for pname in PREDICTOR_NAMES:
        key = f"predictors/{pname}"
        print(f"  {key} ...", end="  ", flush=True)
        data = np.concatenate([s[key][:] for s in stores], axis=0)
        C, H, W = data.shape[1], data.shape[2], data.shape[3]
        _write(out, key, data, (256, C, H, W))
        print(data.shape)

    # ---- targets ------------------------------------------------------
    for tname in TARGET_NAMES:
        key = f"targets/{tname}"
        print(f"  {key} ...", end="  ", flush=True)
        data = np.concatenate([s[key][:] for s in stores], axis=0)
        _write(out, key, data, (256, data.shape[1], data.shape[2], data.shape[3]))
        print(data.shape)

    # ---- center statistics --------------------------------------------
    for cname in CENTER_NAMES:
        key = f"center/{cname}"
        print(f"  {key} ...", end="  ", flush=True)
        data = np.concatenate([s[key][:] for s in stores], axis=0)
        _write(out, key, data, (min(1024, N_total), data.shape[1]))
        print(data.shape)

    print(f"\nDone!  all.zarr → {out_path}")
    print(
        f"  {N_total:,} chips total  |  patch_size=3  |  EPSG:{new_attrs['crs_epsg']}"
    )


if __name__ == "__main__":
    main()
