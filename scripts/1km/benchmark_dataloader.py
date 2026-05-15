"""Benchmark the dataloader across all splits for a full epoch.

Iterates every batch in train, val, and test to measure realistic throughput.

Usage:
    python scripts/1km/benchmark_dataloader.py
    python scripts/1km/benchmark_dataloader.py --backend zarr --num-workers 4
    python scripts/1km/benchmark_dataloader.py --splits train --batch-size 64
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ptev2.data.dataloader import PlantTraitDataset, _open_zarr, _resolve_store_path


def _get_metadata(chips_dir: Path, split: str, backend: str):
    """Return (store_path, predictors, target_layouts, band_names)."""
    if backend == "auto":
        store_path = _resolve_store_path(chips_dir, split)
    elif backend == "h5":
        store_path = chips_dir / f"{split}.h5"
        if not store_path.exists():
            raise FileNotFoundError(store_path)
    else:
        for suffix in (f"{split}.zarr.zip", f"{split}.zarr"):
            p = chips_dir / suffix
            if p.exists():
                store_path = p
                break
        else:
            raise FileNotFoundError(f"No zarr store for {split} in {chips_dir}")

    # Read metadata to discover predictors and targets
    if str(store_path).endswith(".h5"):
        import h5py

        with h5py.File(store_path, "r") as f:
            predictors = list(f["predictors"].keys())
            target_names = list(f["targets"].keys())
            band_names = list(f["targets"].attrs.get("band_names", []))
    else:
        grp = _open_zarr(store_path)
        predictors = [name for name, _ in grp["predictors"].arrays()]
        target_names = [name for name, _ in grp["targets"].arrays()]
        band_names = list(grp["targets"].attrs.get("band_names", []))

    mean_indices = [i for i, b in enumerate(band_names) if b == "mean"]
    target_layouts = {
        name: {"trait_positions": mean_indices, "target_indices": mean_indices}
        for name in target_names
    }
    return store_path, predictors, target_layouts


def benchmark_split(
    store_path: Path,
    predictors: list,
    target_layouts: dict,
    batch_size: int,
    num_workers: int,
) -> dict:
    dataset = PlantTraitDataset(
        store_path,
        predictors=predictors,
        target_layouts=target_layouts,
        return_target_bundle=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    n_batches = len(loader)
    t0 = time.perf_counter()
    for i, (X, _bundle) in enumerate(loader):
        if i == 0:
            x_shape = tuple(X.shape)
    elapsed = time.perf_counter() - t0

    return {
        "n_chips": len(dataset),
        "n_batches": n_batches,
        "x_shape": x_shape,
        "elapsed_s": elapsed,
        "s_per_batch": elapsed / n_batches,
        "chips_per_s": len(dataset) / elapsed,
    }


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=["auto", "h5", "zarr"],
        default="auto",
        help="Storage backend. 'auto' prefers .h5 over .zarr.zip over .zarr.",
    )
    args = parser.parse_args()

    print(
        f"Backend: {args.backend}  |  batch_size={args.batch_size}  |  num_workers={args.num_workers}\n"
    )

    results = {}
    for split in args.splits:
        try:
            store_path, predictors, target_layouts = _get_metadata(
                args.chips_dir, split, args.backend
            )
        except FileNotFoundError as e:
            print(f"[{split}] skipped: {e}")
            continue

        backend_label = store_path.suffix.lstrip(".")
        print(f"[{split}]  {store_path.name}  ({backend_label})")
        r = benchmark_split(
            store_path, predictors, target_layouts, args.batch_size, args.num_workers
        )
        results[split] = r
        print(
            f"  {r['n_chips']:,} chips  |  {r['n_batches']:,} batches  |  "
            f"x_shape={r['x_shape']}  |  "
            f"{r['elapsed_s']:.1f}s total  |  "
            f"{r['s_per_batch']:.3f}s/batch  |  "
            f"{r['chips_per_s']:.0f} chips/s"
        )

    if len(results) > 1:
        print("\n--- Summary ---")
        print(
            f"{'split':<8} {'chips':>8} {'batches':>8} {'total(s)':>10} {'s/batch':>9} {'chips/s':>9}"
        )
        for split, r in results.items():
            print(
                f"{split:<8} {r['n_chips']:>8,} {r['n_batches']:>8,} "
                f"{r['elapsed_s']:>10.1f} {r['s_per_batch']:>9.3f} {r['chips_per_s']:>9.0f}"
            )


if __name__ == "__main__":
    main()
