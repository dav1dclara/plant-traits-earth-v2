"""Benchmark the dataloader: compare HDF5 vs zarr throughput over N batches.

Usage:
    python scripts/1km/benchmark_dataloader.py
    python scripts/1km/benchmark_dataloader.py --backend zarr --num-workers 4
    python scripts/1km/benchmark_dataloader.py --splits train --batch-size 64 --max-batches 50
    python scripts/1km/benchmark_dataloader.py --compare  # runs both h5 and zarr side-by-side
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    max_batches: int | None = None,
) -> dict:
    dataset = PlantTraitDataset(
        store_path,
        predictors=predictors,
        target_layouts=target_layouts,
        return_target_bundle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    limit = min(max_batches, len(loader)) if max_batches else len(loader)
    desc = f"{store_path.suffix.lstrip('.')}:{store_path.stem}"
    x_shape = None
    loader_iter = iter(loader)
    t0 = time.perf_counter()
    for i in tqdm(range(limit), desc=desc, total=limit, unit="batch"):
        X, _bundle = next(loader_iter)
        if i == 0:
            x_shape = tuple(X.shape)
    elapsed = time.perf_counter() - t0
    del loader_iter  # signal workers to stop outside the timed block
    del loader
    gc.collect()  # ensure worker processes are reaped immediately

    return {
        "n_chips": limit * batch_size,
        "n_batches": limit,
        "x_shape": x_shape,
        "elapsed_s": elapsed,
        "s_per_batch": elapsed / limit,
        "chips_per_s": (limit * batch_size) / elapsed,
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
        "--max-batches",
        type=int,
        default=100,
        help="Stop after this many batches per split (0 = full epoch).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "h5", "zarr"],
        default="auto",
        help="Storage backend. 'auto' prefers .h5 over .zarr.zip over .zarr.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both h5 and zarr backends and print a side-by-side comparison.",
    )
    args = parser.parse_args()
    args.max_batches = args.max_batches or None

    backends = ["h5", "zarr"] if args.compare else [args.backend]
    max_b = args.max_batches
    print(
        f"batch_size={args.batch_size}  |  num_workers={args.num_workers}  |  "
        f"max_batches={max_b or 'full'}\n"
    )

    all_results: dict[str, dict[str, dict]] = {}  # backend -> split -> metrics
    for backend in backends:
        all_results[backend] = {}
        for split in args.splits:
            try:
                store_path, predictors, target_layouts = _get_metadata(
                    args.chips_dir, split, backend
                )
            except FileNotFoundError as e:
                print(f"[{backend}/{split}] skipped: {e}")
                continue

            r = benchmark_split(
                store_path,
                predictors,
                target_layouts,
                args.batch_size,
                args.num_workers,
                max_b,
            )
            all_results[backend][split] = r

    print(
        f"\n{'backend':<8} {'split':<8} {'batches':>8} {'total(s)':>10} {'s/batch':>9} {'chips/s':>9}"
    )
    print("-" * 58)
    for backend, splits in all_results.items():
        for split, r in splits.items():
            print(
                f"{backend:<8} {split:<8} {r['n_batches']:>8,} "
                f"{r['elapsed_s']:>10.1f} {r['s_per_batch']:>9.3f} {r['chips_per_s']:>9.0f}"
            )

    if (
        args.compare
        and len(all_results.get("h5", {}))
        and len(all_results.get("zarr", {}))
    ):
        print("\n--- Speedup (h5 vs zarr) ---")
        for split in args.splits:
            if split in all_results["h5"] and split in all_results["zarr"]:
                speedup = (
                    all_results["zarr"][split]["s_per_batch"]
                    / all_results["h5"][split]["s_per_batch"]
                )
                print(f"  {split}: {speedup:.1f}x faster with h5")


if __name__ == "__main__":
    main()
