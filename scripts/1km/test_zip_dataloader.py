"""Smoke-test the zipped test store: open it, print array info, iterate a few batches."""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ptev2.data.dataloader import PlantTraitDataset, _open_zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chips-dir",
        type=Path,
        default=Path("data/1km/chips/patch128_stride64"),
    )
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    zarr_path = args.chips_dir / "test.zarr.zip"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zipped test store not found: {zarr_path}")
    print(f"Opening: {zarr_path}")
    store = _open_zarr(zarr_path)

    print("\n--- Predictors ---")
    predictors = []
    for name, arr in store["predictors"].arrays():
        print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}")
        predictors.append(name)

    print("\n--- Targets ---")
    target_names = []
    for name, arr in store["targets"].arrays():
        print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}")
        target_names.append(name)

    band_names = list(store["targets"].attrs.get("band_names", []))
    mean_indices = [i for i, b in enumerate(band_names) if b == "mean"]
    print(
        f"\nBand names ({len(band_names)}): {band_names[:5]}{'...' if len(band_names) > 5 else ''}"
    )
    print(
        f"'mean' band indices: {mean_indices[:5]}{'...' if len(mean_indices) > 5 else ''}"
    )

    target_layouts = {
        name: {"trait_positions": mean_indices, "target_indices": mean_indices}
        for name in target_names
    }

    dataset = PlantTraitDataset(
        zarr_path,
        predictors=predictors,
        target_layouts=target_layouts,
        return_target_bundle=True,
    )
    print(f"\nDataset: {len(dataset):,} chips")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(
        f"\nIterating {args.n_batches} batches (batch_size={args.batch_size}, num_workers={args.num_workers})..."
    )
    t0 = time.perf_counter()
    for i, (X, bundle) in enumerate(loader):
        if i == 0:
            print(f"  X shape:  {tuple(X.shape)}  dtype={X.dtype}")
            for name, payload in bundle.items():
                print(f"  {name}/y shape: {tuple(payload['y'].shape)}")
        if i + 1 >= args.n_batches:
            break
    elapsed = time.perf_counter() - t0
    print(
        f"\nLoaded {args.n_batches} batches in {elapsed:.2f}s  ({elapsed / args.n_batches:.3f}s/batch)"
    )
    print("\nOK")


if __name__ == "__main__":
    main()
