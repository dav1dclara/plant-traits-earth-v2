"""PyTorch Dataset and DataLoader for zarr chip stores."""

from pathlib import Path

import torch
import zarr
from torch.utils.data import Dataset


class ChipDataset(Dataset):
    def __init__(self, zarr_path: str | Path, predictors: list[str], target: str):
        self.store = zarr.open_group(str(zarr_path), mode="r")
        self.predictors = predictors
        self.target = target
        self.n_samples = self.store[predictors[0]].shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.cat(
            [torch.as_tensor(self.store[name][idx]) for name in self.predictors], dim=0
        )
        y = torch.as_tensor(self.store[self.target][idx])
        return X, y


def get_dataloader(zarr_path):
    # TODO: create dataset

    # TODO: remove this later, only used for testing
    store = zarr.open_group(str(zarr_path), mode="r")

    for name, arr in store.arrays():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}")

    print("\nAttributes:")
    for k, v in store.attrs.items():
        print(f"  {k}: {v}")

    print(f"Arrays: {list(store.keys())}\n")


def get_train_dataloader(zarr_path, batch_size):
    """Get train dataloader from zarr chip store."""
    assert (zarr_path / "train.zarr").exists(), (
        f"Train zarr store not found at {zarr_path / 'train.zarr'}"
    )
    print(f"Getting train data loader from {zarr_path}...")


def get_val_dataloader(zarr_path, batch_size):
    """Get validation dataloader from zarr chip store."""
    assert (zarr_path / "val.zarr").exists(), (
        f"Val zarr store not found at {zarr_path / 'val.zarr'}"
    )
    print(f"Getting val data loader from {zarr_path}...")


def get_test_dataloader(zarr_path, batch_size):
    """Get test dataloader from zarr chip store."""
    assert (zarr_path / "test.zarr").exists(), (
        f"Test zarr store not found at {zarr_path / 'test.zarr'}"
    )
    print(f"Getting test data loader from {zarr_path}...")
