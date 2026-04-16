"""
Dataloader for Multi-Task Learning (MTL).
Adapted from STL dataloader to handle multiple targets if needed.
For now, keeps similar structure but can be extended for MTL.
"""

from pathlib import Path

import torch
import zarr
from torch.utils.data import DataLoader, Dataset


class MTLDataset(Dataset):
    """
    Dataset for MTL. Currently similar to STL, but can load multiple targets.
    Each sample: X (predictors), y (targets, can be multi-task)
    """

    def __init__(
        self, zarr_path: str | Path, predictors: list[str], targets: list[str]
    ):
        self.store = zarr.open_group(str(zarr_path), mode="r")
        self.predictors = predictors
        self.targets = targets
        self.n_chips = self.store[f"predictors/{predictors[0]}"].shape[0]

    def __len__(self) -> int:
        return self.n_chips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenate predictors along channel dim
        X = torch.cat(
            [
                torch.as_tensor(self.store[f"predictors/{name}"][idx])
                for name in self.predictors
            ],
            dim=0,
        ).to(torch.float32)

        # Replace invalid predictor values before the model sees them
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # For MTL, concatenate targets if multiple
        y_list = []
        for target in self.targets:
            y_list.append(torch.as_tensor(self.store[f"targets/{target}"][idx]))
        y = torch.cat(y_list, dim=0) if len(y_list) > 1 else y_list[0]
        y = y.to(torch.float32)

        return X, y


def get_mtl_dataloader(
    zarr_dir: Path,
    split: str,
    predictors: list[str],
    targets: list[str],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Get dataloader for MTL.
    Args:
        zarr_dir: Path to zarr stores
        split: 'train', 'val', 'test'
        predictors: List of predictor names
        targets: List of target names (for MTL, multiple possible)
        batch_size: Batch size
        num_workers: Number of workers
    """
    assert split in ["train", "val", "test"]
    zarr_path = zarr_dir / f"{split}.zarr"
    assert zarr_path.exists(), f"{zarr_path} not found"

    dataset = MTLDataset(zarr_path, predictors=predictors, targets=targets)
    shuffle = split == "train"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
