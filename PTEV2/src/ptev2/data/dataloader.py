"""PyTorch Dataset and DataLoader for zarr chip stores."""

from pathlib import Path

import torch
import zarr
from torch.utils.data import DataLoader, Dataset


class PlantTraitDataset(Dataset):
    """PyTorch Dataset for spatially pre-extracted Earth Observation chips paired with plant trait observations.

    Each sample corresponds to one field observation location. The zarr store is expected to contain
    one array per predictor (e.g. 'canopy_height', 'modis', 'worldclim') and one array for the target
    trait (e.g. 'gbif'). All predictor arrays are concatenated along the channel dimension (dim=0)
    to form a single input tensor X, while the target array provides the label tensor y.

    Args:
        zarr_path: Path to the zarr group store containing predictor and target arrays.
        predictors: List of array names in the store to use as model inputs.
        target: Name of the array in the store to use as the prediction target.
        target_indices: Channel indices to select from the target array as prediction targets.
        source_indices: Channel indices to select from the target array as the source mask.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        predictors: list[str],
        target: str,
        target_indices: list[int],
        source_indices: list[int],
    ):
        self.store = zarr.open_group(str(zarr_path), mode="r")
        self.predictors = predictors
        self.target = target
        self.target_indices = target_indices
        self.source_indices = source_indices
        self.n_chips = self.store[f"predictors/{predictors[0]}"].shape[0]

    def __len__(self) -> int:
        return self.n_chips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = torch.cat(
            [
                torch.as_tensor(self.store[f"predictors/{name}"][idx])
                for name in self.predictors
            ],
            dim=0,
        )
        y_full = torch.as_tensor(self.store[f"targets/{self.target}"][idx])
        y = y_full[self.target_indices]
        source_mask = y_full[self.source_indices]
        return X, y, source_mask


def get_dataloader(
    zarr_dir: Path,
    split: str,
    predictors: list[str],
    target: str,
    target_indices: list[int],
    source_indices: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Get a dataloader for a given split from a zarr chip store.

    Args:
        zarr_dir: Directory containing the split zarr stores (train.zarr, val.zarr, test.zarr).
        split: One of 'train', 'val', or 'test'.
        predictors: List of array names in the store to use as model inputs.
        target: Name of the array in the store to use as the prediction target.
        target_indices: Channel indices to select from the target array as prediction targets.
        source_indices: Channel indices to select from the target array as the source mask.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes to use for data loading.
    """
    assert split in ["train", "val", "test"], (
        f"split must be one of ['train', 'val', 'test'], got '{split}'"
    )
    zarr_path = zarr_dir / f"{split}.zarr"
    assert zarr_path.exists(), f"{split} zarr store not found at {zarr_path}"

    dataset = PlantTraitDataset(
        zarr_path,
        predictors=predictors,
        target=target,
        target_indices=target_indices,
        source_indices=source_indices,
    )

    shuffle = split == "train"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader
