"""Dataset/DataLoader helpers for split zarr chip stores."""

from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset, Subset

VALID_SPLITS = ("train", "val", "test")


class PlantTraitDataset(Dataset):
    """Read one split zarr and return `(X, y, source_mask)` per chip."""

    def __init__(
        self,
        zarr_path: str | Path,
        predictors: list[str],
        target: str,
        target_indices: list[int],
        source_indices: list[int],
        add_group_validity_masks: bool = False,
        validity_mask_groups: list[str] | None = None,
    ):
        self.store = zarr.open_group(str(zarr_path), mode="r")
        self.predictors = predictors
        self.target = target
        self.target_indices = target_indices
        self.source_indices = source_indices
        self.add_group_validity_masks = bool(add_group_validity_masks)
        if validity_mask_groups:
            self.validity_mask_groups = [
                group for group in validity_mask_groups if group in predictors
            ]
        else:
            self.validity_mask_groups = list(predictors)
        self.n_chips = self.store[f"predictors/{predictors[0]}"].shape[0]

    def __len__(self) -> int:
        return self.n_chips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictor_tensors = []
        validity_masks = []
        for name in self.predictors:
            arr = np.asarray(self.store[f"predictors/{name}"][idx], dtype=np.float32)
            tensor = torch.as_tensor(arr)
            predictor_tensors.append(tensor)
            if self.add_group_validity_masks and name in self.validity_mask_groups:
                valid = np.isfinite(arr).all(axis=0, keepdims=True).astype(np.float32)
                validity_masks.append(torch.as_tensor(valid))

        X = torch.cat(predictor_tensors, dim=0)
        if validity_masks:
            X = torch.cat([X, *validity_masks], dim=0)
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
    add_group_validity_masks: bool = False,
    validity_mask_groups: list[str] | None = None,
    split_fraction: float = 1.0,
    split_seed: int = 0,
) -> DataLoader:
    """Build DataLoader for one split (`train|val|test`) with optional subsampling."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {list(VALID_SPLITS)}, got '{split}'.")
    zarr_path = zarr_dir / f"{split}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"{split} zarr store not found at {zarr_path}")

    dataset = PlantTraitDataset(
        zarr_path,
        predictors=predictors,
        target=target,
        target_indices=target_indices,
        source_indices=source_indices,
        add_group_validity_masks=add_group_validity_masks,
        validity_mask_groups=validity_mask_groups,
    )
    split_fraction = float(split_fraction)
    if split_fraction <= 0.0 or split_fraction > 1.0:
        raise ValueError(
            f"split_fraction must be in (0, 1], got {split_fraction} for split='{split}'."
        )
    if split_fraction < 1.0 and len(dataset) > 0:
        n_keep = max(1, int(round(len(dataset) * split_fraction)))
        rng = np.random.default_rng(int(split_seed))
        indices = rng.choice(len(dataset), size=n_keep, replace=False)
        indices = np.sort(indices)
        dataset = Subset(dataset, indices.tolist())

    shuffle = split == "train"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader
