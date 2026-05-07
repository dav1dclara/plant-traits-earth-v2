"""PyTorch Dataset and DataLoader for zarr chip stores."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


class PlantTraitDataset(Dataset):
    """PyTorch Dataset for spatially pre-extracted Earth Observation chips paired with plant trait observations.

    Each sample corresponds to one field observation location. Predictor arrays are
    concatenated along channel dimension (dim=0) to form input tensor X. Targets are
    returned as a dataset bundle, e.g. {"splot": {"y": ..., "source_mask": ...}}.

    Args:
        zarr_path: Path to the zarr group store containing predictor and target arrays.
        predictors: List of array names in the store to use as model inputs.
        target_layouts: Per-target-dataset layout including target/source channel indices.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        predictors: list[str],
        target_layouts: dict[str, dict[str, Any]] | None = None,
        return_target_bundle: bool = True,
        chip_indices: np.ndarray | None = None,
    ):
        self.store = zarr.open_group(str(zarr_path), mode="r")
        self.predictors = predictors
        self.target_layouts = target_layouts or {}
        self.return_target_bundle = bool(return_target_bundle)

        if not self.return_target_bundle:
            raise ValueError(
                "Legacy single-target mode was removed; use target bundles."
            )
        if not self.target_layouts:
            raise ValueError(
                "target_layouts must be provided when return_target_bundle=True."
            )
        self._all_n_chips = self.store[f"predictors/{predictors[0]}"].shape[0]
        if chip_indices is None:
            self.chip_indices = np.arange(self._all_n_chips, dtype=np.int64)
        else:
            self.chip_indices = np.asarray(chip_indices, dtype=np.int64)
        self.n_chips = int(self.chip_indices.shape[0])

    def __len__(self) -> int:
        return self.n_chips

    def _default_source_mask(self, y: torch.Tensor, dataset_name: str) -> torch.Tensor:
        valid = torch.isfinite(y)
        source_value = 2.0 if dataset_name == "splot" else 1.0
        return torch.where(
            valid,
            torch.full_like(y, source_value),
            torch.zeros_like(y),
        )

    def __getitem__(self, idx: int):
        store_idx = int(self.chip_indices[idx])
        X = torch.cat(
            [
                torch.as_tensor(self.store[f"predictors/{name}"][store_idx])
                for name in self.predictors
            ],
            dim=0,
        )

        bundle: dict[str, dict[str, torch.Tensor]] = {}
        for dataset_name, layout in self.target_layouts.items():
            y_full = torch.as_tensor(self.store[f"targets/{dataset_name}"][store_idx])
            trait_positions = layout.get("trait_positions") or layout["target_indices"]
            y = y_full[trait_positions]
            source_indices = layout.get("source_indices") or []
            if source_indices and max(source_indices) < y_full.shape[0]:
                source_mask = y_full[source_indices]
            else:
                source_mask = self._default_source_mask(y, dataset_name)
            bundle[dataset_name] = {"y": y, "source_mask": source_mask}
        return X, bundle


def get_dataloader(
    zarr_dir: Path,
    split: str,
    predictors: list[str],
    batch_size: int,
    num_workers: int,
    target_layouts: dict[str, dict[str, Any]] | None = None,
    return_target_bundle: bool = True,
    split_fraction: float = 1.0,
    subset_seed: int = 0,
) -> DataLoader:
    """Get a dataloader for a given split from a zarr chip store.

    Args:
        zarr_dir: Directory containing the split zarr stores (train.zarr, val.zarr, test.zarr).
        split: One of 'train', 'val', or 'test'.
        predictors: List of array names in the store to use as model inputs.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes to use for data loading.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"split must be one of ['train', 'val', 'test'], got '{split}'"
        )
    zarr_path = zarr_dir / f"{split}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"{split} zarr store not found at {zarr_path}")

    base_store = zarr.open_group(str(zarr_path), mode="r")
    n_total = int(base_store[f"predictors/{predictors[0]}"].shape[0])
    split_fraction = float(split_fraction)
    if not (0.0 < split_fraction <= 1.0):
        raise ValueError(f"split_fraction must be in (0, 1], got {split_fraction}.")
    if split_fraction < 1.0:
        n_keep = max(1, int(round(n_total * split_fraction)))
        rng = np.random.default_rng(int(subset_seed))
        chip_indices = np.sort(rng.choice(n_total, size=n_keep, replace=False))
    else:
        chip_indices = None

    dataset = PlantTraitDataset(
        zarr_path,
        predictors=predictors,
        target_layouts=target_layouts,
        return_target_bundle=return_target_bundle,
        chip_indices=chip_indices,
    )

    shuffle = split == "train"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader
