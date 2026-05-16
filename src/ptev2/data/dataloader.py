"""PyTorch Dataset and DataLoader for chip stores (HDF5 or zarr)."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


def _resolve_store_path(zarr_dir: Path, split: str) -> Path:
    for suffix in (f"{split}.h5", f"{split}.zarr.zip", f"{split}.zarr"):
        p = zarr_dir / suffix
        if p.exists():
            return p
    raise FileNotFoundError(f"{split} store not found at {zarr_dir}")


def _open_zarr(zarr_path: Path) -> zarr.Group:
    if str(zarr_path).endswith(".zip"):
        return zarr.open_group(
            zarr.storage.ZipStore(str(zarr_path), mode="r"), mode="r"
        )
    return zarr.open_group(str(zarr_path), mode="r")


def _read_store_metadata(path: Path) -> tuple[int, dict]:
    """Return (n_chips, {group_path: attrs}) without holding an open handle."""
    if str(path).endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            predictors = list(f["predictors"].keys())
            n_chips = f[f"predictors/{predictors[0]}"].shape[0]
            attrs = {k: dict(v.attrs) for k, v in f.items()}
        return n_chips, attrs
    else:
        store = _open_zarr(path)
        predictors = [name for name, _ in store["predictors"].arrays()]
        n_chips = store[f"predictors/{predictors[0]}"].shape[0]
        attrs = {k: dict(store[k].attrs) for k in store}
        return n_chips, attrs


class PlantTraitDataset(Dataset):
    """PyTorch Dataset for spatially pre-extracted Earth Observation chips paired with plant trait observations.

    Each sample corresponds to one field observation location. Predictor arrays are
    concatenated along channel dimension (dim=0) to form input tensor X. Targets are
    returned as a dataset bundle, e.g. {"splot": {"y": ..., "source_mask": ...}}.

    Supports HDF5 (.h5) and zarr (.zarr.zip / .zarr) backends. The file handle is
    opened lazily per worker so DataLoader multi-processing works safely.

    Args:
        store_path: Path to the store (.h5, .zarr.zip, or .zarr directory).
        predictors: List of array names in the store to use as model inputs.
        target_layouts: Per-target-dataset layout including target/source channel indices.
    """

    def __init__(
        self,
        store_path: str | Path,
        predictors: list[str],
        target_layouts: dict[str, dict[str, Any]] | None = None,
        return_target_bundle: bool = True,
        chip_indices: np.ndarray | None = None,
    ):
        self._path = Path(store_path)
        self._is_h5 = str(self._path).endswith(".h5")
        self._handle = None  # opened lazily per worker in __getitem__

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

        n_total, _ = _read_store_metadata(self._path)
        self._all_n_chips = n_total
        if chip_indices is None:
            self.chip_indices = np.arange(self._all_n_chips, dtype=np.int64)
        else:
            self.chip_indices = np.asarray(chip_indices, dtype=np.int64)
        self.n_chips = int(self.chip_indices.shape[0])

    def _get_handle(self):
        if self._handle is None:
            if self._is_h5:
                import h5py

                self._handle = h5py.File(self._path, "r")
            else:
                self._handle = _open_zarr(self._path)
        return self._handle

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
        store = self._get_handle()
        store_idx = int(self.chip_indices[idx])
        X = torch.cat(
            [
                torch.as_tensor(np.array(store[f"predictors/{name}"][store_idx]))
                for name in self.predictors
            ],
            dim=0,
        )

        bundle: dict[str, dict[str, torch.Tensor]] = {}
        for dataset_name, layout in self.target_layouts.items():
            y_full = torch.as_tensor(
                np.array(store[f"targets/{dataset_name}"][store_idx])
            )
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
    """Get a dataloader for a given split from a chip store (HDF5 or zarr).

    Prefers .h5 over .zarr.zip over .zarr when multiple formats are present.

    Args:
        zarr_dir: Directory containing the split stores (train.h5, val.h5, …).
        split: One of 'train', 'val', or 'test'.
        predictors: List of array names in the store to use as model inputs.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes to use for data loading.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"split must be one of ['train', 'val', 'test'], got '{split}'"
        )
    store_path = _resolve_store_path(zarr_dir, split)
    n_total, _ = _read_store_metadata(store_path)

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
        store_path,
        predictors=predictors,
        target_layouts=target_layouts,
        return_target_bundle=return_target_bundle,
        chip_indices=chip_indices,
    )

    shuffle = split == "train"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader
