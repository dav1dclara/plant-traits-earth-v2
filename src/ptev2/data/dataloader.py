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


def _read_bounds(path: Path) -> np.ndarray:
    """Read chip bounds without keeping an open store handle."""
    if str(path).endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            return np.asarray(f["bounds"][:])

    store = _open_zarr(path)
    return np.asarray(store["bounds"][:])


def _sample_random_indices(n_total: int, n_keep: int, subset_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(subset_seed))
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def _allocate_remaining_quota(
    quotas: np.ndarray,
    capacities: np.ndarray,
    n_remaining: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_remaining <= 0:
        return quotas

    total_capacity = int(capacities.sum())
    if total_capacity <= 0:
        return quotas

    expected = capacities * (n_remaining / total_capacity)
    extra = np.floor(expected).astype(np.int64)
    extra = np.minimum(extra, capacities)
    quotas += extra
    n_remaining -= int(extra.sum())

    if n_remaining <= 0:
        return quotas

    remainders = expected - extra
    candidates = np.flatnonzero((capacities - extra) > 0)
    if candidates.size == 0:
        return quotas

    # Shuffle before stable sorting so equal remainders are resolved deterministically
    # by subset_seed, not by cell id.
    shuffled = rng.permutation(candidates)
    order = shuffled[np.argsort(-remainders[shuffled], kind="stable")]
    quotas[order[:n_remaining]] += 1
    return quotas


def _sample_spatial_grid_indices(
    store_path: Path,
    n_total: int,
    n_keep: int,
    subset_seed: int,
    spatial_grid_size: int,
) -> np.ndarray:
    bounds = _read_bounds(store_path)
    if bounds.shape[0] != n_total or bounds.shape[1] != 4:
        raise ValueError(
            f"Expected bounds with shape ({n_total}, 4), got {bounds.shape}."
        )

    centers_x = (bounds[:, 0] + bounds[:, 2]) / 2.0
    centers_y = (bounds[:, 1] + bounds[:, 3]) / 2.0
    valid = np.isfinite(centers_x) & np.isfinite(centers_y)
    if not np.all(valid):
        raise ValueError("Spatial subset sampling requires finite chip bounds.")

    grid_size = int(spatial_grid_size)
    if grid_size < 1:
        raise ValueError(f"spatial_grid_size must be >= 1, got {grid_size}.")

    x_edges = np.linspace(float(centers_x.min()), float(centers_x.max()), grid_size + 1)
    y_edges = np.linspace(float(centers_y.min()), float(centers_y.max()), grid_size + 1)
    x_bins = np.searchsorted(x_edges[1:-1], centers_x, side="right")
    y_bins = np.searchsorted(y_edges[1:-1], centers_y, side="right")
    cell_ids = y_bins * grid_size + x_bins

    unique_cells, inverse, counts = np.unique(
        cell_ids, return_inverse=True, return_counts=True
    )
    rng = np.random.default_rng(int(subset_seed))
    quotas = np.zeros(unique_cells.shape[0], dtype=np.int64)

    if n_keep >= unique_cells.shape[0]:
        quotas[:] = 1
        capacities = counts - quotas
        quotas = _allocate_remaining_quota(
            quotas=quotas,
            capacities=capacities,
            n_remaining=n_keep - int(quotas.sum()),
            rng=rng,
        )
    else:
        selected_cells = rng.choice(unique_cells.shape[0], size=n_keep, replace=False)
        quotas[selected_cells] = 1

    selected_indices = []
    all_indices = np.arange(n_total, dtype=np.int64)
    for cell_pos, quota in enumerate(quotas):
        if quota <= 0:
            continue
        members = all_indices[inverse == cell_pos]
        selected_indices.append(rng.choice(members, size=int(quota), replace=False))

    return np.sort(np.concatenate(selected_indices).astype(np.int64))


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
    subset_strategy: str = "random",
    spatial_grid_size: int = 24,
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
        subset_strategy = str(subset_strategy)
        if subset_strategy == "random":
            chip_indices = _sample_random_indices(n_total, n_keep, subset_seed)
        elif subset_strategy == "spatial_grid":
            chip_indices = _sample_spatial_grid_indices(
                store_path=store_path,
                n_total=n_total,
                n_keep=n_keep,
                subset_seed=subset_seed,
                spatial_grid_size=spatial_grid_size,
            )
        else:
            raise ValueError(
                "subset_strategy must be one of ['random', 'spatial_grid'], "
                f"got '{subset_strategy}'."
            )
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
