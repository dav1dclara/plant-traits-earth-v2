"""PyTorch Dataset and DataLoader for zarr chip stores."""

from pathlib import Path

import torch
import zarr
from torch.utils.data import DataLoader, Dataset, Subset


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
        source_value_if_missing: int = 2,
    ):
        self.store = zarr.open_group(str(zarr_path), mode="r")
        self.predictors = predictors
        self.target = target
        self.target_indices = target_indices
        self.source_indices = source_indices
        self.source_value_if_missing = int(source_value_if_missing)
        self.n_chips = self.store[f"predictors/{predictors[0]}"].shape[0]
        self._gbif_reorder_idx: torch.Tensor | None = None

        if self.target == "__merged_gbif_splot__":
            targets_group = self.store["targets"]
            band_names = list(targets_group.attrs.get("band_names", []))
            if not band_names:
                raise ValueError(
                    "targets.attrs['band_names'] is required for merged gbif/splot loading."
                )
            n_bands = len(band_names)

            gbif_arr = targets_group["gbif"]
            splot_arr = targets_group["splot"]
            gbif_files = [str(v) for v in gbif_arr.attrs.get("files", [])]
            splot_files = [str(v) for v in splot_arr.attrs.get("files", [])]
            if not gbif_files or not splot_files:
                raise ValueError(
                    "targets/gbif and targets/splot must define attrs['files'] for trait ordering."
                )
            if set(gbif_files) != set(splot_files):
                raise ValueError(
                    "gbif/splot trait sets differ; cannot merge on-the-fly safely."
                )

            n_traits = len(splot_files)
            expected_channels = n_traits * n_bands
            if (
                gbif_arr.shape[1] != expected_channels
                or splot_arr.shape[1] != expected_channels
            ):
                raise ValueError(
                    "Unexpected channel count for merged gbif/splot targets: "
                    f"expected {expected_channels}, got gbif={gbif_arr.shape[1]}, splot={splot_arr.shape[1]}."
                )

            gbif_trait_to_pos = {name: idx for idx, name in enumerate(gbif_files)}
            reorder_idx: list[int] = []
            for splot_trait_name in splot_files:
                gbif_trait_pos = gbif_trait_to_pos[splot_trait_name]
                base = gbif_trait_pos * n_bands
                reorder_idx.extend(base + band_offset for band_offset in range(n_bands))
            self._gbif_reorder_idx = torch.as_tensor(reorder_idx, dtype=torch.long)

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
        if self.target == "__merged_gbif_splot__":
            y_gbif_full = torch.as_tensor(self.store["targets/gbif"][idx])
            y_splot_full = torch.as_tensor(self.store["targets/splot"][idx])
            if self._gbif_reorder_idx is None:
                raise RuntimeError(
                    "Missing gbif reorder indices for merged gbif/splot target."
                )
            y_gbif_full = y_gbif_full[self._gbif_reorder_idx]

            y_gbif = y_gbif_full[self.target_indices]
            y_splot = y_splot_full[self.target_indices]
            valid_gbif = torch.isfinite(y_gbif)
            valid_splot = torch.isfinite(y_splot)

            # sPlot has priority where both sources are available.
            y = torch.where(valid_splot, y_splot, y_gbif)

            source_mask = torch.zeros_like(y)
            source_mask = torch.where(
                valid_gbif, torch.full_like(source_mask, 1.0), source_mask
            )
            source_mask = torch.where(
                valid_splot, torch.full_like(source_mask, 2.0), source_mask
            )
        else:
            y_full = torch.as_tensor(self.store[f"targets/{self.target}"][idx])
            y = y_full[self.target_indices]
            if self.source_indices:
                source_mask = y_full[self.source_indices]
            else:
                valid = torch.isfinite(y)
                source_mask = torch.zeros_like(y)
                source_mask = torch.where(
                    valid,
                    torch.full_like(source_mask, float(self.source_value_if_missing)),
                    source_mask,
                )
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
    source_value_if_missing: int = 2,
    split_fraction: float = 1.0,
    subset_seed: int = 0,
) -> DataLoader:
    """Get a dataloader for a given split from a zarr chip store.

    Args:
        zarr_dir: Directory containing the split zarr stores (train.zarr, val.zarr, test.zarr).
        split: One of 'train', 'val', or 'test'.
        predictors: List of array names in the store to use as model inputs.
        target: Name of the array in the store to use as the prediction target.
        target_indices: Channel indices to select from the target array as prediction targets.
        source_indices: Channel indices to select from the target array as the source mask.
        source_value_if_missing: Source code used when no dedicated source band exists.
        split_fraction: Fraction of this split to use. Must be in (0, 1].
        subset_seed: Seed used for deterministic split subsampling.
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
        source_value_if_missing=source_value_if_missing,
    )

    split_fraction = float(split_fraction)
    if not (0.0 < split_fraction <= 1.0):
        raise ValueError(
            f"split_fraction must be in (0, 1], got {split_fraction} for split '{split}'."
        )
    if split_fraction < 1.0 and len(dataset) > 0:
        n_keep = max(1, int(round(len(dataset) * split_fraction)))
        generator = torch.Generator().manual_seed(int(subset_seed))
        indices = torch.randperm(len(dataset), generator=generator)[:n_keep].tolist()
        dataset = Subset(dataset, indices)

    shuffle = split == "train"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader
