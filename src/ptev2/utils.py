import random
from pathlib import Path

import numpy as np
import torch
import zarr
from omegaconf import DictConfig, OmegaConf


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    model_cfg = cfg.models
    if OmegaConf.select(model_cfg, "_target_") is not None:
        return model_cfg

    active_cfg = OmegaConf.select(model_cfg, "active")
    if active_cfg is not None and OmegaConf.select(active_cfg, "_target_") is not None:
        return active_cfg

    raise ValueError(
        "Model config must define '_target_' either at cfg.models._target_ or cfg.models.active._target_."
    )


def resolve_train_zarr_dir(cfg: DictConfig) -> Path:
    zarr_override = OmegaConf.select(cfg, "data.zarr_dir")
    if zarr_override:
        return Path(str(zarr_override))

    chips_subdir = str(OmegaConf.select(cfg, "data.chips_subdir") or "chips")
    return (
        Path(str(cfg.data.root_dir))
        / f"{cfg.data.resolution_km}km"
        / chips_subdir
        / f"patch{cfg.data.patch_size}_stride{cfg.data.stride}"
    )


def build_target_layout(
    train_cfg: DictConfig,
    train_store: zarr.Group,
) -> tuple[str, list[str], list[str], list[int], list[int]]:
    target_cfg = train_cfg.data.targets
    target_dataset = str(target_cfg.dataset)

    zarr_dataset_names = list(train_store["targets"].keys())
    if target_dataset not in zarr_dataset_names:
        raise ValueError(
            f"Dataset '{target_dataset}' not found in zarr. Available: {', '.join(zarr_dataset_names)}"
        )

    zarr_band_names = [str(v) for v in train_store["targets"].attrs["band_names"]]
    band_to_idx = {name: idx for idx, name in enumerate(zarr_band_names)}

    trait_names_attr = train_store["targets"].attrs.get("trait_names")
    if trait_names_attr is not None:
        zarr_all_traits = [str(v) for v in trait_names_attr]
    else:
        zarr_all_traits = [
            str(f).replace("X", "").replace(".tif", "")
            for f in train_store[f"targets/{target_dataset}"].attrs["files"]
        ]
    traits = (
        [str(v) for v in target_cfg.traits] if target_cfg.traits else zarr_all_traits
    )
    if not traits:
        raise ValueError("data.targets.traits must not be empty after resolution.")

    cfg_bands = [str(v) for v in target_cfg.bands]
    if not cfg_bands:
        raise ValueError("data.targets.bands must not be empty.")

    n_bands = len(zarr_band_names)
    target_indices = [
        trait_pos * n_bands + band_to_idx[band]
        for trait_pos in range(len(traits))
        for band in cfg_bands
    ]
    source_indices = [
        trait_pos * n_bands + band_to_idx["source"] for trait_pos in range(len(traits))
    ]

    return target_dataset, traits, cfg_bands, target_indices, source_indices


def resolve_eval_source_value(target_dataset: str) -> int:
    """Return source code used for evaluation metrics/masking.

    Source convention:
      1 = GBIF
      2 = sPlot
    """
    dataset = str(target_dataset).lower()
    if dataset == "supervision_gbif_only":
        return 1
    return 2


def predict_batch(
    model: torch.nn.Module, X: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_x = torch.isfinite(X).all(dim=1, keepdim=True)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = torch.clamp(X, min=-1e4, max=1e4)
    return model(X), valid_x
