import random
from pathlib import Path
from typing import Any

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
    zarr_dir_override = OmegaConf.select(cfg, "data.zarr_dir")
    if zarr_dir_override:
        return Path(str(zarr_dir_override))
    chips_dirname = str(OmegaConf.select(cfg, "data.chips_dirname") or "chips")
    return (
        Path(str(cfg.data.root_dir))
        / f"{cfg.data.resolution_km}km"
        / chips_dirname
        / f"patch{cfg.data.patch_size}_stride{cfg.data.stride}"
    )


def build_target_layout(
    train_cfg: DictConfig,
    train_store: zarr.Group,
) -> dict[str, Any]:
    target_cfg = train_cfg.data.targets
    target_mode = str(OmegaConf.select(target_cfg, "mode") or "splot_only")
    primary_dataset = str(OmegaConf.select(target_cfg, "primary_dataset") or "splot")
    auxiliary_dataset = str(OmegaConf.select(target_cfg, "auxiliary_dataset") or "gbif")
    eval_dataset = str(OmegaConf.select(target_cfg, "eval_dataset") or primary_dataset)

    if target_mode == "splot_only":
        active_datasets = [primary_dataset]
    elif target_mode in {"dual_source_comb_single_head", "dual_source_single_head"}:
        active_datasets = [primary_dataset, auxiliary_dataset]
    else:
        raise ValueError(
            f"Unsupported data.targets.mode='{target_mode}'. "
            "Use one of ['splot_only', 'dual_source_single_head', 'dual_source_comb_single_head']."
        )

    zarr_dataset_names = list(train_store["targets"].keys())

    for ds_name in active_datasets + [eval_dataset]:
        if ds_name not in zarr_dataset_names:
            raise ValueError(
                f"Dataset '{ds_name}' not found in zarr. Available: {', '.join(zarr_dataset_names)}"
            )

    zarr_band_names = [str(v) for v in train_store["targets"].attrs["band_names"]]
    band_to_idx = {name: idx for idx, name in enumerate(zarr_band_names)}

    cfg_bands = [str(v) for v in target_cfg.bands]
    if not cfg_bands:
        raise ValueError("data.targets.bands must not be empty.")
    missing_bands = [b for b in cfg_bands if b not in band_to_idx]
    if missing_bands:
        raise ValueError(
            f"Requested target bands not found in zarr target band_names: {missing_bands}"
        )

    def _dataset_traits(dataset_name: str) -> list[str]:
        return [
            str(f).replace("X", "").replace(".tif", "")
            for f in train_store[f"targets/{dataset_name}"].attrs["files"]
        ]

    eval_dataset_traits = _dataset_traits(eval_dataset)
    traits = (
        [str(v) for v in target_cfg.traits]
        if target_cfg.traits
        else eval_dataset_traits
    )
    if not traits:
        raise ValueError("data.targets.traits must not be empty after resolution.")

    n_bands = len(zarr_band_names)
    has_source_band = "source" in band_to_idx

    dataset_layouts: dict[str, dict[str, Any]] = {}
    for dataset_name in active_datasets + [eval_dataset]:
        all_traits = _dataset_traits(dataset_name)
        trait_to_pos = {trait_name: i for i, trait_name in enumerate(all_traits)}
        missing_traits = [trait for trait in traits if trait not in trait_to_pos]
        if missing_traits:
            raise ValueError(
                f"Dataset '{dataset_name}' misses requested traits: {missing_traits}"
            )

        trait_positions = [trait_to_pos[trait] for trait in traits]
        target_indices = [
            pos * n_bands + band_to_idx[band]
            for pos in trait_positions
            for band in cfg_bands
        ]
        if len(target_indices) != len(traits) * len(cfg_bands):
            raise ValueError(
                f"Index layout mismatch for dataset '{dataset_name}': expected "
                f"{len(traits) * len(cfg_bands)} channels, got {len(target_indices)}."
            )

        source_indices = (
            [pos * n_bands + band_to_idx["source"] for pos in trait_positions]
            if has_source_band
            else []
        )

        dataset_layouts[dataset_name] = {
            "dataset": dataset_name,
            "all_traits": all_traits,
            "trait_positions": trait_positions,
            "target_indices": target_indices,
            "source_indices": source_indices,
        }

    return {
        "mode": target_mode,
        "traits": traits,
        "bands": cfg_bands,
        "active_datasets": active_datasets,
        "eval_dataset": eval_dataset,
        "layouts": dataset_layouts,
        "primary_dataset": primary_dataset,
        "auxiliary_dataset": auxiliary_dataset,
    }


def predict_batch(
    model: torch.nn.Module,
    X: torch.Tensor,
    *,
    validity_mode: str = "min_fraction",
    min_finite_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(X)
    mode = str(validity_mode).lower()
    if mode in {"all", "all_channels"}:
        valid_x = finite.all(dim=1, keepdim=True)
    elif mode in {"any", "any_channel"}:
        valid_x = finite.any(dim=1, keepdim=True)
    elif mode in {"min_fraction", "fraction"}:
        thr = float(min_finite_ratio)
        if thr <= 0.0 or thr > 1.0:
            raise ValueError(f"min_finite_ratio must be in (0,1], got {thr}.")
        valid_x = finite.to(dtype=torch.float32).mean(dim=1, keepdim=True) >= thr
    else:
        raise ValueError(
            f"Unsupported predictor validity_mode='{validity_mode}'. "
            "Use all_channels|any_channel|min_fraction."
        )
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = torch.clamp(X, min=-1e4, max=1e4)
    return model(X), valid_x


def move_bundle_to_device(
    bundle: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        dataset_name: {
            "y": payload["y"].to(device=device, dtype=torch.float32),
            "source_mask": payload["source_mask"].to(
                device=device, dtype=torch.float32
            ),
        }
        for dataset_name, payload in bundle.items()
    }


def slice_center(t: torch.Tensor, size: int) -> torch.Tensor:
    h = t.shape[-2]
    w = t.shape[-1]
    size = max(1, int(size))
    size = min(size, h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return t[..., top : top + size, left : left + size]


def apply_supervision_tensor(
    t: torch.Tensor,
    mode: str,
    center_crop_size: int,
) -> torch.Tensor:
    if mode == "dense":
        return t
    if mode == "center_pixel":
        return slice_center(t, 1)
    if mode == "center_crop":
        return slice_center(t, center_crop_size)
    raise ValueError(
        f"Unsupported train.supervision.mode='{mode}'. Use dense|center_pixel|center_crop."
    )


def apply_supervision_bundle(
    bundle: dict[str, dict[str, torch.Tensor]],
    mode: str,
    center_crop_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for dataset_name, payload in bundle.items():
        out[dataset_name] = {
            "y": apply_supervision_tensor(payload["y"], mode, center_crop_size),
            "source_mask": apply_supervision_tensor(
                payload["source_mask"], mode, center_crop_size
            ),
        }
    return out


def mask_bundle_targets_with_validity(
    bundle: dict[str, dict[str, torch.Tensor]],
    valid_x: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    for payload in bundle.values():
        payload["y"] = torch.where(
            valid_x.expand_as(payload["y"]), payload["y"], torch.nan
        )
        payload["y"] = torch.where(payload["source_mask"] > 0, payload["y"], torch.nan)
    return bundle
