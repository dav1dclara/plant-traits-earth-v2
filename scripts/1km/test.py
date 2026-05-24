from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import rasterio
import torch
import zarr
from affine import Affine
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rasterio.crs import CRS
from rasterio.windows import Window
from rich.console import Console
from rich.progress import track

import wandb
from ptev2.data.dataloader import _open_zarr, _resolve_store_path, get_dataloader
from ptev2.metrics.core import mae, pearson_r, r2_score, rmse
from ptev2.transformations import denormalize_predictions, load_power_transformer_params
from ptev2.utils import (
    apply_supervision_bundle,
    apply_supervision_tensor,
    assert_supervision_shape,
    build_target_layout,
    mask_bundle_targets_with_validity,
    move_bundle_to_device,
    predict_batch,
    resolve_device,
    resolve_model_cfg,
    resolve_predictor_validity_config,
    resolve_supervision_config,
    resolve_train_zarr_dir,
    seed_all,
)

console = Console()


def _open_chip_store(store_path: Path):
    if str(store_path).endswith(".h5"):
        import h5py

        return h5py.File(store_path, "r")
    return _open_zarr(store_path)


def _resolve_power_transformer_params_csv(train_cfg: DictConfig) -> Path:
    zarr_dir = resolve_train_zarr_dir(train_cfg).resolve()
    for parent in zarr_dir.parents:
        if parent.name == "data":
            return parent / "power_transformer_params.csv"
    raise ValueError(
        f"Could not infer data root from zarr_dir '{zarr_dir}'. "
        "Expected it to live under a 'data/' directory."
    )


def _summarize_denormalized_trait_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    traits: list[str],
    n_bands: int,
    params_df,
    denorm_kwargs: dict | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, list[np.ndarray]]]:
    """
    Compute denormalized metrics per trait.

    Returns:
        trait_metrics: dict of trait_name -> {n_valid, rmse, r2, pearson_r, mae, nrmse, residual_std, target_std, prediction_std}
        trait_samples: dict of trait_name -> [yt_flat, yp_flat] for macro aggregation
    """
    trait_metrics: dict[str, dict[str, float]] = {}
    trait_samples: dict[str, list[np.ndarray]] = {}
    denorm_kwargs = denorm_kwargs or {}
    denorm_kwargs_true = dict(denorm_kwargs)
    denorm_kwargs_true["clip_z_abs"] = None
    denorm_kwargs_true["max_abs_output"] = None
    denorm_kwargs_true["clip_to_inverse_domain"] = False
    denorm_kwargs_true["domain_margin_ratio"] = 0.0
    denorm_kwargs_true["domain_margin_abs"] = 0.0

    for trait_idx, trait_name in enumerate(traits):
        start = trait_idx * n_bands
        stop = start + n_bands

        y_true_trait = denormalize_predictions(
            y_true[:, start:stop, :, :], trait_name, params_df, **denorm_kwargs_true
        )
        y_pred_trait = denormalize_predictions(
            y_pred[:, start:stop, :, :], trait_name, params_df, **denorm_kwargs
        )
        valid_trait = valid_mask[:, start:stop, :, :]
        valid_trait = (
            valid_trait & np.isfinite(y_true_trait) & np.isfinite(y_pred_trait)
        )

        if not bool(valid_trait.any()):
            trait_metrics[trait_name] = {
                "n_valid": 0,
                "rmse": float("nan"),
                "nrmse": float("nan"),
                "r2": float("nan"),
                "pearson_r": float("nan"),
                "mae": float("nan"),
                "residual_std": float("nan"),
                "target_std": float("nan"),
                "prediction_std": float("nan"),
            }
            trait_samples[trait_name] = [np.array([]), np.array([])]
            continue

        yt_trait = y_true_trait[valid_trait]
        yp_trait = y_pred_trait[valid_trait]
        residuals = yp_trait - yt_trait
        value_range = float(np.nanmax(yt_trait) - np.nanmin(yt_trait))
        trait_rmse = float(rmse(yt_trait, yp_trait))

        trait_metrics[trait_name] = {
            "n_valid": int(yt_trait.size),
            "rmse": trait_rmse,
            "nrmse": float(trait_rmse / value_range)
            if np.isfinite(value_range) and value_range > 0.0
            else float("nan"),
            "r2": float(r2_score(yt_trait, yp_trait)),
            "pearson_r": float(pearson_r(yt_trait, yp_trait)),
            "mae": float(mae(yt_trait, yp_trait)),
            "residual_std": float(np.std(residuals)),
            "target_std": float(np.std(yt_trait)),
            "prediction_std": float(np.std(yp_trait)),
        }
        trait_samples[trait_name] = [yt_trait, yp_trait]

    return trait_metrics, trait_samples


def _compute_macro_denormalized_metrics(
    trait_metrics: dict[str, dict[str, float]],
    trait_samples: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    """
    Compute macro metrics from denormalized per-trait data.

    - pearson_r_macro: mean of trait Pearson r values
    - r2_macro: mean of trait R² values
    - rmse_macro: sqrt(mean(MSE)) over all samples
    - mae_macro: mean(|residuals|) over all samples
    - nrmse_macro: mean of trait NRMSE values
    - residual_std_macro: mean of trait residual_std values
    """
    macro_metrics: dict[str, float] = {}

    # Trait-wise averages
    pearson_values = [
        m["pearson_r"] for m in trait_metrics.values() if np.isfinite(m["pearson_r"])
    ]
    r2_values = [m["r2"] for m in trait_metrics.values() if np.isfinite(m["r2"])]
    nrmse_values = [
        m["nrmse"] for m in trait_metrics.values() if np.isfinite(m["nrmse"])
    ]
    residual_std_values = [
        m["residual_std"]
        for m in trait_metrics.values()
        if np.isfinite(m["residual_std"])
    ]

    macro_metrics["pearson_r_macro"] = (
        float(np.mean(pearson_values)) if pearson_values else float("nan")
    )
    macro_metrics["r2_macro"] = float(np.mean(r2_values)) if r2_values else float("nan")
    macro_metrics["nrmse_macro"] = (
        float(np.mean(nrmse_values)) if nrmse_values else float("nan")
    )
    macro_metrics["residual_std_macro"] = (
        float(np.mean(residual_std_values)) if residual_std_values else float("nan")
    )

    # Aggregate over all samples
    all_yt = []
    all_yp = []
    for trait_name in trait_metrics.keys():
        yt, yp = trait_samples.get(trait_name, [np.array([]), np.array([])])
        if yt.size > 0:
            all_yt.append(yt)
            all_yp.append(yp)

    if all_yt:
        all_yt_concat = np.concatenate(all_yt)
        all_yp_concat = np.concatenate(all_yp)
        residuals_all = all_yp_concat - all_yt_concat
        mse_all = np.mean(residuals_all**2)
        macro_metrics["rmse_macro"] = float(np.sqrt(mse_all))
        macro_metrics["mae_macro"] = float(np.mean(np.abs(residuals_all)))
    else:
        macro_metrics["rmse_macro"] = float("nan")
        macro_metrics["mae_macro"] = float("nan")

    return macro_metrics


def _summarize_denormalized_trait_parts(
    *,
    y_true_parts: list[list[np.ndarray]],
    y_pred_parts: list[list[np.ndarray]],
    traits: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, list[np.ndarray]]]:
    trait_metrics: dict[str, dict[str, float]] = {}
    trait_samples: dict[str, list[np.ndarray]] = {}

    for trait_idx, trait_name in enumerate(traits):
        if not y_true_parts[trait_idx]:
            trait_metrics[trait_name] = {
                "n_valid": 0,
                "rmse": float("nan"),
                "nrmse": float("nan"),
                "r2": float("nan"),
                "pearson_r": float("nan"),
                "mae": float("nan"),
                "residual_std": float("nan"),
                "target_std": float("nan"),
                "prediction_std": float("nan"),
            }
            trait_samples[trait_name] = [np.array([]), np.array([])]
            continue

        yt_trait = np.concatenate(y_true_parts[trait_idx])
        yp_trait = np.concatenate(y_pred_parts[trait_idx])
        residuals = yp_trait - yt_trait
        value_range = float(np.nanmax(yt_trait) - np.nanmin(yt_trait))
        trait_rmse = float(rmse(yt_trait, yp_trait))

        trait_metrics[trait_name] = {
            "n_valid": int(yt_trait.size),
            "rmse": trait_rmse,
            "nrmse": float(trait_rmse / value_range)
            if np.isfinite(value_range) and value_range > 0.0
            else float("nan"),
            "r2": float(r2_score(yt_trait, yp_trait)),
            "pearson_r": float(pearson_r(yt_trait, yp_trait)),
            "mae": float(mae(yt_trait, yp_trait)),
            "residual_std": float(np.std(residuals)),
            "target_std": float(np.std(yt_trait)),
            "prediction_std": float(np.std(yp_trait)),
        }
        trait_samples[trait_name] = [yt_trait, yp_trait]

    return trait_metrics, trait_samples


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_override = OmegaConf.select(cfg, "checkpoint_path")
    if checkpoint_override:
        return Path(str(checkpoint_override))

    run_name = OmegaConf.select(cfg, "run_name")
    if run_name:
        return Path(str(cfg.checkpoint_dir)) / f"{str(run_name)}.pth"

    raise ValueError("Set checkpoint_path or run_name.")


def _resolve_denormalization_kwargs(cfg: DictConfig) -> dict:
    """
    Resolve denormalization stability options from test config.

    Returns kwargs forwarded to ptev2.transformations.denormalize_predictions.
    """
    clip_z_abs = OmegaConf.select(cfg, "denormalization.clip_z_abs")
    max_abs_output = OmegaConf.select(cfg, "denormalization.max_abs_output")

    kwargs = {
        "clip_z_abs": float(clip_z_abs) if clip_z_abs is not None else None,
        "clip_to_inverse_domain": bool(
            OmegaConf.select(cfg, "denormalization.clip_to_inverse_domain")
            if OmegaConf.select(cfg, "denormalization.clip_to_inverse_domain")
            is not None
            else True
        ),
        "domain_eps": float(
            OmegaConf.select(cfg, "denormalization.domain_eps")
            if OmegaConf.select(cfg, "denormalization.domain_eps") is not None
            else 1e-6
        ),
        "domain_margin_ratio": float(
            OmegaConf.select(cfg, "denormalization.domain_margin_ratio")
            if OmegaConf.select(cfg, "denormalization.domain_margin_ratio") is not None
            else 0.1
        ),
        "domain_margin_abs": float(
            OmegaConf.select(cfg, "denormalization.domain_margin_abs")
            if OmegaConf.select(cfg, "denormalization.domain_margin_abs") is not None
            else 1e-3
        ),
        "max_abs_output": float(max_abs_output) if max_abs_output is not None else None,
    }
    return kwargs


def _load_checkpoint_and_train_cfg(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, DictConfig]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' did not contain a dict state."
        )

    embedded_cfg = state.get("config")
    if embedded_cfg is not None:
        return state, OmegaConf.create(embedded_cfg)

    cfg_path = checkpoint_path.with_suffix(".yaml")
    if not cfg_path.exists():
        raise ValueError(
            "Checkpoint has no embedded config and companion yaml is missing: "
            f"{cfg_path}"
        )
    return state, OmegaConf.load(cfg_path)


def _store_len(store: zarr.Group, predictors: list[str]) -> int:
    return int(store[f"predictors/{predictors[0]}"].shape[0])


def _read_predictor_batch(
    store: zarr.Group,
    predictors: list[str],
    start: int,
    stop: int,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.as_tensor(np.asarray(store[f"predictors/{name}"][start:stop]))
            for name in predictors
        ],
        dim=1,
    )


def _weight_map(height: int, width: int) -> np.ndarray:
    y = np.minimum(np.arange(1, height + 1), np.arange(height, 0, -1)).astype(
        np.float32
    )
    x = np.minimum(np.arange(1, width + 1), np.arange(width, 0, -1)).astype(np.float32)
    return np.outer(y, x).astype(np.float32) ** 2


def _chip_window_from_bounds(
    bounds: np.ndarray,
    transform: Affine,
    raster_height: int,
    raster_width: int,
    chip_height: int,
    chip_width: int,
    row_offset: int,
    col_offset: int,
) -> tuple[Window, tuple[slice, slice]] | None:
    min_x, _, _, max_y = [float(v) for v in bounds]
    pixel_w = float(transform.a)
    pixel_h = float(abs(transform.e))
    col = round((min_x - transform.c) / pixel_w) + int(col_offset)
    row = round((transform.f - max_y) / pixel_h) + int(row_offset)

    r0 = max(row, 0)
    c0 = max(col, 0)
    r1 = min(row + chip_height, raster_height)
    c1 = min(col + chip_width, raster_width)
    if r0 >= r1 or c0 >= c1:
        return None

    chip_rows = slice(r0 - row, r1 - row)
    chip_cols = slice(c0 - col, c1 - col)
    return Window(c0, r0, c1 - c0, r1 - r0), (chip_rows, chip_cols)


def _update_mosaic(
    *,
    sum_dst: rasterio.DatasetWriter,
    weight_dst: rasterio.DatasetWriter,
    pred: np.ndarray,
    bounds: np.ndarray,
    transform: Affine,
    raster_height: int,
    raster_width: int,
    weights_full: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> None:
    out = _chip_window_from_bounds(
        bounds,
        transform,
        raster_height,
        raster_width,
        pred.shape[-2],
        pred.shape[-1],
        row_offset,
        col_offset,
    )
    if out is None:
        return
    window, (chip_rows, chip_cols) = out
    chip = pred[:, chip_rows, chip_cols]
    finite = np.isfinite(chip)
    if not bool(finite.any()):
        return

    weights = weights_full[chip_rows, chip_cols][None, :, :] * finite
    weighted = np.nan_to_num(chip, nan=0.0, posinf=0.0, neginf=0.0) * weights

    current_sum = sum_dst.read(window=window)
    current_weight = weight_dst.read(window=window)
    current_sum += weighted
    current_weight += weights
    sum_dst.write(current_sum, window=window)
    weight_dst.write(current_weight, window=window)


def _finalize_average(
    *,
    sum_path: Path,
    weight_path: Path,
    output_path: Path,
    profile: dict[str, Any],
    band_names: list[str],
    block_size: int,
) -> None:
    with rasterio.open(sum_path) as sum_src, rasterio.open(weight_path) as weight_src:
        with rasterio.open(output_path, "w", **profile) as dst:
            for band_idx, name in enumerate(band_names, start=1):
                dst.set_band_description(band_idx, name)

            for row in track(
                range(0, profile["height"], block_size), description="Final averaging"
            ):
                height = min(block_size, profile["height"] - row)
                window = Window(0, row, profile["width"], height)
                sums = sum_src.read(window=window)
                weights = weight_src.read(window=window)
                out = np.full_like(sums, np.nan, dtype=np.float32)
                valid = weights > 0
                out[valid] = sums[valid] / weights[valid]
                dst.write(out, window=window)


def _denormalize_prediction_batch(
    pred: np.ndarray,
    traits: list[str],
    bands: list[str],
    params_df,
    denorm_kwargs: dict[str, Any],
) -> np.ndarray:
    denorm = np.full_like(pred, np.nan, dtype=np.float32)
    n_bands = len(bands)
    for trait_idx, trait in enumerate(traits):
        if trait not in params_df.index:
            continue
        start = trait_idx * n_bands
        stop = start + n_bands
        denorm[:, start:stop] = denormalize_predictions(
            pred[:, start:stop],
            trait,
            params_df,
            **denorm_kwargs,
        )
    return denorm


def _write_all_prediction_tif_streaming(
    *,
    model: torch.nn.Module,
    all_store: zarr.Group,
    predictors: list[str],
    traits: list[str],
    bands: list[str],
    power_transform_params,
    denorm_kwargs: dict[str, Any],
    out_tif_path: Path,
    band_names: list[str],
    batch_size: int,
    device: torch.device,
    supervision_mode: str,
    center_crop_size: int,
    predictor_validity_mode: str,
    predictor_min_finite_ratio: float,
    overwrite: bool,
    keep_temp: bool,
    final_block_size: int,
) -> None:
    if out_tif_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prediction TIF already exists: {out_tif_path}. Set overwrite=true to replace it."
        )

    t = all_store.attrs["transform"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs_epsg = int(all_store.attrs["crs_epsg"])
    raster_height = int(all_store.attrs["raster_height"])
    raster_width = int(all_store.attrs["raster_width"])
    out_channels = len(traits) * len(bands)

    patch_size = int(all_store.attrs["patch_size"])
    if supervision_mode == "center_pixel":
        pred_height = pred_width = 1
    elif supervision_mode == "center_crop":
        pred_height = pred_width = int(center_crop_size)
    else:
        pred_height = pred_width = patch_size
    row_offset = max((patch_size - pred_height) // 2, 0)
    col_offset = max((patch_size - pred_width) // 2, 0)
    weights_full = _weight_map(pred_height, pred_width)

    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path = out_tif_path.with_suffix(".sum.tmp.tif")
    weight_path = out_tif_path.with_suffix(".weight.tmp.tif")
    for temp_path in (sum_path, weight_path):
        if temp_path.exists():
            if overwrite:
                temp_path.unlink()
            else:
                raise FileExistsError(
                    f"Temporary mosaic file already exists: {temp_path}. Set overwrite=true to replace it."
                )

    base_profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": raster_height,
        "width": raster_width,
        "count": out_channels,
        "dtype": "float32",
        "crs": CRS.from_epsg(crs_epsg),
        "transform": Affine(
            float(transform.a),
            0,
            float(transform.c),
            0,
            float(transform.e),
            float(transform.f),
        ),
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES",
    }
    temp_profile = dict(base_profile)
    temp_profile.update({"nodata": 0.0})
    out_profile = dict(base_profile)
    out_profile.update({"compress": "deflate", "nodata": np.nan})

    with rasterio.open(sum_path, "w", **temp_profile):
        pass
    with rasterio.open(weight_path, "w", **temp_profile):
        pass

    n = _store_len(all_store, predictors)
    with rasterio.open(sum_path, "r+") as sum_dst:
        with rasterio.open(weight_path, "r+") as weight_dst:
            for start in track(range(0, n, batch_size), description="All-map"):
                stop = min(start + batch_size, n)
                x = _read_predictor_batch(all_store, predictors, start, stop).to(
                    device=device, dtype=torch.float32
                )
                bounds = np.asarray(all_store["bounds"][start:stop], dtype=np.float64)
                with torch.no_grad():
                    y_pred_map, valid_x, _ = _prepare_batch(
                        model,
                        x,
                        None,
                        device=device,
                        supervision_mode=supervision_mode,
                        center_crop_size=center_crop_size,
                        predictor_validity_mode=predictor_validity_mode,
                        predictor_min_finite_ratio=predictor_min_finite_ratio,
                        context="all_map",
                    )
                    y_pred_map = torch.where(
                        valid_x.expand_as(y_pred_map),
                        y_pred_map,
                        torch.full_like(y_pred_map, float("nan")),
                    )
                pred_np = y_pred_map.detach().cpu().numpy()
                pred_np = _denormalize_prediction_batch(
                    pred_np, traits, bands, power_transform_params, denorm_kwargs
                )
                for pred, chip_bounds in zip(pred_np, bounds, strict=True):
                    _update_mosaic(
                        sum_dst=sum_dst,
                        weight_dst=weight_dst,
                        pred=pred,
                        bounds=chip_bounds,
                        transform=transform,
                        raster_height=raster_height,
                        raster_width=raster_width,
                        weights_full=weights_full,
                        row_offset=row_offset,
                        col_offset=col_offset,
                    )

    _finalize_average(
        sum_path=sum_path,
        weight_path=weight_path,
        output_path=out_tif_path,
        profile=out_profile,
        band_names=band_names,
        block_size=int(final_block_size),
    )
    if not keep_temp:
        sum_path.unlink(missing_ok=True)
        weight_path.unlink(missing_ok=True)


def _write_per_trait_prediction_tifs_streaming(
    *,
    model: torch.nn.Module,
    all_store: zarr.Group,
    predictors: list[str],
    traits: list[str],
    bands: list[str],
    power_transform_params,
    denorm_kwargs: dict[str, Any],
    out_dir: Path,
    batch_size: int,
    device: torch.device,
    supervision_mode: str,
    center_crop_size: int,
    predictor_validity_mode: str,
    predictor_min_finite_ratio: float,
    overwrite: bool,
    keep_temp: bool,
    final_block_size: int,
) -> list[Path]:
    t = all_store.attrs["transform"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs_epsg = int(all_store.attrs["crs_epsg"])
    raster_height = int(all_store.attrs["raster_height"])
    raster_width = int(all_store.attrs["raster_width"])
    n_bands = len(bands)

    patch_size = int(all_store.attrs["patch_size"])
    if supervision_mode == "center_pixel":
        pred_height = pred_width = 1
    elif supervision_mode == "center_crop":
        pred_height = pred_width = int(center_crop_size)
    else:
        pred_height = pred_width = patch_size
    row_offset = max((patch_size - pred_height) // 2, 0)
    col_offset = max((patch_size - pred_width) // 2, 0)
    weights_full = _weight_map(pred_height, pred_width)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": raster_height,
        "width": raster_width,
        "count": n_bands,
        "dtype": "float32",
        "crs": CRS.from_epsg(crs_epsg),
        "transform": Affine(
            float(transform.a),
            0,
            float(transform.c),
            0,
            float(transform.e),
            float(transform.f),
        ),
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES",
    }
    temp_profile = dict(base_profile)
    temp_profile.update({"nodata": 0.0})
    out_profile = dict(base_profile)
    out_profile.update({"compress": "deflate", "nodata": np.nan})

    trait_specs: list[dict[str, Any]] = []
    for trait_idx, trait in enumerate(traits):
        start = trait_idx * n_bands
        stop = start + n_bands
        output_path = out_dir / f"{trait}.tif"
        sum_path = out_dir / f"{trait}.sum.tmp.tif"
        weight_path = out_dir / f"{trait}.weight.tmp.tif"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Prediction TIF already exists: {output_path}. Set overwrite=true to replace it."
            )
        for temp_path in (sum_path, weight_path):
            if temp_path.exists():
                if overwrite:
                    temp_path.unlink()
                else:
                    raise FileExistsError(
                        f"Temporary mosaic file already exists: {temp_path}. Set overwrite=true to replace it."
                    )
        with rasterio.open(sum_path, "w", **temp_profile):
            pass
        with rasterio.open(weight_path, "w", **temp_profile):
            pass
        trait_specs.append(
            {
                "trait": trait,
                "start": start,
                "stop": stop,
                "output_path": output_path,
                "sum_path": sum_path,
                "weight_path": weight_path,
            }
        )

    n = _store_len(all_store, predictors)
    with ExitStack() as stack:
        for spec in trait_specs:
            spec["sum_dst"] = stack.enter_context(
                rasterio.open(spec["sum_path"], "r+")
            )
            spec["weight_dst"] = stack.enter_context(
                rasterio.open(spec["weight_path"], "r+")
            )

        for start in track(range(0, n, batch_size), description="All-map"):
            stop = min(start + batch_size, n)
            x = _read_predictor_batch(all_store, predictors, start, stop).to(
                device=device, dtype=torch.float32
            )
            bounds = np.asarray(all_store["bounds"][start:stop], dtype=np.float64)
            with torch.no_grad():
                y_pred_map, valid_x, _ = _prepare_batch(
                    model,
                    x,
                    None,
                    device=device,
                    supervision_mode=supervision_mode,
                    center_crop_size=center_crop_size,
                    predictor_validity_mode=predictor_validity_mode,
                    predictor_min_finite_ratio=predictor_min_finite_ratio,
                    context="all_map",
                )
                y_pred_map = torch.where(
                    valid_x.expand_as(y_pred_map),
                    y_pred_map,
                    torch.full_like(y_pred_map, float("nan")),
                )
            pred_np = y_pred_map.detach().cpu().numpy()
            pred_np = _denormalize_prediction_batch(
                pred_np, traits, bands, power_transform_params, denorm_kwargs
            )
            for pred, chip_bounds in zip(pred_np, bounds, strict=True):
                for spec in trait_specs:
                    _update_mosaic(
                        sum_dst=spec["sum_dst"],
                        weight_dst=spec["weight_dst"],
                        pred=pred[spec["start"] : spec["stop"]],
                        bounds=chip_bounds,
                        transform=transform,
                        raster_height=raster_height,
                        raster_width=raster_width,
                        weights_full=weights_full,
                        row_offset=row_offset,
                        col_offset=col_offset,
                    )

    output_paths: list[Path] = []
    for spec in track(trait_specs, description="Final per-trait averaging"):
        band_names = (
            [spec["trait"]]
            if n_bands == 1
            else [f"{spec['trait']}_{band}" for band in bands]
        )
        _finalize_average(
            sum_path=spec["sum_path"],
            weight_path=spec["weight_path"],
            output_path=spec["output_path"],
            profile=out_profile,
            band_names=band_names,
            block_size=int(final_block_size),
        )
        if not keep_temp:
            spec["sum_path"].unlink(missing_ok=True)
            spec["weight_path"].unlink(missing_ok=True)
        output_paths.append(spec["output_path"])

    return output_paths


def _prepare_batch(
    model: torch.nn.Module,
    X: torch.Tensor,
    bundle: dict[str, dict[str, torch.Tensor]] | None,
    *,
    device: torch.device,
    supervision_mode: str,
    center_crop_size: int,
    predictor_validity_mode: str,
    predictor_min_finite_ratio: float,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]] | None]:
    X = X.to(device=device, dtype=torch.float32)
    if bundle is not None:
        bundle = move_bundle_to_device(bundle, device)
    y_pred, valid_x = predict_batch(
        model,
        X,
        validity_mode=predictor_validity_mode,
        min_finite_ratio=predictor_min_finite_ratio,
    )
    y_pred = apply_supervision_tensor(y_pred, supervision_mode, center_crop_size)
    valid_x = apply_supervision_tensor(valid_x, supervision_mode, center_crop_size)
    assert_supervision_shape(
        y_pred,
        mode=supervision_mode,
        center_crop_size=center_crop_size,
        context=f"{context}/y_pred",
    )
    assert_supervision_shape(
        valid_x,
        mode=supervision_mode,
        center_crop_size=center_crop_size,
        context=f"{context}/valid_x",
    )
    if bundle is not None:
        bundle = apply_supervision_bundle(bundle, supervision_mode, center_crop_size)
        bundle = mask_bundle_targets_with_validity(bundle, valid_x)
        for dataset_name, payload in bundle.items():
            assert_supervision_shape(
                payload["y"],
                mode=supervision_mode,
                center_crop_size=center_crop_size,
                context=f"{context}/{dataset_name}/y",
            )
            assert_supervision_shape(
                payload["source_mask"],
                mode=supervision_mode,
                center_crop_size=center_crop_size,
                context=f"{context}/{dataset_name}/source_mask",
            )
    return y_pred, valid_x, bundle


def _update_source_counts(
    *,
    source_counts_per_trait: dict[str, dict[str, int]],
    source_key: str,
    payload: dict[str, torch.Tensor],
    y_pred: torch.Tensor,
    traits: list[str],
    n_bands: int,
) -> int:
    valid = (
        torch.isfinite(payload["y"])
        & torch.isfinite(y_pred)
        & (payload["source_mask"] > 0)
    )
    total = 0
    for trait_idx, trait_name in enumerate(traits):
        start = trait_idx * n_bands
        stop = start + n_bands
        count = int(valid[:, start:stop, :, :].sum().item())
        source_counts_per_trait[trait_name][source_key] += count
        total += count
    return total


@hydra.main(version_base=None, config_path="../../config/1km/test", config_name="default")
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TEST + FINAL PREDICTION[/bold cyan]")

    device = resolve_device()
    console.print(f"Device: [cyan]{device}[/cyan]")

    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state, train_cfg = _load_checkpoint_and_train_cfg(checkpoint_path, device)
    seed_all(int(train_cfg.train.seed))

    wandb_cfg = OmegaConf.select(train_cfg, "wandb")
    wandb_enabled = bool(OmegaConf.select(train_cfg, "wandb.enabled") or False)
    wandb_run = None
    if wandb_enabled and wandb_cfg is not None:
        # Extract training run name from checkpoint path (e.g., "stl_s0" from "stl_s0.pth")
        training_run_name = checkpoint_path.stem
        entity = str(OmegaConf.select(train_cfg, "wandb.entity") or "")
        project = str(OmegaConf.select(train_cfg, "wandb.project") or "")

        # Try to find and resume existing training run
        run_id_to_resume = None
        try:
            from wandb.apis.public import Api

            api = Api()
            # Query for runs with the training run name
            runs = api.runs(
                f"{entity}/{project}", filters={"display_name": training_run_name}
            )
            if runs:
                run_id_to_resume = runs[0].id
                console.print(
                    f"[cyan]Found existing training run:[/cyan] [yellow]{training_run_name}[/yellow] "
                    f"(id={run_id_to_resume})"
                )
        except Exception as e:
            console.print(f"[yellow]Could not query W&B API:[/yellow] {str(e)}")

        if run_id_to_resume:
            # Resume existing training run instead of creating new one
            wandb_run = wandb.init(
                entity=entity,
                project=project,
                id=run_id_to_resume,
                resume="must",
                job_type="test",
            )
            console.print(f"[cyan]Resumed W&B run for testing[/cyan]")
        else:
            console.print(
                f"[yellow]No existing training run found for:[/yellow] {training_run_name}. "
                f"[yellow]Test metrics will NOT be logged to W&B.[/yellow]"
            )
            wandb_enabled = False  # Disable W&B if no training run found

    zarr_dir_override = OmegaConf.select(cfg, "zarr_dir")
    if zarr_dir_override:
        if OmegaConf.select(train_cfg, "data") is None:
            train_cfg.data = {}
        train_cfg.data.zarr_dir = str(zarr_dir_override)
        console.print(
            f"[yellow]Overriding checkpoint zarr_dir with test config:[/yellow] "
            f"[cyan]{train_cfg.data.zarr_dir}[/cyan]"
        )

    predictors = [k for k, v in train_cfg.data.predictors.items() if bool(v.use)]
    if not predictors:
        raise ValueError("No predictors enabled in checkpoint training config.")

    zarr_dir = resolve_train_zarr_dir(train_cfg)
    test_split = str(OmegaConf.select(cfg, "test_split") or "test")
    layout_store_path = _resolve_store_path(zarr_dir, test_split)
    layout_store = _open_chip_store(layout_store_path)
    console.print(f"Layout store: [cyan]{layout_store_path}[/cyan]")
    target_layout = build_target_layout(train_cfg, layout_store)
    power_transform_csv = _resolve_power_transformer_params_csv(train_cfg)
    power_transform_params = load_power_transformer_params(power_transform_csv)
    denorm_kwargs = _resolve_denormalization_kwargs(cfg)
    console.print(
        "[cyan]Denormalization stability:[/cyan] "
        f"clip_z_abs={denorm_kwargs['clip_z_abs']}, "
        f"clip_to_inverse_domain={denorm_kwargs['clip_to_inverse_domain']}, "
        f"domain_eps={denorm_kwargs['domain_eps']}, "
        f"domain_margin_ratio={denorm_kwargs['domain_margin_ratio']}, "
        f"domain_margin_abs={denorm_kwargs['domain_margin_abs']}, "
        f"max_abs_output={denorm_kwargs['max_abs_output']}"
    )

    # Identify traits with and without power-transform parameters
    original_traits = target_layout["traits"]
    available_traits = [t for t in original_traits if t in power_transform_params.index]
    missing_traits = [
        t for t in original_traits if t not in power_transform_params.index
    ]

    if missing_traits:
        console.print(
            f"[yellow]Warning:[/yellow] Traits without power-transform parameters will be skipped during evaluation: "
            f"{missing_traits}"
        )

    if not available_traits:
        raise ValueError(
            f"No traits with power-transform parameters found. All {len(original_traits)} traits "
            f"are missing parameters. Checked {power_transform_csv}."
        )

    # Filter power_transform_params to only available traits for later use
    power_transform_params_available = power_transform_params.loc[available_traits]

    # Keep original_traits for model loading (checkpoint was trained with all traits)
    # We'll filter predictions to available_traits during evaluation

    eval_dataset_ckpt = str(target_layout["eval_dataset"])
    eval_dataset_override = OmegaConf.select(cfg, "eval_dataset")
    if (
        eval_dataset_override is not None
        and str(eval_dataset_override) != eval_dataset_ckpt
    ):
        raise ValueError(
            "Test eval_dataset override is not allowed. "
            f"Checkpoint expects '{eval_dataset_ckpt}', got '{eval_dataset_override}'."
        )
    eval_dataset = eval_dataset_ckpt

    test_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=test_split,
        predictors=predictors,
        target_layouts=target_layout["layouts"],
        return_target_bundle=True,
        batch_size=int(
            OmegaConf.select(cfg, "test_batch_size")
            or train_cfg.data_loaders.batch_size
        ),
        num_workers=int(
            OmegaConf.select(cfg, "test_num_workers")
            if OmegaConf.select(cfg, "test_num_workers") is not None
            else train_cfg.data_loaders.num_workers
        ),
    )

    model_cfg = resolve_model_cfg(train_cfg)
    checkpoint_state = state.get("state_dict", state)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Checkpoint state for '{checkpoint_path}' is invalid.")

    total_pred_bands = sum(
        layout_store[f"predictors/{name}"].shape[1] for name in predictors
    )
    traits = list(target_layout["traits"])  # All 37 traits (for model architecture)
    bands = list(target_layout["bands"])
    out_channels = len(traits) * len(bands)

    # Compute indices of available traits for later filtering
    available_trait_indices = [
        i for i, trait in enumerate(traits) if trait in available_traits
    ]
    n_available_traits = len(available_trait_indices)

    model = instantiate(
        model_cfg,
        in_channels=total_pred_bands,
        out_channels=out_channels,
    ).to(device)
    model.load_state_dict(checkpoint_state)
    model.eval()

    supervision_mode, center_crop_size = resolve_supervision_config(train_cfg)
    predictor_validity_mode, predictor_min_finite_ratio = (
        resolve_predictor_validity_config(train_cfg)
    )

    cfg_supervision_mode = OmegaConf.select(cfg, "train.supervision.mode")
    if (
        cfg_supervision_mode is not None
        and str(cfg_supervision_mode) != supervision_mode
    ):
        raise AssertionError(
            "train.supervision.mode must match the checkpoint training config during test."
        )
    cfg_center_crop_size = OmegaConf.select(cfg, "train.supervision.center_crop_size")
    if (
        cfg_center_crop_size is not None
        and int(cfg_center_crop_size) != center_crop_size
    ):
        raise AssertionError(
            "train.supervision.center_crop_size must match the checkpoint training config during test."
        )

    if supervision_mode == "dense":
        console.print(
            "[yellow]Dense supervision with overlapping chips can duplicate target pixels in "
            "loss/metrics. This run reports per-chip dense metrics; unique-cell aggregation can be added later.[/yellow]"
        )
    if supervision_mode == "center_crop":
        console.print(
            "[yellow]Center-crop supervision uses only the central crop for loss/metrics; "
            "outer patch pixels provide context only.[/yellow]"
        )

    traits_for_metrics = available_traits
    denorm_kwargs_true = dict(denorm_kwargs)
    denorm_kwargs_true["clip_z_abs"] = None
    denorm_kwargs_true["max_abs_output"] = None
    denorm_kwargs_true["clip_to_inverse_domain"] = False
    denorm_kwargs_true["domain_margin_ratio"] = 0.0
    denorm_kwargs_true["domain_margin_abs"] = 0.0
    y_true_parts: list[list[np.ndarray]] = [[] for _ in traits_for_metrics]
    y_pred_parts: list[list[np.ndarray]] = [[] for _ in traits_for_metrics]
    skipped_batches = 0
    valid_test_batches = 0

    source_counts_per_trait: dict[str, dict[str, int]] = {
        trait: {"n_valid_splot": 0, "n_valid_gbif": 0} for trait in traits
    }
    n_valid_splot = 0
    n_valid_gbif = 0

    with torch.no_grad():
        for X, bundle in track(
            test_loader, description=f"Test [{checkpoint_path.stem}]"
        ):
            y_pred, _, bundle = _prepare_batch(
                model,
                X,
                bundle,
                device=device,
                supervision_mode=supervision_mode,
                center_crop_size=center_crop_size,
                predictor_validity_mode=predictor_validity_mode,
                predictor_min_finite_ratio=predictor_min_finite_ratio,
                context="test",
            )
            if bundle is None:
                skipped_batches += 1
                continue

            splot_payload = bundle.get("splot")
            if splot_payload is not None:
                n_valid_splot += _update_source_counts(
                    source_counts_per_trait=source_counts_per_trait,
                    source_key="n_valid_splot",
                    payload=splot_payload,
                    y_pred=y_pred,
                    traits=traits,
                    n_bands=len(bands),
                )

            gbif_payload = bundle.get("gbif")
            if gbif_payload is not None:
                n_valid_gbif += _update_source_counts(
                    source_counts_per_trait=source_counts_per_trait,
                    source_key="n_valid_gbif",
                    payload=gbif_payload,
                    y_pred=y_pred,
                    traits=traits,
                    n_bands=len(bands),
                )

            eval_payload = bundle.get(eval_dataset)
            if eval_payload is None:
                skipped_batches += 1
                continue

            y_eval = eval_payload["y"]
            src_eval = eval_payload["source_mask"]
            valid_eval = (
                torch.isfinite(y_eval) & torch.isfinite(y_pred) & (src_eval > 0)
            )

            if not bool(valid_eval.any()):
                skipped_batches += 1
                continue

            valid_test_batches += 1
            y_eval_np = y_eval.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            valid_eval_np = valid_eval.detach().cpu().numpy()
            for metric_pos, trait_idx in enumerate(available_trait_indices):
                trait_name = traits_for_metrics[metric_pos]
                start = trait_idx * len(bands)
                stop = start + len(bands)
                y_true_trait = denormalize_predictions(
                    y_eval_np[:, start:stop, :, :],
                    trait_name,
                    power_transform_params_available,
                    **denorm_kwargs_true,
                )
                y_pred_trait = denormalize_predictions(
                    y_pred_np[:, start:stop, :, :],
                    trait_name,
                    power_transform_params_available,
                    **denorm_kwargs,
                )
                valid_trait = valid_eval_np[:, start:stop, :, :]
                valid_trait = (
                    valid_trait
                    & np.isfinite(y_true_trait)
                    & np.isfinite(y_pred_trait)
                )
                if bool(valid_trait.any()):
                    y_true_parts[metric_pos].append(
                        y_true_trait[valid_trait].astype(np.float32, copy=False)
                    )
                    y_pred_parts[metric_pos].append(
                        y_pred_trait[valid_trait].astype(np.float32, copy=False)
                    )

    if valid_test_batches == 0 or not any(y_true_parts):
        raise RuntimeError("No valid test observations found.")

    # Filter source_counts_per_trait to only available traits
    source_counts_per_trait = {
        trait: source_counts_per_trait[trait]
        for trait in available_traits
        if trait in source_counts_per_trait
    }

    denormalized_trait_metrics, trait_samples = _summarize_denormalized_trait_parts(
        y_true_parts=y_true_parts,
        y_pred_parts=y_pred_parts,
        traits=traits_for_metrics,
    )
    denormalized_macro_metrics = _compute_macro_denormalized_metrics(
        denormalized_trait_metrics, trait_samples
    )

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem
    prediction_out_dir = Path(
        str(OmegaConf.select(cfg, "prediction_out_dir") or output_dir)
    )
    prediction_run_dir = prediction_out_dir / stem

    # Write denormalized metrics to CSV (sorted by trait_id)
    csv_path = prediction_run_dir / f"{stem}.test_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_traits = sorted(
        traits_for_metrics, key=lambda x: int(x) if x.isdigit() else float("inf")
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trait_id",
                "n_valid",
                "rmse",
                "r2",
                "pearson_r",
                "mae",
                "nrmse",
                "residual_std",
                "target_std",
                "prediction_std",
            ]
        )
        for trait_name in sorted_traits:
            metrics = denormalized_trait_metrics[trait_name]
            writer.writerow(
                [
                    trait_name,
                    metrics["n_valid"],
                    metrics["rmse"],
                    metrics["r2"],
                    metrics["pearson_r"],
                    metrics["mae"],
                    metrics["nrmse"],
                    metrics["residual_std"],
                    metrics["target_std"],
                    metrics["prediction_std"],
                ]
            )

    console.print(
        f"[green]Denormalized metrics CSV written:[/green] [cyan]{csv_path}[/cyan]"
    )
    summary_json_path = prediction_run_dir / f"{stem}.test_metrics_summary.json"
    n_valid_total = int(sum(m["n_valid"] for m in denormalized_trait_metrics.values()))
    summary_payload = {
        "checkpoint": str(checkpoint_path),
        "split": test_split,
        "mode": target_layout["mode"],
        "eval_dataset": eval_dataset,
        "n_valid": n_valid_total,
        "n_valid_splot": int(n_valid_splot),
        "n_valid_gbif": int(n_valid_gbif),
        "denormalized_macro_metrics": denormalized_macro_metrics,
        "metrics_csv": str(csv_path),
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
    console.print(
        f"[green]Denormalized metrics summary written:[/green] [cyan]{summary_json_path}[/cyan]"
    )

    out_tif_paths: list[Path] = []
    write_all_map_cfg = OmegaConf.select(cfg, "write_all_map")
    write_all_map = True if write_all_map_cfg is None else bool(write_all_map_cfg)
    if write_all_map:
        all_zarr_path = _resolve_store_path(zarr_dir, "all")
        all_store = _open_chip_store(all_zarr_path)
        console.print(f"All-map store: [cyan]{all_zarr_path}[/cyan]")

        legacy_tif_name = OmegaConf.select(cfg, "prediction_tif_name")
        if legacy_tif_name is not None:
            console.print(
                "[yellow]prediction_tif_name is ignored for 1km per-trait exports; "
                f"writing to {prediction_run_dir} instead.[/yellow]"
            )
        out_tif_paths = _write_per_trait_prediction_tifs_streaming(
            model=model,
            all_store=all_store,
            predictors=predictors,
            traits=traits,
            bands=bands,
            power_transform_params=power_transform_params,
            denorm_kwargs=denorm_kwargs,
            out_dir=prediction_run_dir,
            batch_size=int(
                OmegaConf.select(cfg, "prediction_batch_size")
                or train_cfg.data_loaders.batch_size
            ),
            device=device,
            supervision_mode=supervision_mode,
            center_crop_size=center_crop_size,
            predictor_validity_mode=predictor_validity_mode,
            predictor_min_finite_ratio=predictor_min_finite_ratio,
            overwrite=bool(OmegaConf.select(cfg, "overwrite") or False),
            keep_temp=bool(OmegaConf.select(cfg, "keep_temp") or False),
            final_block_size=int(OmegaConf.select(cfg, "final_block_size") or 512),
        )
        console.print(
            "[green]Denormalized per-trait prediction TIFs written:[/green] "
            f"[cyan]{prediction_run_dir}[/cyan] ({len(out_tif_paths)} files)"
        )

    if wandb_run is not None:
        wandb_summary_dict = {
            "test/denorm_macro_rmse": denormalized_macro_metrics["rmse_macro"],
            "test/denorm_macro_nrmse": denormalized_macro_metrics["nrmse_macro"],
            "test/denorm_macro_r2": denormalized_macro_metrics["r2_macro"],
            "test/denorm_macro_pearson_r": denormalized_macro_metrics[
                "pearson_r_macro"
            ],
            "test/denorm_macro_mae": denormalized_macro_metrics["mae_macro"],
            "test/denorm_macro_residual_std": denormalized_macro_metrics[
                "residual_std_macro"
            ],
            "test/denorm_n_valid": int(
                sum(m["n_valid"] for m in denormalized_trait_metrics.values())
            ),
            "test/denorm_n_valid_splot": int(n_valid_splot),
            "test/denorm_n_valid_gbif": int(n_valid_gbif),
        }
        # Log with commit=True to create new "test" section in W&B
        wandb.log(wandb_summary_dict, commit=True)
        wandb.summary.update(wandb_summary_dict)
        console.print(
            f"[green]W&B Summary + Log updated:[/green] {list(wandb_summary_dict.keys())}"
        )

    console.print(f"checkpoint: [cyan]{checkpoint_path}[/cyan]")
    console.print(f"split:      [cyan]{test_split}[/cyan]")
    console.print(f"mode:       [cyan]{target_layout['mode']}[/cyan]")
    console.print(f"eval_data:  [cyan]{eval_dataset}[/cyan]")
    console.print(f"n_valid:    [cyan]{n_valid_total}[/cyan]")
    console.print(f"n_valid_splot: [cyan]{n_valid_splot}[/cyan]")
    console.print(f"n_valid_gbif:  [cyan]{n_valid_gbif}[/cyan]")
    console.print(f"\n[bold]Denormalized Macro Metrics:[/bold]")
    console.print(
        f"macro_rmse: [cyan]{denormalized_macro_metrics['rmse_macro']:.6f}[/cyan]"
    )
    console.print(
        f"macro_nrmse: [cyan]{denormalized_macro_metrics['nrmse_macro']:.6f}[/cyan]"
    )
    console.print(
        f"macro_r2:   [cyan]{denormalized_macro_metrics['r2_macro']:.6f}[/cyan]"
    )
    console.print(
        f"macro_r:    [cyan]{denormalized_macro_metrics['pearson_r_macro']:.6f}[/cyan]"
    )
    console.print(
        f"macro_mae:  [cyan]{denormalized_macro_metrics['mae_macro']:.6f}[/cyan]"
    )
    console.print(
        f"macro_residual_std: [cyan]{denormalized_macro_metrics['residual_std_macro']:.6f}[/cyan]"
    )
    console.print(f"metrics_csv:[cyan]{csv_path}[/cyan]")
    console.print(f"metrics_summary:[cyan]{summary_json_path}[/cyan]")
    if out_tif_paths:
        console.print(f"prediction_dir:[cyan]{prediction_run_dir}[/cyan]")
    else:
        console.print("all-map tif:[yellow]skipped (write_all_map=false)[/yellow]")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
