import csv
from pathlib import Path

import hydra
import numpy as np
import rasterio
import torch
import zarr
from affine import Affine
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rasterio.crs import CRS
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader

import wandb
from ptev2.data.dataloader import PlantTraitDataset, get_dataloader
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


def _resolve_power_transformer_params_csv(train_cfg: DictConfig) -> Path:
    zarr_dir = Path(str(train_cfg.data.zarr_dir)).resolve()
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


def _write_all_prediction_tif(
    preds: np.ndarray,
    all_store: zarr.Group,
    out_tif_path: Path,
    band_names: list[str],
) -> None:
    t = all_store.attrs["transform"]
    geo_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs_epsg = int(all_store.attrs["crs_epsg"])
    pixel_w = float(geo_transform.a)
    pixel_h = float(abs(geo_transform.e))

    bounds = all_store["bounds"][:]
    n_chips, out_ch, h_out, w_out = preds.shape

    canvas_origin_x = geo_transform.c
    canvas_origin_y = geo_transform.f
    canvas_rows = int(all_store.attrs["raster_height"])
    canvas_cols = int(all_store.attrs["raster_width"])

    _yw = np.minimum(np.arange(1, h_out + 1), np.arange(h_out, 0, -1)).astype(
        np.float32
    )
    _xw = np.minimum(np.arange(1, w_out + 1), np.arange(w_out, 0, -1)).astype(
        np.float32
    )
    weight_map = np.outer(_yw, _xw).astype(np.float32) ** 2

    canvas = np.zeros((out_ch, canvas_rows, canvas_cols), dtype=np.float32)
    weight_sum = np.zeros((out_ch, canvas_rows, canvas_cols), dtype=np.float32)

    for i in range(n_chips):
        min_x, _, _, max_y = bounds[i]
        col = round((min_x - canvas_origin_x) / pixel_w)
        row = round((canvas_origin_y - max_y) / pixel_h)

        r0, r1 = max(row, 0), min(row + h_out, canvas_rows)
        c0, c1 = max(col, 0), min(col + w_out, canvas_cols)
        if r0 >= r1 or c0 >= c1:
            continue

        chip = preds[i, :, r0 - row : r1 - row, c0 - col : c1 - col]
        finite = np.isfinite(chip)
        w = weight_map[None, r0 - row : r1 - row, c0 - col : c1 - col] * finite
        canvas[:, r0:r1, c0:c1] += (
            np.nan_to_num(chip, nan=0.0, posinf=0.0, neginf=0.0) * w
        )
        weight_sum[:, r0:r1, c0:c1] += w

    output = np.full_like(canvas, np.nan)
    nonzero = weight_sum > 0
    output[nonzero] = canvas[nonzero] / weight_sum[nonzero]

    out_geo_transform = Affine(
        pixel_w, 0, canvas_origin_x, 0, -pixel_h, canvas_origin_y
    )
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_tif_path,
        "w",
        driver="GTiff",
        height=canvas_rows,
        width=canvas_cols,
        count=out_ch,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=out_geo_transform,
        compress="deflate",
        tiled=True,
        nodata=np.nan,
    ) as dst:
        dst.write(output)
        for b_idx, name in enumerate(band_names, start=1):
            dst.set_band_description(b_idx, name)


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


@hydra.main(
    version_base=None, config_path="../../config/22km/test", config_name="default"
)
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
    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")
    target_layout = build_target_layout(train_cfg, train_store)
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

    test_split = str(OmegaConf.select(cfg, "test_split") or "test")
    test_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=test_split,
        predictors=predictors,
        target_layouts=target_layout["layouts"],
        return_target_bundle=True,
        batch_size=int(train_cfg.data_loaders.batch_size),
        num_workers=int(train_cfg.data_loaders.num_workers),
    )

    model_cfg = resolve_model_cfg(train_cfg)
    checkpoint_state = state.get("state_dict", state)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Checkpoint state for '{checkpoint_path}' is invalid.")

    total_pred_bands = sum(
        train_store[f"predictors/{name}"].shape[1] for name in predictors
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

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    skipped_batches = 0

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

            all_y_pred.append(y_pred.detach().cpu().numpy())
            all_y_true.append(y_eval.detach().cpu().numpy())
            all_valid.append(valid_eval.detach().cpu().numpy())

    if not all_y_true:
        raise RuntimeError("No valid test observations found.")

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    valid_mask = np.concatenate(all_valid, axis=0)

    # Filter to only available traits (exclude traits without power-transform params)
    # Predictions have shape (batch, traits*bands, h, w), need to filter trait dimension
    y_true_filtered_list = []
    y_pred_filtered_list = []
    valid_mask_filtered_list = []
    for trait_idx in available_trait_indices:
        start = trait_idx * len(bands)
        stop = start + len(bands)
        y_true_filtered_list.append(y_true[:, start:stop, :, :])
        y_pred_filtered_list.append(y_pred[:, start:stop, :, :])
        valid_mask_filtered_list.append(valid_mask[:, start:stop, :, :])

    y_true = np.concatenate(y_true_filtered_list, axis=1)
    y_pred = np.concatenate(y_pred_filtered_list, axis=1)
    valid_mask = np.concatenate(valid_mask_filtered_list, axis=1)

    # Update traits list for metrics calculation to only available traits
    traits_for_metrics = available_traits

    # Filter source_counts_per_trait to only available traits
    source_counts_per_trait = {
        trait: source_counts_per_trait[trait]
        for trait in available_traits
        if trait in source_counts_per_trait
    }

    denormalized_trait_metrics, trait_samples = _summarize_denormalized_trait_metrics(
        y_true=y_true,
        y_pred=y_pred,
        valid_mask=valid_mask,
        traits=traits_for_metrics,
        n_bands=len(bands),
        params_df=power_transform_params_available,
        denorm_kwargs=denorm_kwargs,
    )
    denormalized_macro_metrics = _compute_macro_denormalized_metrics(
        denormalized_trait_metrics, trait_samples
    )

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem

    # Write denormalized metrics to CSV (sorted by trait_id)
    csv_path = output_dir / f"{stem}.test_metrics.csv"
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

    out_tif_path = None
    write_all_map_cfg = OmegaConf.select(cfg, "write_all_map")
    write_all_map = True if write_all_map_cfg is None else bool(write_all_map_cfg)
    if write_all_map:
        all_zarr_path = zarr_dir / "all.zarr"
        if not all_zarr_path.exists():
            raise FileNotFoundError(f"all.zarr not found: {all_zarr_path}")
        all_store = zarr.open_group(str(all_zarr_path), mode="r")

        all_dataset = PlantTraitDataset(
            all_zarr_path,
            predictors=predictors,
            target_layouts=target_layout["layouts"],
            return_target_bundle=True,
        )
        all_loader = DataLoader(
            all_dataset,
            batch_size=int(train_cfg.data_loaders.batch_size),
            shuffle=False,
            num_workers=int(train_cfg.data_loaders.num_workers),
        )

        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for X, _ in track(
                all_loader, description=f"All-map [{checkpoint_path.stem}]"
            ):
                y_pred_map, valid_x, _ = _prepare_batch(
                    model,
                    X,
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
                all_preds.append(y_pred_map.detach().cpu().numpy())

        preds_all = np.concatenate(all_preds, axis=0)

        # Denormalize predictions
        preds_denorm = np.full_like(preds_all, float("nan"), dtype=np.float32)
        for trait_idx, trait_name in enumerate(traits):
            if trait_name not in available_traits:
                # Skip traits without transformation params
                continue
            start = trait_idx * len(bands)
            stop = start + len(bands)
            preds_denorm[:, start:stop, :, :] = denormalize_predictions(
                preds_all[:, start:stop, :, :],
                trait_name,
                power_transform_params,
                **denorm_kwargs,
            )

        prediction_out_dir = Path(
            str(OmegaConf.select(cfg, "prediction_out_dir") or output_dir)
        )
        prediction_filename = str(
            OmegaConf.select(cfg, "prediction_tif_name") or f"{stem}.tif"
        )
        out_tif_path = prediction_out_dir / prediction_filename
        band_names = [f"{trait}_{band}" for trait in traits for band in bands]
        _write_all_prediction_tif(preds_denorm, all_store, out_tif_path, band_names)
        console.print(
            f"[green]Denormalized predictions TIF written:[/green] [cyan]{out_tif_path}[/cyan]"
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
    n_valid_total = int(sum(m["n_valid"] for m in denormalized_trait_metrics.values()))
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
    if out_tif_path is not None:
        console.print(f"all-map tif:[cyan]{out_tif_path}[/cyan]")
    else:
        console.print("all-map tif:[yellow]skipped (write_all_map=false)[/yellow]")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
