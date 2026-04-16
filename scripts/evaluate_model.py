import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader
from ptev2.metrics.aoa import collect_patch_features, compute_aoa_metrics
from ptev2.metrics.core import pearson_r as pearson_r_metric
from ptev2.metrics.core import r2_score
from ptev2.utils import (
    checkpoint_paths_from_cfg,
    run_name_from_cfg,
    seed_all,
)


def _infer_checkpoint_output_channels(
    state_dict: dict[str, torch.Tensor],
) -> int | None:
    for key, tensor in state_dict.items():
        if key.endswith("head.bias") and tensor.ndim == 1:
            return int(tensor.shape[0])
    for key, tensor in state_dict.items():
        if key.endswith("head.weight") and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_override = cfg.evaluation.checkpoint_path
    if checkpoint_override:
        return Path(str(checkpoint_override))

    best_path, last_path = checkpoint_paths_from_cfg(cfg)
    if best_path.exists():
        return best_path
    return last_path


def _target_layout_from_cfg(
    cfg: DictConfig,
) -> tuple[int, list[str], list[int], list[int]]:
    target_cfg = cfg.training.data.targets
    available_statistics = [str(v) for v in target_cfg.available_statistics]
    selected_statistics = [str(v) for v in target_cfg.selected_statistics]
    source_statistic = str(target_cfg.source_statistic)

    if not available_statistics:
        raise ValueError("training.data.target.available_statistics must not be empty.")
    if not selected_statistics:
        raise ValueError("training.data.target.selected_statistics must not be empty.")
    if source_statistic not in available_statistics:
        raise ValueError(
            f"source_statistic='{source_statistic}' not in {available_statistics}"
        )

    selected_trait_ids = (
        [int(v) for v in target_cfg.trait_ids] if target_cfg.trait_ids else []
    )
    n_traits = (
        len(selected_trait_ids) if selected_trait_ids else int(target_cfg.n_traits)
    )
    if n_traits <= 0:
        raise ValueError("training.data.target.n_traits must be > 0.")

    stats_per_trait = len(available_statistics)
    stat_to_idx = {name: idx for idx, name in enumerate(available_statistics)}
    target_indices: list[int] = []
    source_indices: list[int] = []
    for trait_pos in range(n_traits):
        for stat_name in selected_statistics:
            target_indices.append(trait_pos * stats_per_trait + stat_to_idx[stat_name])
        source_indices.append(
            trait_pos * stats_per_trait + stat_to_idx[source_statistic]
        )

    return n_traits, selected_statistics, target_indices, source_indices


def _split_target_and_source(
    y_full: torch.Tensor,
    target_indices: list[int],
    source_indices: list[int],
    valid_source_values: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if y_full.ndim != 4:
        raise ValueError(
            f"Expected y_full shape (B, C, H, W), got {tuple(y_full.shape)}"
        )

    y_target = y_full[:, target_indices]
    source_mask_raw = (
        torch.nan_to_num(y_full[:, source_indices], nan=0.0).round().to(torch.int64)
    )
    source_mask = torch.zeros_like(source_mask_raw)
    for value in valid_source_values:
        source_mask = torch.where(
            source_mask_raw == int(value),
            source_mask_raw,
            source_mask,
        )
    return y_target, source_mask


def evaluate_model(cfg: DictConfig) -> dict:
    train_cfg = cfg.training.train
    seed_all(int(train_cfg.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    run = None
    if cfg.wandb.enabled:
        eval_run_name = f"eval_{run_name_from_cfg(cfg)}"
        try:
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=eval_run_name,
                job_type="evaluation",
                config=OmegaConf.to_container(cfg, resolve=False),
                reinit="finish_previous",
            )
            print(f"W&B eval run started: {eval_run_name}")
        except Exception as exc:
            print(f"W&B init failed ({exc}). Continuing without W&B logging.")
            run = None

    predictors = [k for k, v in cfg.training.data.predictors.items() if v.use]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    n_traits, selected_statistics, target_channel_indices, source_indices = (
        _target_layout_from_cfg(cfg)
    )
    print(
        "Target layout: "
        f"traits={n_traits}, "
        f"selected_statistics={selected_statistics}, "
        f"output_channels={len(target_channel_indices)}"
    )

    source_encoding_cfg = cfg.training.data.targets.source_encoding
    source_valid_values = sorted(
        {
            int(source_encoding_cfg.gbif),
            int(source_encoding_cfg.splot),
        }
    )

    zarr_dir = Path(cfg.training.data.zarr_dir)
    test_zarr_dir_cfg = cfg.evaluation.test_zarr_dir
    test_zarr_dir = Path(str(test_zarr_dir_cfg)) if test_zarr_dir_cfg else zarr_dir
    batch_size = int(cfg.training.data_loaders.batch_size)
    num_workers = int(cfg.training.data_loaders.num_workers)
    test_split = str(cfg.evaluation.test_split)

    test_loader = get_dataloader(
        zarr_dir=test_zarr_dir,
        split=test_split,
        predictors=predictors,
        target=str(cfg.training.data.targets.source),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    compute_aoa = bool(cfg.evaluation.compute_aoa)
    aoa_metrics = None
    if compute_aoa:
        train_loader = get_dataloader(
            zarr_dir=zarr_dir,
            split=str(cfg.training.data.train_split),
            predictors=predictors,
            target=str(cfg.training.data.targets.source),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        train_features = collect_patch_features(train_loader, device=device)
        test_features = collect_patch_features(test_loader, device=device)
        aoa_metrics = compute_aoa_metrics(train_features, test_features, q=0.95)

    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass a custom path with +evaluation.checkpoint_path=/path/to/model.pth"
        )

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    checkpoint_output_channels = _infer_checkpoint_output_channels(state)
    if checkpoint_output_channels is not None and checkpoint_output_channels != len(
        target_channel_indices
    ):
        raise ValueError(
            "Checkpoint/model channel mismatch: "
            f"checkpoint_output_channels={checkpoint_output_channels}, "
            f"expected_output_channels={len(target_channel_indices)}."
        )

    model = instantiate(cfg.models.active, out_channels=len(target_channel_indices)).to(
        device
    )
    model.load_state_dict(state)
    model.eval()

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    skipped_batches = 0

    splot_code = int(cfg.training.data.targets.source_encoding.splot)
    splot_y_true: list[np.ndarray] = []
    splot_y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for X, y_full in tqdm(test_loader, desc=f"Evaluating [{test_split}]"):
            X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
            y_full = y_full.to(device=device, dtype=torch.float32)
            y, source_mask = _split_target_and_source(
                y_full=y_full,
                target_indices=target_channel_indices,
                source_indices=source_indices,
                valid_source_values=source_valid_values,
            )

            y_pred = model(X)
            if y_pred.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                )

            valid = torch.isfinite(y_pred) & torch.isfinite(y) & (source_mask > 0)
            if not bool(valid.any()):
                skipped_batches += 1
                continue

            all_y_pred.append(y_pred[valid].detach().cpu().numpy())
            all_y_true.append(y[valid].detach().cpu().numpy())

            valid_splot = (
                torch.isfinite(y_pred) & torch.isfinite(y) & (source_mask == splot_code)
            )
            if bool(valid_splot.any()):
                splot_y_pred.append(y_pred[valid_splot].detach().cpu().numpy())
                splot_y_true.append(y[valid_splot].detach().cpu().numpy())

    if all_y_true:
        y_true_vec = np.concatenate(all_y_true)
        y_pred_vec = np.concatenate(all_y_pred)
        n_valid = int(y_true_vec.size)
        r2 = float(r2_score(y_true_vec, y_pred_vec))
        pearson_r = float(pearson_r_metric(y_true_vec, y_pred_vec))
    else:
        n_valid = 0
        r2 = float("nan")
        pearson_r = float("nan")

    if splot_y_true:
        y_true_splot = np.concatenate(splot_y_true)
        y_pred_splot = np.concatenate(splot_y_pred)
        splot_count = int(y_true_splot.size)
        rmse_splot_overall = float(
            math.sqrt(mean_squared_error(y_true_splot, y_pred_splot))
        )
    else:
        splot_count = 0
        rmse_splot_overall = float("nan")

    results = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "split": test_split,
        "n_valid": n_valid,
        "skipped_batches": skipped_batches,
        "r2": r2,
        "pearson_r": pearson_r,
        "rmse_splot_overall": rmse_splot_overall,
        "n_splot_valid": int(splot_count),
    }
    if aoa_metrics is not None:
        results.update(aoa_metrics)

    output_path_cfg = cfg.evaluation.output_path
    output_path = (
        Path(str(output_path_cfg))
        if output_path_cfg
        else checkpoint_path.with_suffix(".test_metrics.json")
    )
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Evaluation output directory does not exist: {output_path.parent}. "
            "Create it explicitly or set evaluation.output_path to an existing directory."
        )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Evaluation finished")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  train_zarr:  {zarr_dir}")
    print(f"  test_zarr:   {test_zarr_dir}")
    print(f"  split:      {test_split}")
    print(f"  n_valid:    {n_valid}")
    print(f"  r2:         {r2:.6f}")
    print(f"  pearson_r:  {pearson_r:.6f}")
    print(f"  rmse_splot: {rmse_splot_overall:.6f}")
    print(f"  metrics:    {output_path}")

    if run is not None:
        wandb.log(results)
        run.summary["metrics_file"] = str(output_path)
        run.finish()

    return results


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
