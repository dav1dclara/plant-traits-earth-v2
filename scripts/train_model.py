import math
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from ptev2.data.dataloader import get_dataloader
from ptev2.utils import (
    checkpoint_paths_from_cfg,
    run_name_from_cfg,
    seed_all,
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _get_scheduler_metric(
    metrics: dict[str, float],
    metric_name: str,
) -> float:
    value = float(metrics.get(metric_name, float("nan")))
    if math.isfinite(value):
        return value
    fallback = float(metrics.get("val_loss_comb_weighted", float("nan")))
    return fallback


def _step_scheduler(
    scheduler: object | None,
    scheduler_mode: str,
    scheduler_metric_name: str,
    metrics: dict[str, float],
) -> None:
    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(_get_scheduler_metric(metrics, scheduler_metric_name))
        return

    if scheduler_mode in {"epoch", "metric"}:
        scheduler.step()
        return

    raise ValueError(
        f"Unsupported training.scheduler_step.mode='{scheduler_mode}'. Use 'epoch' or 'metric'."
    )


def _monitor_value_for_checkpoint(
    metrics: dict[str, float],
    monitor_metric: str,
) -> tuple[float, str]:
    value = float(metrics.get(monitor_metric, float("nan")))
    if math.isfinite(value):
        return value, monitor_metric

    fallback_metric = "val_loss_comb_weighted"
    fallback_value = float(metrics.get(fallback_metric, float("nan")))
    return fallback_value, fallback_metric


def _target_layout_from_cfg(
    cfg: DictConfig,
) -> tuple[int, list[str], list[int], list[int]]:
    target_cfg = cfg.training.data.target
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


def _source_mask_nodata_values(cfg: DictConfig) -> list[int]:
    raw = cfg.training.data.target.source_mask_nodata_values
    if raw is None:
        return []
    if isinstance(raw, (int, str)):
        return [int(raw)]
    return [int(v) for v in raw]


def _split_target_and_source(
    y_full: torch.Tensor,
    target_indices: list[int],
    source_indices: list[int],
    nodata_values: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if y_full.ndim != 4:
        raise ValueError(
            f"Expected y_full shape (B, C, H, W), got {tuple(y_full.shape)}"
        )

    y_target = y_full[:, target_indices]
    source_mask = (
        torch.nan_to_num(y_full[:, source_indices], nan=0.0).round().to(torch.int64)
    )
    for nodata_value in nodata_values:
        source_mask = torch.where(
            source_mask == nodata_value,
            torch.zeros_like(source_mask),
            source_mask,
        )
    return y_target, source_mask


def train_model(cfg: DictConfig) -> float:
    seed_all(cfg.training.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    run_name = run_name_from_cfg(cfg)
    print(f"Run name: {run_name}")

    n_traits, selected_statistics, target_channel_indices, source_indices = (
        _target_layout_from_cfg(cfg)
    )
    print(
        "Target layout: "
        f"traits={n_traits}, "
        f"selected_statistics={selected_statistics}, "
        f"output_channels={len(target_channel_indices)}"
    )

    run = None
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=False),
            reinit="finish_previous",
        )

    predictors = [k for k, v in cfg.training.data.predictors.items() if v.use]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    zarr_dir = Path(cfg.training.data.zarr_dir)
    target_name = str(cfg.training.data.target.source)
    mask_nodata_values = _source_mask_nodata_values(cfg)

    batch_size = int(cfg.training.data_loaders.batch_size)
    num_workers = int(cfg.training.data_loaders.num_workers)

    train_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=str(cfg.training.data.train_split),
        predictors=predictors,
        target=target_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=str(cfg.training.data.val_split),
        predictors=predictors,
        target=target_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = instantiate(cfg.models.active, out_channels=len(target_channel_indices)).to(
        device
    )
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    loss_fn = instantiate(cfg.training.loss)
    if not hasattr(loss_fn, "loss_components"):
        raise TypeError(
            "training.loss must implement loss_components(prediction, target, source_mask)."
        )

    scheduler_cfg = cfg.training.scheduler
    scheduler = (
        instantiate(scheduler_cfg, optimizer=optimizer)
        if scheduler_cfg is not None
        else None
    )
    scheduler_mode = str(cfg.training.scheduler_step.mode).lower()
    scheduler_metric_name = str(cfg.training.scheduler_step.metric)

    grad_clip_norm_cfg = cfg.training.gradient_clip_norm
    grad_clip_norm = (
        float(grad_clip_norm_cfg) if grad_clip_norm_cfg is not None else None
    )

    checkpoint_save_model = bool(cfg.training.checkpoint.save_model)
    best_checkpoint_path, last_checkpoint_path = checkpoint_paths_from_cfg(
        cfg,
        run_name=run_name,
    )
    checkpoint_dir = best_checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_monitor_metric = str(cfg.training.checkpoint.monitor_metric)
    checkpoint_monitor_mode = str(cfg.training.checkpoint.monitor_mode).lower()
    if checkpoint_monitor_mode != "min":
        raise ValueError(
            "Only monitor_mode='min' is currently supported for checkpointing."
        )

    early_stopping_enabled = bool(cfg.training.early_stopping.enabled)
    early_stopping_patience = int(cfg.training.early_stopping.patience)
    early_stopping_min_delta = float(cfg.training.early_stopping.min_delta)

    log_val_mae_splot = bool(cfg.training.logging.log_val_mae_splot)

    splot_code = int(cfg.data.source_mask_encoding.splot)

    best_monitor_value = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        train_num_total = 0.0
        train_den_total = 0.0
        train_valid_pixels = 0
        train_skipped_batches = 0
        printed_shapes = False

        for X, y_full in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [train]"
        ):
            X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
            y_full = y_full.to(device=device, dtype=torch.float32)
            y, source_mask = _split_target_and_source(
                y_full=y_full,
                target_indices=target_channel_indices,
                source_indices=source_indices,
                nodata_values=mask_nodata_values,
            )

            valid_comb = torch.isfinite(y) & (source_mask > 0)
            batch_valid_pixels = int(valid_comb.sum().item())
            train_valid_pixels += batch_valid_pixels
            if batch_valid_pixels == 0:
                train_skipped_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(X)
            if y_pred.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                )

            if not printed_shapes:
                print(
                    "Batch shapes: "
                    f"X={tuple(X.shape)}, y={tuple(y.shape)}, source_mask={tuple(source_mask.shape)}"
                )
                printed_shapes = True

            batch_num_t, batch_den_t = loss_fn.loss_components(y_pred, y, source_mask)
            batch_den = float(batch_den_t.detach().item())
            if batch_den <= 0.0:
                train_skipped_batches += 1
                continue

            loss = batch_num_t / batch_den_t
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            optimizer.step()

            train_num_total += float(batch_num_t.detach().item())
            train_den_total += batch_den

        model.eval()
        val_num_total = 0.0
        val_den_total = 0.0
        val_valid_pixels = 0
        val_skipped_batches = 0

        splot_sqerr_sum = 0.0
        splot_abserr_sum = 0.0
        splot_count = 0

        with torch.no_grad():
            for X, y_full in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [val]"
            ):
                X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
                y_full = y_full.to(device=device, dtype=torch.float32)
                y, source_mask = _split_target_and_source(
                    y_full=y_full,
                    target_indices=target_channel_indices,
                    source_indices=source_indices,
                    nodata_values=mask_nodata_values,
                )

                valid_comb = torch.isfinite(y) & (source_mask > 0)
                batch_valid_pixels = int(valid_comb.sum().item())
                val_valid_pixels += batch_valid_pixels
                if batch_valid_pixels == 0:
                    val_skipped_batches += 1
                    continue

                y_pred = model(X)
                if y_pred.shape != y.shape:
                    raise ValueError(
                        f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                    )

                batch_num_t, batch_den_t = loss_fn.loss_components(
                    y_pred, y, source_mask
                )
                batch_den = float(batch_den_t.detach().item())
                if batch_den > 0.0:
                    val_num_total += float(batch_num_t.detach().item())
                    val_den_total += batch_den

                valid_splot = (
                    torch.isfinite(y_pred)
                    & torch.isfinite(y)
                    & (source_mask == splot_code)
                )
                batch_splot_count = int(valid_splot.sum().item())
                if batch_splot_count > 0:
                    diff = y_pred - y
                    splot_sqerr_sum += float(diff[valid_splot].pow(2).sum().item())
                    splot_abserr_sum += float(diff[valid_splot].abs().sum().item())
                    splot_count += batch_splot_count

        train_loss_comb_weighted = _safe_ratio(train_num_total, train_den_total)
        val_loss_comb_weighted = _safe_ratio(val_num_total, val_den_total)
        val_rmse_splot_overall = (
            math.sqrt(splot_sqerr_sum / splot_count)
            if splot_count > 0
            else float("nan")
        )
        val_mae_splot_overall = (
            splot_abserr_sum / splot_count if splot_count > 0 else float("nan")
        )

        metrics = {
            "train_loss_comb_weighted": train_loss_comb_weighted,
            "val_loss_comb_weighted": val_loss_comb_weighted,
            "val_rmse_splot_overall": val_rmse_splot_overall,
            "train_valid_pixels": float(train_valid_pixels),
            "val_valid_pixels": float(val_valid_pixels),
            "val_splot_pixels": float(splot_count),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if log_val_mae_splot:
            metrics["val_mae_splot_overall"] = val_mae_splot_overall

        checkpoint_monitor_value, monitor_used = _monitor_value_for_checkpoint(
            metrics,
            checkpoint_monitor_metric,
        )
        metrics["checkpoint_monitor_value"] = checkpoint_monitor_value

        _step_scheduler(
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
            scheduler_metric_name=scheduler_metric_name,
            metrics=metrics,
        )

        improved = math.isfinite(
            checkpoint_monitor_value
        ) and checkpoint_monitor_value < (best_monitor_value - early_stopping_min_delta)
        if improved:
            best_monitor_value = checkpoint_monitor_value
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if checkpoint_save_model:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "metrics": metrics,
                        "monitor_metric": checkpoint_monitor_metric,
                        "monitor_metric_used": monitor_used,
                        "target_channel_indices": target_channel_indices,
                    },
                    best_checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if checkpoint_save_model:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": metrics,
                    "target_channel_indices": target_channel_indices,
                },
                last_checkpoint_path,
            )

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} - "
            f"train_loss_comb_weighted={train_loss_comb_weighted:.6f}, "
            f"val_loss_comb_weighted={val_loss_comb_weighted:.6f}, "
            f"val_rmse_splot_overall={val_rmse_splot_overall:.6f}, "
            f"valid_pixels(train/val/splot)={train_valid_pixels}/{val_valid_pixels}/{splot_count}, "
            f"skipped(train/val)={train_skipped_batches}/{val_skipped_batches}, "
            f"monitor({monitor_used})={checkpoint_monitor_value:.6f}"
        )

        if run is not None:
            wandb.log({"epoch": epoch + 1, **metrics})

        if (
            early_stopping_enabled
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"patience={early_stopping_patience}, "
                f"min_delta={early_stopping_min_delta}, "
                f"best_epoch={best_epoch}, best_monitor={best_monitor_value:.6f}"
            )
            break

    if checkpoint_save_model and best_checkpoint_path.exists():
        state = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"Reloaded best checkpoint: {best_checkpoint_path}")

    if run is not None:
        run.finish()

    if math.isfinite(best_monitor_value):
        return best_monitor_value
    return float("nan")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    best_monitor = train_model(cfg)
    print(f"Training completed. Best monitor value: {best_monitor:.6f}")


if __name__ == "__main__":
    main()
