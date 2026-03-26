import math
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader
from ptev2.utils import (
    checkpoint_paths_from_cfg,
    run_name_from_cfg,
    seed_all,
)


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
    for source_value in valid_source_values:
        source_mask = torch.where(
            source_mask_raw == int(source_value),
            source_mask_raw,
            source_mask,
        )
    return y_target, source_mask


def train_model(cfg: DictConfig) -> float:
    train_cfg = cfg.training.train
    seed_all(int(train_cfg.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    run_name = run_name_from_cfg(cfg)
    print(f"Run name: {run_name}")

    run = None
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=False),
            reinit="finish_previous",
        )

    predictors = [
        name
        for name, predictor_cfg in cfg.training.data.predictors.items()
        if bool(predictor_cfg.use)
    ]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    target_cfg = cfg.training.data.targets
    target_name = str(target_cfg.source)
    source_encoding = target_cfg.source_encoding
    valid_source_values = [int(source_encoding.gbif), int(source_encoding.splot)]
    splot_code = int(source_encoding.splot)

    dataloader_cfg = {
        "zarr_dir": Path(cfg.training.data.zarr_dir),
        "predictors": predictors,
        "target": target_name,
        "batch_size": int(cfg.training.data_loaders.batch_size),
        "num_workers": int(cfg.training.data_loaders.num_workers),
    }
    train_loader = get_dataloader(
        split=str(cfg.training.data.train_split), **dataloader_cfg
    )
    val_loader = get_dataloader(
        split=str(cfg.training.data.val_split), **dataloader_cfg
    )

    available_statistics = [str(v) for v in target_cfg.available_statistics]
    selected_statistics = [str(v) for v in target_cfg.selected_statistics]
    stats_per_trait = len(available_statistics)
    zarr_channels = int(train_loader.dataset.store[f"targets/{target_name}"].shape[1])
    if zarr_channels % stats_per_trait != 0:
        raise ValueError(
            f"Target channel count ({zarr_channels}) is not divisible by number of "
            f"available statistics ({stats_per_trait})."
        )
    trait_ids = [int(v) for v in target_cfg.trait_ids] if target_cfg.trait_ids else []
    n_traits = len(trait_ids) if trait_ids else (zarr_channels // stats_per_trait)

    stat_to_idx = {name: idx for idx, name in enumerate(available_statistics)}
    source_idx = stat_to_idx[str(target_cfg.source_statistic)]
    target_channel_indices = [
        trait_pos * stats_per_trait + stat_to_idx[stat_name]
        for trait_pos in range(n_traits)
        for stat_name in selected_statistics
    ]
    source_indices = [
        trait_pos * stats_per_trait + source_idx for trait_pos in range(n_traits)
    ]

    print(
        "Target layout: "
        f"traits={n_traits}, "
        f"selected_statistics={selected_statistics}, "
        f"output_channels={len(target_channel_indices)}"
    )

    model = instantiate(cfg.models.active, out_channels=len(target_channel_indices)).to(
        device
    )
    optimizer = instantiate(train_cfg.optimizer, params=model.parameters())
    loss_fn = instantiate(train_cfg.loss)
    if not hasattr(loss_fn, "loss_components"):
        raise TypeError(
            "training.loss must implement loss_components(prediction, target, source_mask)."
        )

    scheduler_cfg = train_cfg.scheduler
    scheduler = (
        instantiate(scheduler_cfg, optimizer=optimizer)
        if scheduler_cfg is not None
        else None
    )
    scheduler_metric_name = str(train_cfg.scheduler_step.metric)

    grad_clip_norm_cfg = train_cfg.gradient_clip_norm
    grad_clip_norm = (
        float(grad_clip_norm_cfg) if grad_clip_norm_cfg is not None else None
    )

    checkpoint_save_model = bool(cfg.training.checkpoint.save_model)
    best_checkpoint_path, _ = checkpoint_paths_from_cfg(
        cfg,
        run_name=run_name,
    )
    checkpoint_dir = best_checkpoint_path.parent
    if checkpoint_save_model and not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}. "
            "Create it explicitly or set training.checkpoint.dir to an existing path."
        )

    checkpoint_monitor_metric = str(cfg.training.checkpoint.monitor_metric)
    checkpoint_monitor_mode = str(cfg.training.checkpoint.monitor_mode).lower()
    if checkpoint_monitor_mode != "min":
        raise ValueError(
            "Only monitor_mode='min' is currently supported for checkpointing."
        )

    early_stopping_enabled = bool(train_cfg.early_stopping.enabled)
    early_stopping_patience = int(train_cfg.early_stopping.patience)
    early_stopping_min_delta = float(train_cfg.early_stopping.min_delta)

    log_val_mae_splot = bool(train_cfg.logging.log_val_mae_splot)

    best_monitor_value = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(int(train_cfg.epochs)):
        model.train()
        train_num_total = 0.0
        train_den_total = 0.0
        train_valid_pixels = 0
        train_skipped_batches = 0
        train_nonfinite_output_batches = 0
        printed_shapes = False

        for X, y_full in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{train_cfg.epochs} [train]"
        ):
            X = X.to(device=device, dtype=torch.float32)
            valid = torch.isfinite(X).all(dim=1, keepdim=True)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.clamp(X, min=-1e4, max=1e4)
            y_full = y_full.to(device=device, dtype=torch.float32)
            y, source_mask = _split_target_and_source(
                y_full=y_full,
                target_indices=target_channel_indices,
                source_indices=source_indices,
                valid_source_values=valid_source_values,
            )
            y = torch.where(valid.expand_as(y), y, torch.nan)

            valid_comb = torch.isfinite(y) & (source_mask > 0)
            batch_valid_pixels = int(valid_comb.sum().item())
            train_valid_pixels += batch_valid_pixels
            if batch_valid_pixels == 0:
                train_skipped_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(X)
            if not torch.isfinite(y_pred).all():
                train_nonfinite_output_batches += 1
                train_skipped_batches += 1
                continue

            batch_num_t, batch_den_t = loss_fn.loss_components(y_pred, y, source_mask)
            if not torch.isfinite(batch_num_t) or not torch.isfinite(batch_den_t):
                train_skipped_batches += 1
                continue
            batch_den = float(batch_den_t.detach().item())
            if batch_den <= 0.0:
                train_skipped_batches += 1
                continue

            loss = batch_num_t / batch_den_t
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected during training. "
                    "Check input scaling/normalization and learning rate."
                )
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
        val_nonfinite_output_batches = 0

        comb_y_true: list[torch.Tensor] = []
        comb_y_pred: list[torch.Tensor] = []
        splot_y_true: list[torch.Tensor] = []
        splot_y_pred: list[torch.Tensor] = []

        with torch.no_grad():
            for X, y_full in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{train_cfg.epochs} [val]"
            ):
                X = X.to(device=device, dtype=torch.float32)
                valid = torch.isfinite(X).all(dim=1, keepdim=True)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = torch.clamp(X, min=-1e4, max=1e4)
                y_full = y_full.to(device=device, dtype=torch.float32)
                y, source_mask = _split_target_and_source(
                    y_full=y_full,
                    target_indices=target_channel_indices,
                    source_indices=source_indices,
                    valid_source_values=valid_source_values,
                )
                y = torch.where(valid.expand_as(y), y, torch.nan)

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
                if not torch.isfinite(y_pred).all():
                    val_nonfinite_output_batches += 1
                    val_skipped_batches += 1
                    continue

                batch_num_t, batch_den_t = loss_fn.loss_components(
                    y_pred, y, source_mask
                )
                if not torch.isfinite(batch_num_t) or not torch.isfinite(batch_den_t):
                    val_skipped_batches += 1
                    continue
                batch_den = float(batch_den_t.detach().item())
                if batch_den > 0.0:
                    val_num_total += float(batch_num_t.detach().item())
                    val_den_total += batch_den
                else:
                    val_skipped_batches += 1

                if batch_valid_pixels > 0:
                    comb_y_pred.append(y_pred[valid_comb].detach().cpu())
                    comb_y_true.append(y[valid_comb].detach().cpu())

                valid_splot = (
                    torch.isfinite(y_pred)
                    & torch.isfinite(y)
                    & (source_mask == splot_code)
                )
                if bool(valid_splot.any()):
                    splot_y_pred.append(y_pred[valid_splot].detach().cpu())
                    splot_y_true.append(y[valid_splot].detach().cpu())

        train_loss_comb_weighted = (
            float("nan")
            if train_den_total <= 0.0
            else (train_num_total / train_den_total)
        )
        val_loss_comb_weighted = (
            float("nan") if val_den_total <= 0.0 else (val_num_total / val_den_total)
        )
        if comb_y_true:
            y_true_comb = torch.cat(comb_y_true).numpy()
            y_pred_comb = torch.cat(comb_y_pred).numpy()
            comb_count = int(y_true_comb.size)
            val_rmse_comb_overall = float(
                math.sqrt(mean_squared_error(y_true_comb, y_pred_comb))
            )
            val_mae_comb_overall = float(mean_absolute_error(y_true_comb, y_pred_comb))
        else:
            comb_count = 0
            val_rmse_comb_overall = float("nan")
            val_mae_comb_overall = float("nan")

        if splot_y_true:
            y_true_splot = torch.cat(splot_y_true).numpy()
            y_pred_splot = torch.cat(splot_y_pred).numpy()
            splot_count = int(y_true_splot.size)
            val_rmse_splot_overall = float(
                math.sqrt(mean_squared_error(y_true_splot, y_pred_splot))
            )
            val_mae_splot_overall = float(
                mean_absolute_error(y_true_splot, y_pred_splot)
            )
        else:
            splot_count = 0
            val_rmse_splot_overall = float("nan")
            val_mae_splot_overall = float("nan")

        metrics = {
            "train_loss_comb_weighted": train_loss_comb_weighted,
            "val_loss_comb_weighted": val_loss_comb_weighted,
            "val_rmse_comb_overall": val_rmse_comb_overall,
            "val_rmse_splot_overall": val_rmse_splot_overall,
            "train_valid_pixels": float(train_valid_pixels),
            "val_valid_pixels": float(val_valid_pixels),
            "val_comb_pixels": float(comb_count),
            "val_splot_pixels": float(splot_count),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if log_val_mae_splot:
            metrics["val_mae_comb_overall"] = val_mae_comb_overall
            metrics["val_mae_splot_overall"] = val_mae_splot_overall

        monitor_used = checkpoint_monitor_metric
        checkpoint_monitor_value = float(metrics.get(monitor_used, float("nan")))
        if not math.isfinite(checkpoint_monitor_value):
            monitor_used = "val_loss_comb_weighted"
            checkpoint_monitor_value = float(metrics.get(monitor_used, float("nan")))
        metrics["checkpoint_monitor_value"] = checkpoint_monitor_value

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_value = float(
                    metrics.get(scheduler_metric_name, float("nan"))
                )
                if not math.isfinite(scheduler_value):
                    scheduler_value = float(
                        metrics.get("val_loss_comb_weighted", float("nan"))
                    )
                scheduler.step(scheduler_value)
            else:
                scheduler.step()

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

        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs} - "
            f"train_loss_comb_weighted={train_loss_comb_weighted:.6f}, "
            f"val_loss_comb_weighted={val_loss_comb_weighted:.6f}, "
            f"val_rmse_comb_overall={val_rmse_comb_overall:.6f}, "
            f"val_rmse_splot_overall={val_rmse_splot_overall:.6f}, "
            f"valid_pixels(train/val/comb/splot)={train_valid_pixels}/{val_valid_pixels}/{comb_count}/{splot_count}, "
            f"skipped(train/val)={train_skipped_batches}/{val_skipped_batches}, "
            f"nonfinite_output_batches(train/val)={train_nonfinite_output_batches}/{val_nonfinite_output_batches}, "
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
