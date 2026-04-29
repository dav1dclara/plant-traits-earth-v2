import math
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import track

from ptev2.data.dataloader import get_dataloader
from ptev2.metrics.evaluation import summarize_single_trait_metrics
from ptev2.utils import (
    apply_supervision_bundle,
    apply_supervision_tensor,
    build_target_layout,
    mask_bundle_targets_with_validity,
    move_bundle_to_device,
    predict_batch,
    resolve_device,
    resolve_model_cfg,
    resolve_train_zarr_dir,
    seed_all,
)

console = Console()


def _build_loss(cfg: DictConfig):
    """Instantiate SourceAwareLoss from config and target layout defaults."""
    loss_cfg = OmegaConf.create(OmegaConf.to_container(cfg.train.loss, resolve=False))

    alias_target = OmegaConf.select(loss_cfg, "target")
    if alias_target is not None:
        OmegaConf.update(loss_cfg, "_target_", alias_target, force_add=True)
        if "target" in loss_cfg:
            del loss_cfg["target"]

    loss_target = str(OmegaConf.select(loss_cfg, "_target_") or "")
    target_cfg = cfg.data.targets
    if not loss_target.endswith("SourceAwareLoss"):
        raise ValueError(
            "Current pipeline supports only ptev2.loss.SourceAwareLoss. "
            f"Got: {loss_target or '<missing _target_>'}"
        )

    lambda_from_loss = OmegaConf.select(loss_cfg, "lambda_gbif")
    # Backward-compatible alias for old CLI usage: w_gbif/w_splot -> lambda_gbif ratio.
    w_gbif = OmegaConf.select(loss_cfg, "w_gbif")
    w_splot = OmegaConf.select(loss_cfg, "w_splot")
    lambda_from_ratio = None
    if w_gbif is not None and w_splot is not None and float(w_splot) > 0.0:
        lambda_from_ratio = float(w_gbif) / float(w_splot)

    return instantiate(
        loss_cfg,
        mode=str(OmegaConf.select(target_cfg, "mode") or "splot_only"),
        primary_dataset=str(OmegaConf.select(target_cfg, "primary_dataset") or "splot"),
        auxiliary_dataset=str(
            OmegaConf.select(target_cfg, "auxiliary_dataset") or "gbif"
        ),
        lambda_gbif=float(
            lambda_from_loss
            if lambda_from_loss is not None
            else (lambda_from_ratio if lambda_from_ratio is not None else 0.1)
        ),
    )


def _resolve_selection_value(
    metric_name: str,
    val_loss: float,
    val_summary: dict[str, Any] | None,
) -> float:
    if metric_name == "val_loss":
        return float(val_loss)

    if val_summary is None:
        return float("nan")

    lookup = {
        "val_splot_macro_rmse": float(val_summary["macro_rmse"]),
        "val_splot_macro_pearson": float(val_summary["macro_pearson_r"]),
        "val_splot_rmse": float(val_summary["rmse"]),
    }
    if metric_name not in lookup:
        raise ValueError(
            "Unsupported train.selection.metric='{}'".format(metric_name)
            + ". Use val_splot_macro_rmse|val_splot_macro_pearson|val_splot_rmse|val_loss."
        )
    return lookup[metric_name]


def _is_better(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not math.isfinite(current):
        return False
    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError("train.selection.mode must be 'min' or 'max'.")


def _prepare_batch(
    model: torch.nn.Module,
    X: torch.Tensor,
    bundle: dict[str, dict[str, torch.Tensor]],
    *,
    device: torch.device,
    supervision_mode: str,
    center_crop_size: int,
    predictor_validity_mode: str,
    predictor_min_finite_ratio: float,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    X = X.to(device, dtype=torch.float32)
    bundle = move_bundle_to_device(bundle, device)

    y_pred, valid = predict_batch(
        model,
        X,
        validity_mode=predictor_validity_mode,
        min_finite_ratio=predictor_min_finite_ratio,
    )
    y_pred = apply_supervision_tensor(y_pred, supervision_mode, center_crop_size)
    valid = apply_supervision_tensor(valid, supervision_mode, center_crop_size)
    bundle = apply_supervision_bundle(bundle, supervision_mode, center_crop_size)
    bundle = mask_bundle_targets_with_validity(bundle, valid)
    return y_pred, bundle


@hydra.main(config_path="../config", config_name="training/default", version_base=None)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TRAINING[/bold cyan]")

    seed_all(cfg.train.seed)

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    console.print(f"Device: [cyan]{device}[/cyan]")

    console.print("[bold]\nData[/bold]")
    zarr_dir = resolve_train_zarr_dir(cfg)
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Zarr directory does not exist: {zarr_dir}")

    console.print(f"Resolution (km): [cyan]{cfg.data.resolution_km}[/cyan]")
    console.print(f"Patch size: [cyan]{cfg.data.patch_size}[/cyan]")
    console.print(f"Stride: [cyan]{cfg.data.stride}[/cyan]")
    console.print(f"Zarr directory: [cyan]{zarr_dir}[/cyan]")

    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")

    console.print("[bold]\nPredictors:[/bold]")
    predictors = [
        name
        for name, predictor_cfg in cfg.data.predictors.items()
        if bool(predictor_cfg.use)
    ]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    total_pred_bands = 0
    for predictor in predictors:
        n_bands = train_store[f"predictors/{predictor}"].shape[1]
        total_pred_bands += n_bands
        console.print(f"  - [cyan]{predictor}[/cyan] ({n_bands} bands)")
    console.print(f"  Total: [cyan]{total_pred_bands} bands[/cyan]")

    console.print("\n[bold]Targets:[/bold]")
    target_layout = build_target_layout(cfg, train_store)
    traits = list(target_layout["traits"])
    bands = list(target_layout["bands"])
    eval_dataset = str(target_layout["eval_dataset"])

    console.print(f"Mode: [cyan]{target_layout['mode']}[/cyan]")
    console.print(f"Train datasets: [cyan]{target_layout['active_datasets']}[/cyan]")
    console.print(f"Eval dataset: [cyan]{eval_dataset}[/cyan]")
    console.print(f"Traits ([cyan]{len(traits)}[/cyan]):")
    for trait in traits:
        console.print(f"  - {trait}")

    console.print("\n[bold]Data loaders:[/bold]")
    batch_size = int(cfg.data_loaders.batch_size)
    num_workers = int(cfg.data_loaders.num_workers)

    console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"Num workers: [cyan]{num_workers}[/cyan]")
    train_fraction = float(OmegaConf.select(cfg, "data_loaders.train_fraction") or 1.0)
    val_fraction = float(OmegaConf.select(cfg, "data_loaders.val_fraction") or 1.0)
    subset_seed = int(
        OmegaConf.select(cfg, "data_loaders.subset_seed") or int(cfg.train.seed)
    )
    console.print(f"Train fraction: [cyan]{train_fraction:.3f}[/cyan]")
    console.print(f"Val fraction: [cyan]{val_fraction:.3f}[/cyan]")

    dataloader_cfg = {
        "zarr_dir": zarr_dir,
        "predictors": predictors,
        "target_layouts": target_layout["layouts"],
        "return_target_bundle": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "subset_seed": subset_seed,
    }

    train_loader = get_dataloader(
        split="train", split_fraction=train_fraction, **dataloader_cfg
    )
    val_loader = get_dataloader(
        split="val", split_fraction=val_fraction, **dataloader_cfg
    )

    X_t, _ = next(iter(train_loader))
    X_v, _ = next(iter(val_loader))
    console.print(f"Train samples used: [cyan]{len(train_loader.dataset):,}[/cyan]")
    console.print(f"Val samples used: [cyan]{len(val_loader.dataset):,}[/cyan]")
    if X_t.shape[1:] != X_v.shape[1:]:
        raise ValueError(
            "X channel/spatial mismatch: "
            f"train={tuple(X_t.shape[1:])} vs val={tuple(X_v.shape[1:])}"
        )
    console.print(f"Predictor shape (C,H,W): [cyan]{tuple(X_t.shape[1:])}[/cyan]")

    model_cfg = resolve_model_cfg(cfg)
    out_channels = len(traits) * len(bands)

    console.print("\n[bold]Model and training configuration[/bold]")
    model = instantiate(
        model_cfg,
        in_channels=total_pred_bands,
        out_channels=out_channels,
    ).to(device)

    init_checkpoint_path = OmegaConf.select(cfg, "train.init_checkpoint_path")
    if init_checkpoint_path:
        init_state = torch.load(str(init_checkpoint_path), map_location=device)
        model_state = init_state.get("state_dict", init_state)
        model.load_state_dict(model_state, strict=False)
        console.print(f"Init checkpoint loaded: [cyan]{init_checkpoint_path}[/cyan]")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model:         [cyan]{model_cfg._target_}[/cyan]")
    console.print(f"  In channels:  [cyan]{total_pred_bands}[/cyan]")
    console.print(f"  Out channels: [cyan]{out_channels}[/cyan]")
    console.print(f"  Parameters:   [cyan]{n_params:,}[/cyan]")

    loss_fn = _build_loss(cfg)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    model_params = list(model.parameters())
    loss_params = []
    if isinstance(loss_fn, torch.nn.Module):
        loss_params = [p for p in loss_fn.parameters() if p.requires_grad]

    optimizer = instantiate(cfg.train.optimizer, params=model_params + loss_params)
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    grad_clip_norm = float(cfg.train.gradient_clip_norm)

    supervision_mode = str(OmegaConf.select(cfg, "train.supervision.mode") or "dense")
    center_crop_size = int(
        OmegaConf.select(cfg, "train.supervision.center_crop_size") or 3
    )
    predictor_validity_mode = str(
        OmegaConf.select(cfg, "train.predictor_validity.mode") or "min_fraction"
    )
    predictor_min_finite_ratio = float(
        OmegaConf.select(cfg, "train.predictor_validity.min_finite_ratio") or 0.2
    )
    selection_metric_name = str(
        OmegaConf.select(cfg, "train.selection.metric") or "val_splot_macro_rmse"
    )
    selection_mode = str(OmegaConf.select(cfg, "train.selection.mode") or "min")

    console.print(f"Loss:                  [cyan]{loss_fn.__class__.__name__}[/cyan]")
    console.print(f"Scheduler:             [cyan]{cfg.train.scheduler._target_}[/cyan]")
    console.print(f"Gradient clip norm:    [cyan]{grad_clip_norm}[/cyan]")
    console.print(
        f"Supervision:           [cyan]{supervision_mode} (center_crop_size={center_crop_size})[/cyan]"
    )
    console.print(
        "Predictor validity:    "
        f"[cyan]{predictor_validity_mode} (min_finite_ratio={predictor_min_finite_ratio})[/cyan]"
    )
    console.print(
        f"Selection metric:      [cyan]{selection_metric_name} ({selection_mode})[/cyan]"
    )

    early_stopping_enabled = bool(cfg.train.early_stopping.enabled)
    early_stopping_patience = int(cfg.train.early_stopping.patience)
    early_stopping_min_delta = float(cfg.train.early_stopping.min_delta)

    requested_run_name = OmegaConf.select(cfg, "train.run_name")
    if requested_run_name is not None:
        requested_run_name = str(requested_run_name)
    requested_run_group = OmegaConf.select(cfg, "train.group")
    if requested_run_group is not None:
        requested_run_group = str(requested_run_group)

    wandb_module = None
    if cfg.wandb.enabled:
        import wandb as wandb_module

        run = wandb_module.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=requested_run_name,
            group=requested_run_group,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        run_name = requested_run_name or run.name
        console.print(f"W&B logging enabled. Run: [cyan]{run_name}[/cyan]")
    else:
        run_name = requested_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        console.print("[yellow]W&B logging disabled.[/yellow]")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    config_path = checkpoint_dir / f"{run_name}.yaml"
    OmegaConf.save(cfg, config_path)

    best_selection = float("inf") if selection_mode == "min" else float("-inf")
    best_epoch = -1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(int(cfg.train.epochs)):
        console.rule(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        model.train()
        train_num_total = 0.0
        train_den_total = 0.0

        for X, bundle in track(train_loader, description="Training"):
            y_pred, bundle = _prepare_batch(
                model,
                X,
                bundle,
                device=device,
                supervision_mode=supervision_mode,
                center_crop_size=center_crop_size,
                predictor_validity_mode=predictor_validity_mode,
                predictor_min_finite_ratio=predictor_min_finite_ratio,
            )

            if not torch.isfinite(y_pred).all():
                continue

            optimizer.zero_grad(set_to_none=True)
            batch_num, batch_den, _ = loss_fn.loss_components(y_pred, bundle, None)
            if (
                not torch.isfinite(batch_num)
                or not torch.isfinite(batch_den)
                or batch_den <= 0.0
            ):
                continue

            loss = batch_num / batch_den
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            train_num_total += batch_num.detach().item()
            train_den_total += batch_den.detach().item()

        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss_avg = (
            train_num_total / train_den_total if train_den_total > 0.0 else float("nan")
        )
        console.print(f"  train_loss={train_loss_avg:.6f}")

        model.eval()
        val_num_total = 0.0
        val_den_total = 0.0
        val_y_true_parts = []
        val_y_pred_parts = []
        val_valid_parts = []

        with torch.no_grad():
            for X, bundle in track(val_loader, description="Validation"):
                y_pred, bundle = _prepare_batch(
                    model,
                    X,
                    bundle,
                    device=device,
                    supervision_mode=supervision_mode,
                    center_crop_size=center_crop_size,
                    predictor_validity_mode=predictor_validity_mode,
                    predictor_min_finite_ratio=predictor_min_finite_ratio,
                )

                if not torch.isfinite(y_pred).all():
                    continue

                batch_num, batch_den, _ = loss_fn.loss_components(y_pred, bundle, None)
                if (
                    torch.isfinite(batch_num)
                    and torch.isfinite(batch_den)
                    and batch_den > 0.0
                ):
                    val_num_total += batch_num.item()
                    val_den_total += batch_den.item()

                eval_payload = bundle.get(eval_dataset)
                if eval_payload is None:
                    continue

                y_eval = eval_payload["y"]
                src_eval = eval_payload["source_mask"]
                valid_eval = (
                    torch.isfinite(y_eval) & torch.isfinite(y_pred) & (src_eval > 0)
                )

                if not bool(valid_eval.any()):
                    continue

                val_y_true_parts.append(y_eval.detach().cpu())
                val_y_pred_parts.append(y_pred.detach().cpu())
                val_valid_parts.append(valid_eval.detach().cpu())

        val_loss_avg = (
            val_num_total / val_den_total if val_den_total > 0.0 else float("nan")
        )
        console.print(f"  val_loss={val_loss_avg:.6f}")

        val_metric_summary = None
        if val_y_true_parts:
            val_metric_summary = summarize_single_trait_metrics(
                y_true=torch.cat(val_y_true_parts, dim=0),
                y_pred=torch.cat(val_y_pred_parts, dim=0),
                source_mask=None,
                valid_mask=torch.cat(val_valid_parts, dim=0),
                trait_names=traits,
                n_bands=len(bands),
            )
            console.print(
                "  val_splot_rmse={rmse:.6f}  val_splot_macro_rmse={macro_rmse:.6f}  "
                "val_splot_macro_pearson={macro_pearson_r:.6f}".format(
                    rmse=val_metric_summary["rmse"],
                    macro_rmse=val_metric_summary["macro_rmse"],
                    macro_pearson_r=val_metric_summary["macro_pearson_r"],
                )
            )

        selection_value = _resolve_selection_value(
            selection_metric_name,
            val_loss_avg,
            val_metric_summary,
        )
        is_best = _is_better(
            selection_value, best_selection, selection_mode, early_stopping_min_delta
        )

        if is_best:
            best_selection = selection_value
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "selection_metric": selection_metric_name,
                    "selection_value": best_selection,
                    "in_channels": total_pred_bands,
                    "out_channels": out_channels,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "wandb_run_id": getattr(run, "id", None)
                    if cfg.wandb.enabled
                    else None,
                },
                best_checkpoint_path,
            )
            console.print(
                f"[green]New best model saved[/green] ({selection_metric_name}={best_selection:.6f})"
            )
        else:
            epochs_without_improvement += 1

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(selection_value)
        else:
            scheduler.step()

        if cfg.wandb.enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss_avg,
                "val/loss": val_loss_avg,
                "train/lr": current_lr,
                "val/selection_metric_name": selection_metric_name,
                "val/selection_metric_value": selection_value,
            }
            if val_metric_summary is not None:
                log_dict["val/splot/rmse"] = val_metric_summary["rmse"]
                log_dict["val/splot/r2"] = val_metric_summary["r2"]
                log_dict["val/splot/pearson_r"] = val_metric_summary["pearson_r"]
                log_dict["val/splot/macro_rmse"] = val_metric_summary["macro_rmse"]
                log_dict["val/splot/macro_pearson_r"] = val_metric_summary[
                    "macro_pearson_r"
                ]
            if wandb_module is not None:
                wandb_module.log(log_dict)

        if (
            early_stopping_enabled
            and epochs_without_improvement >= early_stopping_patience
        ):
            console.print(
                "[yellow]Early stopping triggered[/yellow] "
                f"(patience={early_stopping_patience}, best_epoch={best_epoch})"
            )
            break

    console.rule("[bold cyan]DONE[/bold cyan]")
    console.print(
        f"Best {selection_metric_name}={best_selection:.6f} at epoch {best_epoch}"
    )
    console.print(f"Checkpoint: [cyan]{best_checkpoint_path}[/cyan]")

    if cfg.wandb.enabled:
        run.summary["best_selection_metric_name"] = selection_metric_name
        run.summary["best_selection_metric_value"] = float(best_selection)
        run.summary["best_val_loss"] = (
            float(best_val_loss) if math.isfinite(best_val_loss) else None
        )
        run.summary["best_epoch"] = int(best_epoch)
        run.summary["checkpoint_path"] = str(best_checkpoint_path)
        run.finish()


if __name__ == "__main__":
    main()
