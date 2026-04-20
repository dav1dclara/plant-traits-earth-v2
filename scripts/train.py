import math
from datetime import datetime
from pathlib import Path

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
    build_target_layout,
    predict_batch,
    resolve_device,
    resolve_model_cfg,
    resolve_train_zarr_dir,
    seed_all,
)

console = Console()


def _build_loss(cfg: DictConfig, n_outputs: int):
    """Instantiate the configured loss with model-output-aware defaults when needed."""
    loss_cfg = OmegaConf.create(OmegaConf.to_container(cfg.train.loss, resolve=False))

    alias_target = OmegaConf.select(loss_cfg, "target")
    if alias_target is not None:
        OmegaConf.update(loss_cfg, "_target_", alias_target, force_add=True)
        if "target" in loss_cfg:
            del loss_cfg["target"]

    loss_target = str(OmegaConf.select(loss_cfg, "_target_") or "")

    loss_kwargs = {}
    if loss_target.endswith("UncertaintyWeightedMTLLoss"):
        loss_kwargs["n_traits"] = int(n_outputs)
        for incompatible_key in ("error_type", "huber_delta"):
            if incompatible_key in loss_cfg:
                del loss_cfg[incompatible_key]

    return instantiate(loss_cfg, **loss_kwargs)


def _loss_components(
    loss_fn,
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerator/denominator for losses with or without loss_components API."""
    if hasattr(loss_fn, "loss_components"):
        return loss_fn.loss_components(prediction, target, source_mask)

    try:
        scalar_loss = loss_fn(prediction, target, source_mask)
    except TypeError:
        scalar_loss = loss_fn(prediction, target)

    denominator = scalar_loss.new_tensor(1.0)
    return scalar_loss, denominator


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
    assert zarr_dir.exists(), f"Zarr directory does not exist: {zarr_dir}"

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
    target_dataset, traits, bands, target_indices, source_indices = build_target_layout(
        cfg, train_store
    )

    console.print(f"Dataset: [cyan]{target_dataset}[/cyan]")
    console.print(f"Traits ([cyan]{len(traits)}[/cyan]):")
    for trait in traits:
        console.print(f"  - {trait}")
    console.print(f"Bands ([cyan]{len(bands)}[/cyan]):")
    for band in bands:
        console.print(f"  - {band}")

    console.print("\n[bold]Data loaders:[/bold]")
    batch_size = cfg.data_loaders.batch_size
    num_workers = cfg.data_loaders.num_workers

    console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"Num workers: [cyan]{num_workers}[/cyan]")

    dataloader_cfg = {
        "zarr_dir": zarr_dir,
        "predictors": predictors,
        "target": target_dataset,
        "target_indices": target_indices,
        "source_indices": source_indices,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    train_loader = get_dataloader(split="train", **dataloader_cfg)
    val_loader = get_dataloader(split="val", **dataloader_cfg)

    X_t, y_t, src_t = next(iter(train_loader))
    X_v, y_v, src_v = next(iter(val_loader))

    assert X_t.shape[1:] == X_v.shape[1:], (
        f"X channel/spatial mismatch: train={tuple(X_t.shape[1:])} vs val={tuple(X_v.shape[1:])}"
    )
    assert y_t.shape[1:] == y_v.shape[1:], (
        f"y channel/spatial mismatch: train={tuple(y_t.shape[1:])} vs val={tuple(y_v.shape[1:])}"
    )
    assert src_t.shape[1:] == src_v.shape[1:], (
        f"source_mask channel/spatial mismatch: train={tuple(src_t.shape[1:])} vs val={tuple(src_v.shape[1:])}"
    )

    console.print(f"Predictor shape (C,H,W): [cyan]{tuple(X_t.shape[1:])}[/cyan]")
    console.print(f"Target shape (C,H,W): [cyan]{tuple(y_t.shape[1:])}[/cyan]")
    console.print(f"Source mask shape (C,H,W): [cyan]{tuple(src_t.shape[1:])}[/cyan]")

    model_cfg = resolve_model_cfg(cfg)

    console.print("\n[bold]Model and training configuration[/bold]")
    model = instantiate(
        model_cfg, in_channels=total_pred_bands, out_channels=len(target_indices)
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model:         [cyan]{model_cfg._target_}[/cyan]")
    console.print(f"  In channels:  [cyan]{total_pred_bands}[/cyan]")
    console.print(f"  Out channels: [cyan]{len(target_indices)}[/cyan]")
    console.print(f"  Parameters:   [cyan]{n_params:,}[/cyan]")

    loss_fn = _build_loss(cfg, n_outputs=len(target_indices))
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    model_params = list(model.parameters())
    loss_params = []
    if isinstance(loss_fn, torch.nn.Module):
        loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer = instantiate(cfg.train.optimizer, params=model_params + loss_params)
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    grad_clip_norm = float(cfg.train.gradient_clip_norm)
    console.print(f"Optimizer:             [cyan]{cfg.train.optimizer._target_}[/cyan]")
    loss_name = f"{loss_fn.__class__.__module__}.{loss_fn.__class__.__name__}"
    console.print(f"Loss:                  [cyan]{loss_name}[/cyan]")
    console.print(f"Scheduler:             [cyan]{cfg.train.scheduler._target_}[/cyan]")
    console.print(f"Gradient clip norm:    [cyan]{grad_clip_norm}[/cyan]")

    early_stopping_enabled = bool(cfg.train.early_stopping.enabled)
    early_stopping_patience = int(cfg.train.early_stopping.patience)
    early_stopping_min_delta = float(cfg.train.early_stopping.min_delta)
    console.print(f"Early stopping enabled: [cyan]{early_stopping_enabled}[/cyan]")
    if early_stopping_enabled:
        console.print(f"  Patience (epochs):   [cyan]{early_stopping_patience}[/cyan]")
        console.print(f"  Min delta:           [cyan]{early_stopping_min_delta}[/cyan]")

    requested_run_name = OmegaConf.select(cfg, "train.run_name")
    if requested_run_name is not None:
        requested_run_name = str(requested_run_name)
    requested_run_group = OmegaConf.select(cfg, "train.group")
    if requested_run_group is not None:
        requested_run_group = str(requested_run_group)

    wandb_module = None

    if cfg.wandb.enabled:
        try:
            import wandb as wandb_module
        except Exception as exc:
            raise RuntimeError(
                "W&B is enabled but import failed. "
                "Set wandb.enabled=false for debugging or fix the environment. "
                f"Original error: {exc}"
            ) from exc

        run = wandb_module.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=requested_run_name,
            group=requested_run_group,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        run_name = run.name
        console.print(f"W&B logging enabled. Run: [cyan]{run_name}[/cyan]")
    else:
        run_name = requested_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        console.print("[yellow]W&B logging disabled.[/yellow]")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_checkpoint_path = checkpoint_dir / f"{run_name}.pth"
    config_path = checkpoint_dir / f"{run_name}.yaml"
    OmegaConf.save(cfg, config_path)

    for epoch in range(cfg.train.epochs):
        console.rule(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        model.train()
        train_num_total = 0.0
        train_den_total = 0.0

        for X, y, src in track(train_loader, description="Training"):
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            src = src.to(device, dtype=torch.float32)

            y_pred, valid = predict_batch(model, X)
            y = torch.where(valid.expand_as(y), y, torch.nan)
            y = torch.where(src > 0, y, torch.nan)

            if not torch.isfinite(y).any():
                continue

            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(y_pred).all():
                continue
            batch_num, batch_den = _loss_components(loss_fn, y_pred, y, src)
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

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss_avg = (
            train_num_total / train_den_total if train_den_total > 0.0 else float("nan")
        )
        console.print(f"  train_loss={train_loss_avg:.6f}")

        model.eval()
        val_num_total = 0.0
        val_den_total = 0.0
        val_y_true_parts = []
        val_y_pred_parts = []
        val_src_parts = []

        with torch.no_grad():
            for X, y, src in track(val_loader, description="Validation"):
                X = X.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                src = src.to(device, dtype=torch.float32)

                y_pred, valid = predict_batch(model, X)
                y = torch.where(valid.expand_as(y), y, torch.nan)
                y = torch.where(src > 0, y, torch.nan)

                if not torch.isfinite(y).any():
                    continue

                if not torch.isfinite(y_pred).all():
                    continue

                val_y_true_parts.append(y.detach().cpu())
                val_y_pred_parts.append(y_pred.detach().cpu())
                val_src_parts.append(src.detach().cpu())

                batch_num, batch_den = _loss_components(loss_fn, y_pred, y, src)
                if (
                    not torch.isfinite(batch_num)
                    or not torch.isfinite(batch_den)
                    or batch_den <= 0.0
                ):
                    continue
                val_num_total += batch_num.item()
                val_den_total += batch_den.item()

        val_loss_avg = (
            val_num_total / val_den_total if val_den_total > 0.0 else float("nan")
        )
        console.print(f"  val_loss={val_loss_avg:.6f}")

        val_metric_summary = None
        if val_y_true_parts:
            val_metric_summary = summarize_single_trait_metrics(
                y_true=torch.cat(val_y_true_parts, dim=0),
                y_pred=torch.cat(val_y_pred_parts, dim=0),
                source_mask=torch.cat(val_src_parts, dim=0),
                trait_names=traits,
                n_bands=len(bands),
            )
            console.print(
                "  val_rmse={rmse:.6f}  val_r2={r2:.6f}  val_pearson_r={pearson_r:.6f}".format(
                    rmse=val_metric_summary["rmse"],
                    r2=val_metric_summary["r2"],
                    pearson_r=val_metric_summary["pearson_r"],
                )
            )

        val_loss_valid = math.isfinite(val_loss_avg)
        if not val_loss_valid:
            console.print(
                "[yellow]val_loss is NaN — skipping checkpoint and early-stopping update.[/yellow]"
            )

        is_best = (
            val_loss_valid and val_loss_avg < best_val_loss - early_stopping_min_delta
        )
        if is_best:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "in_channels": total_pred_bands,
                    "out_channels": len(target_indices),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "wandb_run_id": getattr(run, "id", None)
                    if cfg.wandb.enabled
                    else None,
                },
                best_checkpoint_path,
            )
            console.print(
                f"[green]New best model saved[/green] (val_loss={best_val_loss:.6f})"
            )
        elif val_loss_valid:
            epochs_without_improvement += 1

        if cfg.wandb.enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss_avg,
                "val/loss": val_loss_avg,
                "train/lr": current_lr,
            }
            if val_metric_summary is not None:
                log_dict["val/rmse"] = val_metric_summary["rmse"]
                log_dict["val/r2"] = val_metric_summary["r2"]
                log_dict["val/pearson_r"] = val_metric_summary["pearson_r"]
            if is_best:
                log_dict["val/loss_best"] = best_val_loss
            if wandb_module is not None:
                wandb_module.log(log_dict)

        if (
            early_stopping_enabled
            and epochs_without_improvement >= early_stopping_patience
        ):
            console.print(
                f"[yellow]Early stopping triggered[/yellow] (patience={early_stopping_patience}, best_epoch={best_epoch})"
            )
            break

    console.rule("[bold cyan]DONE[/bold cyan]")
    console.print(f"Best val_loss={best_val_loss:.6f} at epoch {best_epoch}")
    console.print(f"Checkpoint: [cyan]{best_checkpoint_path}[/cyan]")

    if cfg.wandb.enabled:
        final_status = (
            "ok" if math.isfinite(best_val_loss) and best_epoch > 0 else "failed"
        )
        run.summary["wandb_run_id"] = getattr(run, "id", None)
        run.summary["best_val_loss"] = (
            float(best_val_loss) if math.isfinite(best_val_loss) else None
        )
        run.summary["best_epoch"] = int(best_epoch)
        run.summary["checkpoint_path"] = str(best_checkpoint_path)
        run.summary["status"] = final_status

    if cfg.wandb.enabled:
        run.finish()


if __name__ == "__main__":
    main()
