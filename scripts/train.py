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
from ptev2.utils import seed_all

console = Console()


@hydra.main(config_path="../config", config_name="training/default", version_base=None)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TRAINING[/bold cyan]")

    # Set random seed
    seed_all(cfg.train.seed)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    console.print(f"Device: [cyan]{device}[/cyan]")

    # Data configuration
    console.print("[bold]\nData[/bold]")
    root_dir = Path(cfg.data.root_dir)
    resolution_km = cfg.data.resolution_km
    patch_size = cfg.data.patch_size
    stride = cfg.data.stride
    zarr_dir = (
        root_dir / f"{resolution_km}km" / "chips" / f"patch{patch_size}_stride{stride}"
    )
    assert zarr_dir.exists(), f"Zarr directory does not exist: {zarr_dir}"

    console.print(f"Resolution (km): [cyan]{resolution_km}[/cyan]")
    console.print(f"Patch size: [cyan]{patch_size}[/cyan]")
    console.print(f"Stride: [cyan]{stride}[/cyan]")
    console.print(f"Zarr directory: [cyan]{zarr_dir}[/cyan]")

    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")

    # Predictors
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

    # Targets
    console.print("\n[bold]Targets:[/bold]")

    # What's available in the zarr store
    zarr_dataset_names = list(train_store["targets"].keys())
    zarr_band_names = train_store["targets"].attrs["band_names"]
    band_to_idx = {name: idx for idx, name in enumerate(zarr_band_names)}

    # What's requested in the config
    target_cfg = cfg.data.targets
    target_dataset = str(target_cfg.dataset)
    cfg_bands = [str(v) for v in target_cfg.bands]

    # Validate config against zarr store
    if target_dataset not in zarr_dataset_names:
        raise ValueError(
            f"Dataset '{target_dataset}' not found in zarr. Available: {', '.join(zarr_dataset_names)}"
        )
    zarr_all_traits = [
        f.replace("X", "").replace(".tif", "")
        for f in train_store[f"targets/{target_dataset}"].attrs["files"]
    ]
    traits = (
        [str(v) for v in target_cfg.traits] if target_cfg.traits else zarr_all_traits
    )
    for trait in traits:
        if trait not in zarr_all_traits:
            raise ValueError(
                f"Trait '{trait}' not found in dataset '{target_dataset}' in zarr."
            )
    for band in cfg_bands:
        if band not in band_to_idx:
            raise ValueError(
                f"Band '{band}' not found in zarr. Available: {', '.join(zarr_band_names)}"
            )
    bands = cfg_bands

    # Print what we're using
    console.print(f"Dataset: [cyan]{target_dataset}[/cyan]")
    console.print(f"Traits ([cyan]{len(traits)}[/cyan]):")
    for trait in traits:
        console.print(f"  - {trait}")
    console.print(f"Bands ([cyan]{len(bands)}[/cyan]):")
    for band in bands:
        console.print(f"  - {band}")

    n_bands = len(zarr_band_names)
    target_indices = [
        trait_pos * n_bands + band_to_idx[band]
        for trait_pos in range(len(traits))
        for band in bands
    ]
    source_indices = [
        trait_pos * n_bands + band_to_idx["source"] for trait_pos in range(len(traits))
    ]

    # Dataloader configuration
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

    # Model  # TODO ask Luca what happens here
    console.print("\n[bold]Model and training configuration[/bold]")
    model = instantiate(
        cfg.models, in_channels=total_pred_bands, out_channels=len(target_indices)
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model:         [cyan]{cfg.models._target_}[/cyan]")
    console.print(f"  In channels:  [cyan]{total_pred_bands}[/cyan]")
    console.print(f"  Out channels: [cyan]{len(target_indices)}[/cyan]")
    console.print(f"  Parameters:   [cyan]{n_params:,}[/cyan]")

    # Optimizer, loss, scheduler
    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())
    loss_fn = instantiate(cfg.train.loss)
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    # scheduler_metric_name = str(cfg.train.scheduler_step.metric)  # TODO: what does this do?
    grad_clip_norm = float(cfg.train.gradient_clip_norm)
    console.print(f"Optimizer:             [cyan]{cfg.train.optimizer._target_}[/cyan]")
    console.print(f"Loss:                  [cyan]{cfg.train.loss._target_}[/cyan]")
    console.print(f"Scheduler:             [cyan]{cfg.train.scheduler._target_}[/cyan]")
    # console.print(f"Scheduler step metric: [cyan]{scheduler_metric_name}[/cyan]")
    console.print(f"Gradient clip norm:    [cyan]{grad_clip_norm}[/cyan]")

    # Early stopping configuration
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

    # W&B
    if cfg.wandb.enabled:
        try:
            import wandb as wandb_module  # Lazy import for debugger stability.
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

    # Checkpoint and early stopping state
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

        # Training loop
        model.train()
        train_num_total = 0.0
        train_den_total = 0.0
        # train_total_pixels = 0
        # train_valid_pixels = 0

        for X, y, src in track(train_loader, description="Training"):
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            src = src.to(device, dtype=torch.float32)

            # Mask y where predictors are invalid or source is unlabeled
            valid = torch.isfinite(X).all(dim=1, keepdim=True)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.clamp(X, min=-1e4, max=1e4)
            y = torch.where(valid.expand_as(y), y, torch.nan)
            y = torch.where(src > 0, y, torch.nan)

            # train_total_pixels += y.numel()
            # train_valid_pixels += int(torch.isfinite(y).sum().item())

            # Skip batch if no valid observations
            if not torch.isfinite(y).any():
                continue

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(X)
            if not torch.isfinite(y_pred).all():
                continue
            batch_num, batch_den = loss_fn.loss_components(y_pred, y, src)
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
        # train_valid_pct = 100.0 * train_valid_pixels / train_total_pixels if train_total_pixels > 0 else float("nan")
        console.print(f"  train_loss={train_loss_avg:.6f}")

        # Validation loop
        model.eval()
        val_num_total = 0.0
        val_den_total = 0.0
        # val_total_pixels = 0
        # val_valid_pixels = 0

        with torch.no_grad():
            for X, y, src in track(val_loader, description="Validation"):
                X = X.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                src = src.to(device, dtype=torch.float32)

                valid = torch.isfinite(X).all(dim=1, keepdim=True)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = torch.clamp(X, min=-1e4, max=1e4)
                y = torch.where(valid.expand_as(y), y, torch.nan)
                y = torch.where(src > 0, y, torch.nan)

                # val_total_pixels += y.numel()
                # val_valid_pixels += int(torch.isfinite(y).sum().item())

                if not torch.isfinite(y).any():
                    continue

                y_pred = model(X)
                if not torch.isfinite(y_pred).all():
                    continue
                batch_num, batch_den = loss_fn.loss_components(y_pred, y, src)
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
        # val_valid_pct = 100.0 * val_valid_pixels / val_total_pixels if val_total_pixels > 0 else float("nan")
        console.print(f"  val_loss={val_loss_avg:.6f}")

        # Checkpoint: save best model
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
                },
                best_checkpoint_path,
            )
            console.print(
                f"[green]New best model saved[/green] (val_loss={best_val_loss:.6f})"
            )
        elif val_loss_valid:
            epochs_without_improvement += 1

        # W&B logging
        if cfg.wandb.enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss_avg,
                "val/loss": val_loss_avg,
                "train/lr": current_lr,
            }
            if is_best:
                log_dict["val/loss_best"] = best_val_loss
            if wandb_module is not None:
                wandb_module.log(log_dict)

        # Early stopping
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
        run.summary["best_val_loss"] = (
            float(best_val_loss) if math.isfinite(best_val_loss) else None
        )
        run.summary["best_epoch"] = int(best_epoch)
        run.summary["checkpoint_path"] = str(best_checkpoint_path)
        run.summary["status"] = final_status

    # Close wandb run
    if cfg.wandb.enabled:
        run.finish()


if __name__ == "__main__":
    main()
