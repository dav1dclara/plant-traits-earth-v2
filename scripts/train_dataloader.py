import os
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader
from ptev2.utils import seed_all


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")

        # GTX 1080 Ti is sm_61; modern torch builds may require >= sm_70.
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:
            print(
                "CUDA device capability "
                f"sm_{major}{minor} is too old for this PyTorch build. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

    return torch.device(requested_device)


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _masked_mse(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> tuple[torch.Tensor | None, int]:
    valid = torch.isfinite(y_pred) & torch.isfinite(y_true)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return None, 0
    diff = y_pred[valid] - y_true[valid]
    return (diff * diff).mean(), n_valid


def _validate_batch_shapes(
    X: torch.Tensor,
    y: torch.Tensor,
    expected_in_channels: int,
    expected_n_traits: int,
) -> torch.Tensor:
    if X.ndim != 4:
        raise ValueError(f"Expected X to have shape (B,C,H,W), got {tuple(X.shape)}")

    if X.shape[1] != expected_in_channels:
        raise ValueError(
            "Input channel mismatch: "
            f"expected {expected_in_channels}, got {X.shape[1]}."
        )

    if y.ndim == 1:
        y = y.unsqueeze(1)
    elif y.ndim != 2:
        raise ValueError(
            "Target shape mismatch: expected rank-1 or rank-2 target tensor "
            f"(for n_traits={expected_n_traits}), got {tuple(y.shape)}."
        )

    available_n_traits = y.shape[1]
    if available_n_traits < expected_n_traits:
        raise ValueError(
            "Configured data.n_traits exceeds loaded target dimension: "
            f"data.n_traits={expected_n_traits}, loaded_target_dim={available_n_traits}."
        )

    # Single-trait (or subset) training from a multi-trait target array.
    return y[:, :expected_n_traits]


def train(cfg: DictConfig) -> None:
    # Set random seed
    seed_all(cfg.training.seed)

    # Set device
    device = _resolve_device(cfg.training.device)
    print(f"Using device: {device}\n")

    print(
        "W&B config: "
        f"enabled={cfg.wandb.enabled}, "
        f"entity={cfg.wandb.entity}, "
        f"project={cfg.wandb.project}, "
        f"run_name={cfg.wandb.run_name}"
    )
    for env_name in ["WANDB_MODE", "WANDB_DISABLED", "WANDB_SILENT"]:
        env_val = os.environ.get(env_name)
        if env_val is not None:
            print(f"{env_name}={env_val}")

    run = None
    if cfg.wandb.enabled:
        try:
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit="finish_previous",
            )
            print("W&B logging enabled.")
            print(
                "W&B run info: "
                f"id={run.id}, mode={run.settings.mode}, "
                f"project={run.project}, entity={run.entity}"
            )
            if run.url:
                print(f"W&B URL: {run.url}")
            else:
                print("W&B URL unavailable (likely offline/disabled mode).")
        except Exception as exc:
            run = None
            print(f"W&B init failed ({exc}). Continuing without W&B logging.")
    else:
        print("W&B logging disabled.")

    # Get data configuration for training
    print("--- DATA PROPERTIES ---")
    training_data_cfg = cfg.training.data
    target = training_data_cfg.target.source
    predictors = training_data_cfg.predictors
    used_predictors = [
        name for name, predictor_cfg in predictors.items() if predictor_cfg.use
    ]
    if not used_predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    print(f"Zarr directory:      {training_data_cfg.zarr_dir}")
    print(f"Target used:         {target}")
    print("Predictors used:")
    for name in used_predictors:
        print(f"  - {name}")
    print(f"Configured n_traits: {cfg.data.n_traits}")
    print(f"Configured channels: {cfg.data.in_channels}")

    print()

    # Get data loaders
    print("--- DATA LOADERS ---")
    zarr_dir = Path(training_data_cfg.zarr_dir)
    data_loader_cfg = cfg.training.data_loaders
    batch_size = data_loader_cfg.batch_size
    num_workers = data_loader_cfg.num_workers

    print(f"Batch size:          {batch_size}")
    print(f"Number of workers:   {num_workers}")

    train_loader = get_dataloader(
        zarr_dir,
        split=training_data_cfg.train_split,
        predictors=used_predictors,
        target=target,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = get_dataloader(
        zarr_dir,
        split=training_data_cfg.val_split,
        predictors=used_predictors,
        target=target,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Get model / optimization components from config
    model = instantiate(cfg.models.active).to(device)
    loss_fn = instantiate(cfg.training.loss)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer)

    total_params, trainable_params = _count_parameters(model)
    print("\n--- MODEL ---")
    print(f"Model:               {model.__class__.__name__}")
    print(
        f"Parameters:          {trainable_params:,} trainable / {total_params:,} total"
    )

    # Training loop
    print("\n--- TRAINING ---")
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_skipped = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [train]"
        )
        for batch_idx, (X, y) in enumerate(train_bar):
            y = _validate_batch_shapes(
                X=X,
                y=y,
                expected_in_channels=int(cfg.data.in_channels),
                expected_n_traits=int(cfg.data.n_traits),
            )

            X_raw = X
            y_raw = y
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            y_pred = model(X)
            loss, n_valid = _masked_mse(y_pred, y)
            if loss is None:
                train_skipped += 1
                continue

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu().item())
            train_batches += 1

            if epoch == 0 and batch_idx == 0:
                print(f"First batch X shape: {tuple(X.shape)}")
                print(f"First batch y shape: {tuple(y.shape)}")
                print(f"First batch y_pred shape: {tuple(y_pred.shape)}")
                print(
                    "First batch NaN stats: "
                    f"X={int(torch.isnan(X_raw).sum().item())}, "
                    f"y={int(torch.isnan(y_raw).sum().item())}, "
                    f"valid_targets={n_valid}"
                )

            train_bar.set_postfix(loss=f"{loss.item():.5f}")

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_skipped = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [val]"
            )
            for X, y in val_bar:
                y = _validate_batch_shapes(
                    X=X,
                    y=y,
                    expected_in_channels=int(cfg.data.in_channels),
                    expected_n_traits=int(cfg.data.n_traits),
                )

                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                y_pred = model(X)
                loss, _ = _masked_mse(y_pred, y)
                if loss is None:
                    val_skipped += 1
                    continue

                val_loss_sum += float(loss.detach().cpu().item())
                val_batches += 1

                val_bar.set_postfix(loss=f"{loss.item():.5f}")

        avg_train_loss = (
            train_loss_sum / train_batches if train_batches > 0 else float("nan")
        )
        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else float("nan")
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} "
            f"train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f} "
            f"(skipped train={train_skipped}, val={val_skipped})"
        )
        if train_batches == 0:
            print(
                "Warning: no valid train targets in this epoch after masking "
                "(all batches skipped)."
            )
        if val_batches == 0:
            print(
                "Warning: no valid val targets in this epoch after masking "
                "(all batches skipped)."
            )
        if run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_skipped_batches": train_skipped,
                    "val_skipped_batches": val_skipped,
                }
            )
        # TODO: save model to use on test set after training loop finishes

    if run is not None:
        run.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
