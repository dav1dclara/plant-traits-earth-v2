"""
Training script for MTL models (STL, MTL, or MMoE).
Configuration at top - just edit config section and run.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dataloader import get_mtl_dataloader
from loss_v2 import UncertaintyWeightedMTLLoss, per_trait_masked_loss
from metrics import compute_metrics
from models_v2 import MMoEModel, MTLModel, STLModel
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# ─ CONFIGURATION — EDIT THIS SECTION ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Data
ZARR_DIR = Path("/scratch3/plant-traits-v2/data/22km/chips/patch15_stride10")
PREDICTORS = ["modis", "worldclim", "soil_grids", "vodca", "canopy_height"]
TARGETS = ["comb"]  # target array name in zarr
BATCH_SIZE = 32
NUM_WORKERS = 4

# Model architecture
MODEL_TYPE = "mtl"  # "stl", "mtl", or "mmoe"
N_TRAITS = 37
IN_CHANNELS = 150
BASE_CHANNELS = 48
NORM = "gn"
DROPOUT_P = 0.1
STRIDE_BLOCKS = (1, 1, 1, 1)  # no downsampling for spatial predictions

# MMoE-specific
N_EXPERTS = 6
EXPERT_HIDDEN = 192

# Training
EPOCHS = 100
LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 5e-5
GRAD_CLIP = 5.0

# Loss weighting (source-mask based)
W_GBIF = 1.0
W_SPLOT = 16.0

# Early stopping
EARLY_STOP_PATIENCE = 15
BEST_BY = "mean_r"  # Choose best checkpoint by 'val_loss' or 'mean_r'
VAL_METRIC_SOURCE = "splot"  # 'splot', 'gbif', or 'all'

# Device & seed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Output
SCRIPT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_ROOT / "results" / MODEL_TYPE / "checkpoints"
METRICS_DIR = SCRIPT_ROOT / "results" / MODEL_TYPE / "metrics"

# ═══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int = 42) -> None:
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_target_and_source(
    y_full: torch.Tensor,  # (B, 259, H, W) = (B, 37 traits * 7 stats, H, W)
    n_traits: int = 37,
    n_stats: int = 7,
    mean_idx: int = 0,
    source_idx: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract trait means and source masks from concatenated zarr target.

    Zarr stores: (37 traits * 7 statistics) = 259 channels
    Stats order: [mean, std, median, q05, q95, count, source]

    Returns:
        y_target  : (B, 37, H, W)  — mean values per trait
        src_mask  : (B, 37, H, W)  — source code (0=missing, 1=GBIF, 2=sPlot)
    """
    mean_indices = [t * n_stats + mean_idx for t in range(n_traits)]
    source_indices = [t * n_stats + source_idx for t in range(n_traits)]

    y_target = y_full[:, mean_indices, :, :]  # (B, 37, H, W)
    src_raw = y_full[:, source_indices, :, :]  # (B, 37, H, W)

    # Clean source mask: convert to 0=missing, 1=GBIF, 2=sPlot
    src_clean = torch.nan_to_num(src_raw, nan=0.0, posinf=0.0, neginf=0.0)
    src_mask = torch.zeros_like(src_clean, dtype=torch.int64)
    src_mask = torch.where((src_clean >= 1.5) & (src_clean < 2.5), 2, src_mask)
    src_mask = torch.where((src_clean >= 0.5) & (src_clean < 1.5), 1, src_mask)

    return y_target, src_mask


def _is_better(current_score: float, best_score: float, best_by: str) -> bool:
    if best_by == "val_loss":
        return current_score < best_score
    if best_by == "mean_r":
        return current_score > best_score
    raise ValueError(f"Unknown BEST_BY value: {best_by}")


def _initial_best_score(best_by: str) -> float:
    return float("inf") if best_by == "val_loss" else float("-inf")


def build_model(
    model_type: str,
    in_channels: int,
    n_traits: int,
    base_channels: int,
    norm: str,
    dropout_p: float,
    stride_blocks: tuple,
    n_experts: int = 4,
    expert_hidden: int = 64,
) -> nn.Module:
    """Instantiate model based on type."""
    if model_type == "stl":
        return STLModel(
            in_channels=in_channels,
            n_traits=n_traits,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout_p,
            stride_blocks=stride_blocks,
        )
    elif model_type == "mtl":
        return MTLModel(
            in_channels=in_channels,
            n_traits=n_traits,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout_p,
            stride_blocks=stride_blocks,
        )
    elif model_type == "mmoe":
        return MMoEModel(
            in_channels=in_channels,
            n_traits=n_traits,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout_p,
            stride_blocks=stride_blocks,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: UncertaintyWeightedMTLLoss,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    """Training loop for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y_full in tqdm(train_loader, desc="Train"):
        X = X.to(device)
        y_full = y_full.to(device)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract trait means and source mask
        y_target, src_mask = _split_target_and_source(y_full)

        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X)  # (B, 37, H, W)

        # Compute loss
        loss = loss_fn(y_pred, y_target, src_mask)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def eval_epoch(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> tuple[float, dict]:
    """Evaluation loop, returns loss and validation metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_trait_losses = []
    all_preds = []
    all_targets = []
    all_source_masks = []

    with torch.no_grad():
        for X, y_full in tqdm(val_loader, desc="Val"):
            X = X.to(device)
            y_full = y_full.to(device)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract trait means and source mask
            y_target, src_mask = _split_target_and_source(y_full)

            # Forward pass
            y_pred = model(X)  # (B, 37, H, W)

            # Per-trait losses (for logging)
            trait_losses = per_trait_masked_loss(
                y_pred, y_target, src_mask, W_GBIF, W_SPLOT, reduction="mean"
            )
            all_trait_losses.append(trait_losses.cpu().numpy())

            # Scalar loss
            loss = trait_losses[torch.isfinite(trait_losses)].mean()
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
            all_source_masks.append(src_mask.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    per_trait = np.nanmean(np.stack(all_trait_losses), axis=0)  # (37,)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    source_masks = np.concatenate(all_source_masks, axis=0)
    val_metrics = compute_metrics(
        preds,
        targets,
        source_mask=source_masks,
        eval_source=VAL_METRIC_SOURCE,
    )
    val_metrics["per_trait_loss"] = per_trait.tolist()
    val_metrics["n_traits_r_above_0_5"] = int(
        np.sum(np.array(val_metrics["per_trait_r"]) > 0.5)
    )

    return avg_loss, val_metrics


def main() -> None:
    """Main training loop."""
    set_seed(SEED)
    device = torch.device(DEVICE)

    # Create directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Minimum learning rate: {MIN_LR}")
    print(f"Validation metric source: {VAL_METRIC_SOURCE}")

    # Dataloaders
    print("Loading dataloaders...")
    train_loader = get_mtl_dataloader(
        ZARR_DIR, "train", PREDICTORS, TARGETS, BATCH_SIZE, NUM_WORKERS
    )
    val_loader = get_mtl_dataloader(
        ZARR_DIR, "val", PREDICTORS, TARGETS, BATCH_SIZE, NUM_WORKERS
    )

    # Model
    print(f"Building {MODEL_TYPE} model...")
    model = build_model(
        MODEL_TYPE,
        IN_CHANNELS,
        N_TRAITS,
        BASE_CHANNELS,
        NORM,
        DROPOUT_P,
        STRIDE_BLOCKS,
        N_EXPERTS,
        EXPERT_HIDDEN,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR,
    )

    # Loss function
    loss_fn = UncertaintyWeightedMTLLoss(w_gbif=W_GBIF, w_splot=W_SPLOT)
    loss_fn = loss_fn.to(device)

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_mean_r = float("-inf")
    max_val_mean_r = float("-inf")
    max_val_mean_r_epoch = 0
    best_score = _initial_best_score(BEST_BY)
    patience_counter = 0
    best_checkpoint = None

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, GRAD_CLIP
        )
        train_losses.append(train_loss)

        # Eval
        val_loss, val_metrics = eval_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        val_mean_r = float(val_metrics["pearson_r_mean"])
        n_traits_above_0_5 = int(val_metrics["n_traits_r_above_0_5"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        if val_mean_r > max_val_mean_r:
            max_val_mean_r = val_mean_r
            max_val_mean_r_epoch = epoch

        elapsed = time.time() - t_start

        # Log
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_mean_r={val_mean_r:.4f} | r>0.5={n_traits_above_0_5} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

        # Save the latest checkpoint
        last_ckpt_path = CHECKPOINT_DIR / f"{MODEL_TYPE}_last.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_r": val_mean_r,
                "lr": current_lr,
                "n_traits_r_above_0_5": n_traits_above_0_5,
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
            },
            last_ckpt_path,
        )

        # Best checkpoint selection
        current_score = val_loss if BEST_BY == "val_loss" else val_mean_r
        if _is_better(current_score, best_score, BEST_BY):
            best_score = current_score
            best_val_loss = val_loss
            best_mean_r = val_mean_r
            patience_counter = 0
            best_ckpt_path = CHECKPOINT_DIR / f"{MODEL_TYPE}_best.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "loss_state_dict": loss_fn.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mean_r": val_mean_r,
                    "lr": current_lr,
                    "n_traits_r_above_0_5": n_traits_above_0_5,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
            best_checkpoint = best_ckpt_path
            print(f"  ✓ Best checkpoint updated by {BEST_BY}: {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    # Summary
    print(f"\nTraining complete!")
    print(f"Best selection metric ({BEST_BY}): {best_score:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Validation mean_r at selected checkpoint: {best_mean_r:.6f}")
    print(
        f"Max validation mean_r observed: {max_val_mean_r:.6f} (epoch {max_val_mean_r_epoch})"
    )
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Results saved to: {CHECKPOINT_DIR.parent}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "best_mean_r": float(best_mean_r),
        "max_val_mean_r": float(max_val_mean_r),
        "max_val_mean_r_epoch": int(max_val_mean_r_epoch),
        "best_selection_metric": float(best_score),
        "best_by": BEST_BY,
        "initial_lr": float(LR),
        "min_lr": float(MIN_LR),
        "epochs_trained": epoch,
    }
    import json

    with open(METRICS_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {METRICS_DIR / 'train_history.json'}")


if __name__ == "__main__":
    main()
