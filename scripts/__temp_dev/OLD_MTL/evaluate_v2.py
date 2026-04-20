"""
Evaluation script for MTL models.
Configuration at top - match train_v2.py settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from dataloader import get_mtl_dataloader
from gate_analysis import resolve_trait_ids, summarize_gate_weights
from metrics import compute_metrics
from models_v2 import GatedMMoEModel, MMoEModel, MTLModel, STLModel
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# ─ CONFIGURATION — MATCH YOUR TRAINING CONFIG ────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Data
ZARR_DIR = Path("/scratch3/plant-traits-v2/data/22km/chips/patch15_stride10")
PREDICTORS = ["modis", "worldclim", "soil_grids", "vodca", "canopy_height"]
TARGETS = ["comb"]
BATCH_SIZE = 32
NUM_WORKERS = 4

# Model architecture - MUST MATCH TRAINING CONFIG
MODEL_TYPE = "mmoe_gated"  # "stl", "mtl", "mmoe", or "mmoe_gated"
N_TRAITS = 37
IN_CHANNELS = 150
BASE_CHANNELS = 48
NORM = "gn"
DROPOUT_P = 0.1
STRIDE_BLOCKS = (1, 1, 1, 1)

# MMoE-specific (if using mmoe)
N_EXPERTS = 6
EXPERT_HIDDEN = 192
GATE_TEMPERATURE = 0.5
GATE_TOP_K = 2

# Checkpoint to evaluate
SCRIPT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    SCRIPT_ROOT / "results" / MODEL_TYPE / "checkpoints" / f"{MODEL_TYPE}_best.pth"
)

# Results output
RESULTS_DIR = SCRIPT_ROOT / "results" / MODEL_TYPE / "eval_results"

# Metric source selection
EVAL_SOURCE = "splot"  # 'splot', 'gbif', or 'all'

# Trait metadata
TRAIT_DIR = ZARR_DIR.parents[1] / "targets" / "comb"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════


def _split_target_and_source(
    y_full: torch.Tensor,
    n_traits: int = 37,
    n_stats: int = 7,
    mean_idx: int = 0,
    source_idx: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract trait means and source masks."""
    mean_indices = [t * n_stats + mean_idx for t in range(n_traits)]
    source_indices = [t * n_stats + source_idx for t in range(n_traits)]

    y_target = y_full[:, mean_indices, :, :]
    src_raw = y_full[:, source_indices, :, :]

    src_clean = torch.nan_to_num(src_raw, nan=0.0, posinf=0.0, neginf=0.0)
    src_mask = torch.zeros_like(src_clean, dtype=torch.int64)
    src_mask = torch.where((src_clean >= 1.5) & (src_clean < 2.5), 2, src_mask)
    src_mask = torch.where((src_clean >= 0.5) & (src_clean < 1.5), 1, src_mask)

    return y_target, src_mask


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
    gate_temperature: float = 1.0,
    gate_top_k: int | None = None,
):
    """Build model matching train config."""
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
    elif model_type == "mmoe_gated":
        return GatedMMoEModel(
            in_channels=in_channels,
            n_traits=n_traits,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout_p,
            stride_blocks=stride_blocks,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            gate_temperature=gate_temperature,
            gate_top_k=gate_top_k,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Evaluate model on test set.
    Returns: predictions, targets, and source masks, all (N_samples, N_traits, H, W)
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_source_masks = []
    all_gate_weights = []

    with torch.no_grad():
        for X, y_full in tqdm(test_loader, desc="Evaluating"):
            X = X.to(device)
            y_full = y_full.to(device)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract means
            y_target, src_mask = _split_target_and_source(y_full)

            # Predict
            y_pred = model(X)  # (B, 37, H, W)

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
            all_source_masks.append(src_mask.cpu().numpy())
            gate_weights = getattr(model, "last_gate_weights", None)
            if gate_weights is not None:
                all_gate_weights.append(gate_weights.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N, 37, H, W)
    targets = np.concatenate(all_targets, axis=0)  # (N, 37, H, W)
    source_masks = np.concatenate(all_source_masks, axis=0)  # (N, 37, H, W)
    gate_weights = None
    if all_gate_weights:
        gate_weights = np.concatenate(all_gate_weights, axis=0)

    return preds, targets, source_masks, gate_weights


def main() -> None:
    """Main evaluation."""
    device = torch.device(DEVICE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Evaluation source: {EVAL_SOURCE}")
    if MODEL_TYPE == "mmoe_gated":
        print(f"Gate temperature: {GATE_TEMPERATURE}")
        print(f"Gate top-k: {GATE_TOP_K}")

    # Load data
    print("Loading test set...")
    test_loader = get_mtl_dataloader(
        ZARR_DIR, "test", PREDICTORS, TARGETS, BATCH_SIZE, NUM_WORKERS
    )

    # Build model
    print("Building model...")
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
        GATE_TEMPERATURE,
        GATE_TOP_K,
    )
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"Make sure to run train_v2.py first")
        return

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate
    print("Evaluating...")
    preds, targets, source_masks, gate_weights = evaluate(model, test_loader, device)

    # Save predictions
    print(f"Saving results to {RESULTS_DIR}")
    np.save(RESULTS_DIR / "test_preds.npy", preds)
    np.save(RESULTS_DIR / "test_targets.npy", targets)
    np.save(RESULTS_DIR / "test_source_masks.npy", source_masks)
    if gate_weights is not None:
        np.save(RESULTS_DIR / "test_gate_weights.npy", gate_weights)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(
        preds,
        targets,
        source_mask=source_masks,
        eval_source=EVAL_SOURCE,
    )
    trait_ids = resolve_trait_ids(TRAIT_DIR, N_TRAITS)
    metrics["trait_ids"] = trait_ids
    metrics["per_trait_summary"] = [
        {
            "trait_id": trait_id,
            "pearson_r": float(metrics["per_trait_r"][idx]),
            "r2": float(metrics["per_trait_r2"][idx]),
            "rmse": float(metrics["per_trait_rmse"][idx]),
        }
        for idx, trait_id in enumerate(trait_ids)
    ]

    gate_summary = None
    if gate_weights is not None:
        gate_summary = summarize_gate_weights(gate_weights, trait_ids)
        metrics["gate_mean_entropy"] = float(gate_summary["mean_entropy"])
        metrics["gate_expert_usage_mean"] = gate_summary["expert_usage_mean"]
        with open(RESULTS_DIR / "gate_summary.json", "w") as f:
            json.dump(
                {
                    key: value
                    for key, value in gate_summary.items()
                    if not key.startswith("_")
                },
                f,
                indent=2,
            )
        np.save(
            RESULTS_DIR / "gate_similarity_matrix.npy",
            gate_summary["_gate_similarity_array"],
        )
        np.save(
            RESULTS_DIR / "mean_gate_weights.npy",
            gate_summary["_mean_gate_weights_array"],
        )

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {MODEL_TYPE}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Metric source: {metrics['eval_source']}")
    print(f"Evaluated pixels: {metrics['n_eval_pixels']}")
    if metrics["eval_source"] == "splot":
        print(f"Valid sPlot pixels: {metrics['n_valid_pixels']}")
    print(f"\nPearson r (all): {metrics['pearson_r_all']:.4f}")
    print(f"Pearson r (mean across traits): {metrics['pearson_r_mean']:.4f}")
    print(f"R^2 (all): {metrics['r2_all']:.4f}")
    print(f"R^2 (mean across traits): {metrics['r2_mean']:.4f}")
    print(f"RMSE (all pixels): {metrics['rmse_all']:.4f}")
    print(f"RMSE (per-trait mean): {metrics['rmse_mean']:.4f}")
    if gate_summary is not None:
        print(f"Gate mean entropy: {metrics['gate_mean_entropy']:.4f}")
        print(
            "Expert usage mean: "
            + ", ".join(
                f"e{expert_idx}={usage:.3f}"
                for expert_idx, usage in enumerate(metrics["gate_expert_usage_mean"])
            )
        )
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
