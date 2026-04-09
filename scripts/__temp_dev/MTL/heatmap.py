"""
Heatmap and task-similarity utilities for MTL.

Supports:
- prediction vs truth patch heatmaps from saved eval arrays
- gradient similarity heatmap across trait heads using the shared encoder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_mtl_dataloader
from models_v2 import MMoEModel, MTLModel, STLModel

# Data
ZARR_DIR = Path("/scratch3/plant-traits-v2/data/22km/chips/patch15_stride10")
PREDICTORS = ["modis", "worldclim", "soil_grids", "vodca", "canopy_height"]
TARGETS = ["comb"]
BATCH_SIZE = 32
NUM_WORKERS = 4

# Model configuration
MODEL_TYPE = "mmoe"  # "stl", "mtl", or "mmoe"
N_TRAITS = 37
IN_CHANNELS = 150
BASE_CHANNELS = 48
NORM = "gn"
DROPOUT_P = 0.1
STRIDE_BLOCKS = (1, 1, 1, 1)
N_EXPERTS = 6
EXPERT_HIDDEN = 192

# Checkpoints and output
SCRIPT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    SCRIPT_ROOT / "results" / MODEL_TYPE / "checkpoints" / f"{MODEL_TYPE}_best.pth"
)
DEFAULT_PREDS_PATH = (
    SCRIPT_ROOT / "results" / MODEL_TYPE / "eval_results" / "test_preds.npy"
)
DEFAULT_TRUE_PATH = (
    SCRIPT_ROOT / "results" / MODEL_TYPE / "eval_results" / "test_targets.npy"
)
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "results" / MODEL_TYPE / "heatmaps"
TRAIT_DIR = ZARR_DIR.parents[1] / "targets" / "comb"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _split_target_and_source(
    y_full: torch.Tensor,
    n_traits: int = 37,
    n_stats: int = 7,
    mean_idx: int = 0,
    source_idx: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract trait means and source masks from concatenated targets."""
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
) -> nn.Module:
    """Instantiate a model matching training/evaluation config."""
    if model_type == "stl":
        return STLModel(
            in_channels=in_channels,
            n_traits=n_traits,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout_p,
            stride_blocks=stride_blocks,
        )
    if model_type == "mtl":
        return MTLModel(
            in_channels=in_channels,
            n_traits=n_traits,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout_p,
            stride_blocks=stride_blocks,
        )
    if model_type == "mmoe":
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
    raise ValueError(f"Unknown model_type: {model_type}")


def resolve_trait_ids(n_traits: int) -> list[str]:
    """Resolve trait IDs from merged target rasters when available."""
    if TRAIT_DIR.exists():
        trait_ids = sorted(path.stem for path in TRAIT_DIR.glob("*.tif"))
        if len(trait_ids) >= n_traits:
            return trait_ids[:n_traits]
    return [f"trait_{idx:02d}" for idx in range(n_traits)]


def _reorder_square_matrix(
    matrix: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Reorder a square matrix so similar traits appear together visually."""
    n_traits = matrix.shape[0]
    filled = np.nan_to_num(matrix.astype(np.float64), nan=0.0)
    filled = (filled + filled.T) / 2.0
    np.fill_diagonal(filled, 1.0)

    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        distance = np.clip(1.0 - filled, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        linkage_matrix = linkage(squareform(distance), method="average")
        order = np.array(dendrogram(linkage_matrix, no_plot=True)["leaves"])
    except ImportError:
        eigvals, eigvecs = np.linalg.eigh(filled)
        order = np.argsort(eigvecs[:, np.argmax(eigvals)])

    reordered_matrix = matrix[np.ix_(order, order)]
    reordered_names = [names[idx] for idx in order]
    return reordered_matrix, reordered_names, order


def _plot_matrix(
    matrix: np.ndarray,
    names: list[str],
    save_path: Path,
    title: str,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; heatmap generation is skipped.")
        return matrix, names, np.arange(len(names))

    reordered_matrix, reordered_names, order = _reorder_square_matrix(matrix, names)

    size = max(10, len(reordered_names) * 0.35)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(
        reordered_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(reordered_names)))
    ax.set_yticks(range(len(reordered_names)))
    ax.set_xticklabels(reordered_names, rotation=90, fontsize=8)
    ax.set_yticklabels(reordered_names, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {save_path}")
    return reordered_matrix, reordered_names, order


def generate_prediction_heatmaps(
    preds_path: Path,
    true_path: Path,
    save_dir: Path,
    n_samples: int = 5,
    trait_index: int = 0,
) -> None:
    """Generate simple prediction-vs-truth patch heatmaps for one trait."""
    save_dir.mkdir(parents=True, exist_ok=True)
    preds = np.load(preds_path)
    truth = np.load(true_path)

    if preds.ndim == 3:
        preds = preds[:, None, ...]
    if truth.ndim == 3:
        truth = truth[:, None, ...]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; heatmap generation is skipped.")
        return

    trait_ids = resolve_trait_ids(preds.shape[1])
    trait_index = min(max(trait_index, 0), preds.shape[1] - 1)
    n_samples = min(n_samples, preds.shape[0])

    for sample_idx in range(n_samples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(preds[sample_idx, trait_index], cmap="viridis")
        axes[0].set_title(f"Pred {trait_ids[trait_index]} sample {sample_idx}")
        axes[1].imshow(truth[sample_idx, trait_index], cmap="viridis")
        axes[1].set_title(f"True {trait_ids[trait_index]} sample {sample_idx}")
        fig.tight_layout()
        fig_path = (
            save_dir / f"pred_vs_true_sample{sample_idx}_{trait_ids[trait_index]}.png"
        )
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    print(f"Saved prediction heatmaps to {save_dir}")


def _pairwise_corr_matrix(values: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation with NaN-aware masking."""
    n_traits = values.shape[1]
    corr = np.full((n_traits, n_traits), np.nan, dtype=np.float32)

    for i in range(n_traits):
        for j in range(n_traits):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            if np.sum(valid) < 3:
                continue
            vec_i = values[valid, i]
            vec_j = values[valid, j]
            if np.std(vec_i) < 1e-8 or np.std(vec_j) < 1e-8:
                continue
            corr[i, j] = float(np.corrcoef(vec_i, vec_j)[0, 1])

    return corr


def generate_target_correlation_heatmap(
    save_dir: Path,
    split: str = "val",
    n_batches: int | None = None,
    source_filter: str = "splot",
) -> np.ndarray:
    """Generate target-only trait correlation heatmap without using a model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    dataloader = get_mtl_dataloader(
        ZARR_DIR,
        split,
        PREDICTORS,
        TARGETS,
        BATCH_SIZE,
        NUM_WORKERS,
    )

    all_targets = []
    all_masks = []
    for batch_idx, (_, y_full) in enumerate(dataloader):
        if n_batches is not None and batch_idx >= n_batches:
            break

        y_target, src_mask = _split_target_and_source(y_full.to(dtype=torch.float32))
        all_targets.append(y_target.cpu().numpy())
        all_masks.append(src_mask.cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    if source_filter == "splot":
        valid = masks == 2
    elif source_filter == "gbif":
        valid = masks == 1
    elif source_filter == "all":
        valid = masks > 0
    else:
        raise ValueError(f"Unknown source_filter: {source_filter}")

    targets = np.where(valid, targets, np.nan)
    flat_targets = targets.transpose(0, 2, 3, 1).reshape(-1, N_TRAITS)
    corr = _pairwise_corr_matrix(flat_targets)

    trait_ids = resolve_trait_ids(N_TRAITS)
    save_path = save_dir / f"target_correlation_{split}_{source_filter}.png"
    corr_ordered, ordered_trait_ids, order = _plot_matrix(
        corr,
        trait_ids,
        save_path,
        title=f"Target correlation ({split}, {source_filter})",
    )
    np.save(save_dir / f"target_correlation_{split}_{source_filter}.npy", corr)
    np.save(
        save_dir / f"target_correlation_{split}_{source_filter}_ordered.npy",
        corr_ordered,
    )
    with open(
        save_dir / f"target_correlation_{split}_{source_filter}_order.txt", "w"
    ) as handle:
        handle.write("\n".join(ordered_trait_ids) + "\n")

    pairs = []
    for i in range(N_TRAITS):
        for j in range(i + 1, N_TRAITS):
            if np.isfinite(corr[i, j]):
                pairs.append((corr[i, j], trait_ids[i], trait_ids[j]))

    print("Top correlated target pairs:")
    for score, name_a, name_b in sorted(pairs, reverse=True)[:10]:
        print(f"  {name_a:<8} <-> {name_b:<8}  r={score:.3f}")

    print("Top anti-correlated target pairs:")
    for score, name_a, name_b in sorted(pairs)[:10]:
        print(f"  {name_a:<8} <-> {name_b:<8}  r={score:.3f}")

    return corr


def generate_heatmaps(
    preds_path: Path,
    true_path: Path,
    save_dir: Path,
    n_samples: int = 5,
) -> None:
    """Backward-compatible wrapper for the original prediction heatmap API."""
    generate_prediction_heatmaps(
        preds_path,
        true_path,
        save_dir,
        n_samples=n_samples,
        trait_index=0,
    )


def _masked_trait_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    src_mask: torch.Tensor,
    trait_idx: int,
    source_filter: str,
) -> torch.Tensor | None:
    """Compute masked MSE for one trait over the full patch."""
    pred_t = y_pred[:, trait_idx, :, :]
    true_t = y_true[:, trait_idx, :, :]
    mask_t = src_mask[:, trait_idx, :, :]

    valid = torch.isfinite(pred_t) & torch.isfinite(true_t)
    if source_filter == "splot":
        valid &= mask_t == 2
    elif source_filter == "gbif":
        valid &= mask_t == 1
    elif source_filter == "all":
        valid &= mask_t > 0
    else:
        raise ValueError(f"Unknown source_filter: {source_filter}")

    if not valid.any():
        return None
    return F.mse_loss(pred_t[valid], true_t[valid])


def generate_gradient_similarity_heatmap(
    checkpoint_path: Path,
    save_dir: Path,
    split: str = "val",
    n_batches: int = 10,
    source_filter: str = "splot",
) -> np.ndarray:
    """Generate gradient similarity heatmap using shared encoder gradients."""
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    dataloader = get_mtl_dataloader(
        ZARR_DIR,
        split,
        PREDICTORS,
        TARGETS,
        BATCH_SIZE,
        NUM_WORKERS,
    )

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
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    encoder_params = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    n_params = sum(parameter.numel() for parameter in encoder_params)
    grad_accum = torch.zeros(N_TRAITS, n_params, device=device)
    grad_count = torch.zeros(N_TRAITS, device=device)

    for batch_idx, (X, y_full) in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
        y_full = y_full.to(device=device, dtype=torch.float32)
        y_target, src_mask = _split_target_and_source(y_full)

        for trait_idx in range(N_TRAITS):
            model.zero_grad(set_to_none=True)
            y_pred = model(X)
            loss_t = _masked_trait_loss(
                y_pred,
                y_target,
                src_mask,
                trait_idx,
                source_filter,
            )
            if loss_t is None:
                continue
            loss_t.backward()

            gradients = [
                parameter.grad.detach().flatten()
                for parameter in encoder_params
                if parameter.grad is not None
            ]
            if gradients:
                grad_vector = torch.cat(gradients)
                grad_accum[trait_idx] += grad_vector
                grad_count[trait_idx] += 1

    valid_traits = grad_count > 0
    for trait_idx in range(N_TRAITS):
        if grad_count[trait_idx] > 0:
            grad_accum[trait_idx] /= grad_count[trait_idx]

    grad_norm = F.normalize(grad_accum, dim=1)
    similarity = (grad_norm @ grad_norm.T).cpu().numpy()

    invalid = (~valid_traits.cpu().numpy()).astype(bool)
    similarity[invalid, :] = np.nan
    similarity[:, invalid] = np.nan

    trait_ids = resolve_trait_ids(N_TRAITS)
    save_path = save_dir / f"gradient_similarity_{MODEL_TYPE}_{source_filter}.png"
    similarity_ordered, ordered_trait_ids, order = _plot_matrix(
        similarity,
        trait_ids,
        save_path,
        title=f"Gradient similarity ({MODEL_TYPE}, {source_filter})",
    )

    pairs = []
    for i in range(N_TRAITS):
        for j in range(i + 1, N_TRAITS):
            if np.isfinite(similarity[i, j]):
                pairs.append((similarity[i, j], trait_ids[i], trait_ids[j]))

    print("Top positive trait pairs:")
    for score, name_a, name_b in sorted(pairs, reverse=True)[:10]:
        print(f"  {name_a:<8} <-> {name_b:<8}  cos={score:.3f}")

    print("Top negative trait pairs:")
    for score, name_a, name_b in sorted(pairs)[:10]:
        print(f"  {name_a:<8} <-> {name_b:<8}  cos={score:.3f}")

    np.save(
        save_dir / f"gradient_similarity_{MODEL_TYPE}_{source_filter}.npy", similarity
    )
    np.save(
        save_dir / f"gradient_similarity_{MODEL_TYPE}_{source_filter}_ordered.npy",
        similarity_ordered,
    )
    with open(
        save_dir / f"gradient_similarity_{MODEL_TYPE}_{source_filter}_order.txt",
        "w",
    ) as handle:
        handle.write("\n".join(ordered_trait_ids) + "\n")
    return similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MTL heatmaps and trait-similarity plots."
    )
    parser.add_argument(
        "--mode",
        choices=["pred", "grad", "corr", "all"],
        default="grad",
    )
    parser.add_argument("--preds-path", type=Path, default=DEFAULT_PREDS_PATH)
    parser.add_argument("--true-path", type=Path, default=DEFAULT_TRUE_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--trait-index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument(
        "--source-filter",
        choices=["all", "splot", "gbif"],
        default="splot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {"pred", "all"}:
        generate_prediction_heatmaps(
            args.preds_path,
            args.true_path,
            args.output_dir / "pred_vs_true",
            n_samples=args.n_samples,
            trait_index=args.trait_index,
        )

    if args.mode in {"grad", "all"}:
        generate_gradient_similarity_heatmap(
            args.checkpoint_path,
            args.output_dir / "task_similarity",
            split=args.split,
            n_batches=args.n_batches,
            source_filter=args.source_filter,
        )

    if args.mode in {"corr", "all"}:
        generate_target_correlation_heatmap(
            args.output_dir / "target_correlation",
            split=args.split,
            n_batches=args.n_batches,
            source_filter=args.source_filter,
        )


if __name__ == "__main__":
    main()
