"""
Evaluation script for MTL models.
Loads a checkpoint, runs the test split, and exports metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from dataloader import get_mtl_dataloader
from models import build_model
from utils import ensure_dir, parse_list_argument, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained STL/MTL/MMoE model."
    )
    parser.add_argument(
        "--zarr-dir", type=Path, required=True, help="Root folder containing test.zarr"
    )
    parser.add_argument(
        "--predictors", type=str, required=True, help="Comma-separated predictor names"
    )
    parser.add_argument(
        "--targets", type=str, required=True, help="Comma-separated target names"
    )
    parser.add_argument("--model", choices=["stl", "mtl", "mmoe"], default="mtl")
    parser.add_argument(
        "--checkpoint-path", type=Path, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        default=4,
        help="Number of MMoE experts if using mmoe model",
    )
    parser.add_argument(
        "--expert-dim",
        type=int,
        default=64,
        help="Expert hidden dimension for MMoE model",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store evaluation outputs",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "results"
    return args


def compute_metrics(preds: np.ndarray, targets: np.ndarray, names: list[str]) -> dict:
    metrics: dict = {}
    if preds.ndim == 3:
        preds = preds[:, None, ...]
    if targets.ndim == 3:
        targets = targets[:, None, ...]

    n_tasks = preds.shape[1]
    if len(names) != n_tasks:
        names = [f"task_{i}" for i in range(n_tasks)]

    for idx, name in enumerate(names):
        pred = preds[:, idx].astype(np.float64)
        truth = targets[:, idx].astype(np.float64)
        valid = np.isfinite(pred) & np.isfinite(truth)
        if valid.sum() == 0:
            metrics[name] = {"mae": None, "mse": None, "count": 0}
            continue
        diff = pred[valid] - truth[valid]
        metrics[name] = {
            "mae": float(np.mean(np.abs(diff))),
            "mse": float(np.mean(diff**2)),
            "count": int(valid.sum()),
        }
    return metrics


def evaluate_mtl_model(
    zarr_dir: Path,
    predictors: list[str],
    targets: list[str],
    model_type: str = "mtl",
    checkpoint_path: Path | None = None,
    n_experts: int = 4,
    expert_dim: int = 64,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> dict:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = ensure_dir(output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    preds_dir = ensure_dir(output_dir / "predictions")

    targets = list(targets)
    predictors = list(predictors)
    n_tasks = len(targets)

    test_loader = get_mtl_dataloader(
        zarr_dir, "test", predictors, targets, batch_size, num_workers
    )
    sample_X, _ = next(iter(test_loader))
    in_channels = sample_X.shape[1]

    model = build_model(
        model_type,
        in_channels=in_channels,
        n_tasks=n_tasks,
        n_experts=n_experts,
        expert_dim=expert_dim,
    )
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        state["model_state"]
        if isinstance(state, dict) and "model_state" in state
        else state
    )
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    np.save(preds_dir / "test_preds.npy", preds_np)
    np.save(preds_dir / "test_true.npy", targets_np)

    metrics = compute_metrics(preds_np, targets_np, targets)
    save_json(metrics, metrics_dir / "eval_metrics.json")
    print(f"Saved evaluation metrics to {metrics_dir / 'eval_metrics.json'}")
    return metrics


def main() -> None:
    args = parse_args()
    targets = parse_list_argument(args.targets)
    predictors = parse_list_argument(args.predictors)
    evaluate_mtl_model(
        zarr_dir=args.zarr_dir,
        predictors=predictors,
        targets=targets,
        model_type=args.model,
        checkpoint_path=args.checkpoint_path,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
