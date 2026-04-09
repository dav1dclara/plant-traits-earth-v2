"""
Training script for MTL.
Provides training logic and a simple CLI for STL/MTL/MMoE experiments.
"""

import argparse
import time
from pathlib import Path

import torch
from dataloader import get_mtl_dataloader
from loss import MaskedMSELoss
from models import MMoEModel, MTLModel, STLModel
from torch import nn, optim
from utils import ensure_dir, parse_list_argument, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple STL/MTL/MMoE model.")
    parser.add_argument(
        "--zarr-dir",
        type=Path,
        required=True,
        help="Root folder containing train.zarr, val.zarr, test.zarr",
    )
    parser.add_argument(
        "--predictors", type=str, required=True, help="Comma-separated predictor names"
    )
    parser.add_argument(
        "--targets", type=str, required=True, help="Comma-separated target names"
    )
    parser.add_argument("--model", choices=["stl", "mtl", "mmoe"], default="mtl")
    parser.add_argument(
        "--n-experts", type=int, default=4, help="Number of MMoE experts"
    )
    parser.add_argument(
        "--expert-dim",
        type=int,
        default=64,
        help="Expert hidden dimension for MMoE model",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store checkpoints and metrics",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "results"
    return args


def build_model(model_type: str, in_channels: int, n_tasks: int, **kwargs) -> nn.Module:
    if model_type == "stl":
        return STLModel(in_channels=in_channels, out_channels=n_tasks, **kwargs)
    if model_type == "mtl":
        return MTLModel(in_channels=in_channels, n_traits=n_tasks, **kwargs)
    if model_type == "mmoe":
        return MMoEModel(in_channels=in_channels, n_traits=n_tasks, **kwargs)
    raise ValueError(f"Unknown model type: {model_type}")


def train_mtl_model(
    zarr_dir: Path,
    predictors: list[str],
    targets: list[str],
    model_type: str = "mtl",
    n_experts: int = 4,
    expert_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_workers: int = 4,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> dict:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")

    targets = list(targets)
    predictors = list(predictors)
    n_tasks = len(targets)

    train_loader = get_mtl_dataloader(
        zarr_dir, "train", predictors, targets, batch_size, num_workers
    )
    val_loader = get_mtl_dataloader(
        zarr_dir, "val", predictors, targets, batch_size, num_workers
    )

    sample_X, _ = next(iter(train_loader))
    in_channels = sample_X.shape[1]

    model = build_model(
        model_type,
        in_channels=in_channels,
        n_tasks=n_tasks,
        n_experts=n_experts,
        expert_dim=expert_dim,
    )
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = MaskedMSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_checkpoint_path = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                preds = model(X)
                val_loss += loss_fn(preds, y).item()
        val_loss /= max(len(val_loader), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        checkpoint_path = checkpoints_dir / f"{model_type}_epoch_{epoch}.pth"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_type": model_type,
                "epoch": epoch,
            },
            checkpoint_path,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoints_dir / f"{model_type}_best.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": model_type,
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )

    summary = {
        "model_type": model_type,
        "predictors": predictors,
        "targets": targets,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None,
    }
    save_json(summary, metrics_dir / "train_metrics.json")
    print(f"Saved training metrics to {metrics_dir / 'train_metrics.json'}")
    return summary


def main() -> None:
    args = parse_args()
    targets = parse_list_argument(args.targets)
    predictors = parse_list_argument(args.predictors)
    train_mtl_model(
        zarr_dir=args.zarr_dir,
        predictors=predictors,
        targets=targets,
        model_type=args.model,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
