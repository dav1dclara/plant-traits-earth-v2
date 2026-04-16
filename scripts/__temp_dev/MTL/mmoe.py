"""
Main MMoE script.
Similar to MTL but for Mixture of Experts.
"""

from pathlib import Path

from evaluate import evaluate_mtl_model
from heatmap import generate_heatmaps
from train import train_mtl_model


def run_mmoe_experiment(
    zarr_dir: Path,
    predictors: list[str],
    targets: list[str],
    n_experts: int = 4,
    epochs: int = 10,
    batch_size: int = 32,
    save_dir: Path = Path("MTL/results_mmoe"),
):
    """
    Run full MMoE experiment.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print("Training MMoE model...")
    train_mtl_model(
        zarr_dir=zarr_dir,
        predictors=predictors,
        targets=targets,
        model_type="mmoe",
        n_experts=n_experts,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=save_dir,
    )

    # Evaluate
    print("Evaluating...")
    checkpoint_path = save_dir / "checkpoints" / "mmoe_best.pth"
    mse = evaluate_mtl_model(
        zarr_dir=zarr_dir,
        predictors=predictors,
        targets=targets,
        model_type="mmoe",
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        output_dir=save_dir,
    )

    # Heatmaps
    print("Generating heatmaps...")
    preds_path = save_dir / "predictions" / "test_preds.npy"
    true_path = save_dir / "predictions" / "test_true.npy"
    heatmap_dir = save_dir / "heatmaps"
    generate_heatmaps(preds_path, true_path, heatmap_dir)

    print(f"MMoE experiment complete. Results in {save_dir}")


if __name__ == "__main__":
    zarr_dir = Path("/scratch3/plant-traits-v2/data/22km/chips/patch15_stride10")
    predictors = ["canopy_height", "modis", "soil_grids", "vodca", "worldclim"]
    targets = ["comb"]
    run_mmoe_experiment(zarr_dir, predictors, targets)
