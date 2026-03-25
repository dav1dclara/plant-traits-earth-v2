import json
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from ptev2.data.dataloader import get_dataloader
from ptev2.metrics.aoa import collect_patch_features, compute_aoa_metrics
from ptev2.metrics.core import pearson_r as pearson_r_metric
from ptev2.metrics.core import r2_score
from ptev2.utils import (
    run_name_from_cfg,
    seed_all,
)


def _band_names_for_source(source: str, bands_per_trait: int) -> list[str]:
    source_l = source.lower()
    if bands_per_trait == 6:
        if source_l == "gbif":
            return ["mean", "std", "median", "q05", "q95", "count"]
        if source_l == "splot":
            return ["mean", "count", "std", "median", "q05", "q95"]
    return [f"band{i}" for i in range(1, bands_per_trait + 1)]


def _infer_checkpoint_output_channels(
    state_dict: dict[str, torch.Tensor],
) -> int | None:
    for key, tensor in state_dict.items():
        if key.endswith("head.bias") and tensor.ndim == 1:
            return int(tensor.shape[0])
    for key, tensor in state_dict.items():
        if key.endswith("head.weight") and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def evaluate_model(cfg: DictConfig) -> dict:
    seed_all(cfg.training.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    run = None
    if cfg.wandb.enabled:
        eval_run_name = f"eval_{run_name_from_cfg(cfg)}"
        try:
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=eval_run_name,
                job_type="evaluation",
                config=OmegaConf.to_container(cfg, resolve=False),
                reinit="finish_previous",
            )
            print(f"W&B eval run started: {eval_run_name}")
        except Exception as exc:
            print(f"W&B init failed ({exc}). Continuing without W&B logging.")
            run = None

    predictors = [k for k, v in cfg.training.data.predictors.items() if v.use]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    target = cfg.training.data.target.source
    zarr_dir = Path(cfg.training.data.zarr_dir)
    batch_size = int(cfg.training.data_loaders.batch_size)
    num_workers = int(cfg.training.data_loaders.num_workers)
    test_split = str(cfg.training.data.test_split)

    trait_ids_cfg = OmegaConf.select(cfg, "training.data.target.trait_ids")
    trait_positions_cfg = OmegaConf.select(cfg, "training.data.target.trait_positions")
    bands_per_trait = int(
        OmegaConf.select(cfg, "training.data.target.bands_per_trait", default=6)
    )

    selected_trait_ids = (
        [int(trait_ids_cfg)]
        if isinstance(trait_ids_cfg, (str, int))
        else [int(t) for t in trait_ids_cfg]
    )
    trait_positions = (
        [int(trait_positions_cfg)]
        if isinstance(trait_positions_cfg, int)
        else [int(p) for p in trait_positions_cfg]
    )
    if len(selected_trait_ids) != len(trait_positions):
        raise ValueError(
            "training.data.target.trait_ids und trait_positions muessen gleich lang sein."
        )

    target_channel_indices: list[int] = []
    for pos in trait_positions:
        start = pos * bands_per_trait
        target_channel_indices.extend(range(start, start + bands_per_trait))

    selected_trait_count = len(selected_trait_ids)
    effective_target_channels = len(target_channel_indices)
    band_names = _band_names_for_source(target, bands_per_trait)

    print(
        "Target layout: "
        f"traits={selected_trait_count}, "
        f"bands_per_trait={bands_per_trait}, "
        f"output_channels={effective_target_channels}"
    )
    print(f"Band order ({target}): {band_names}")

    test_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=test_split,
        predictors=predictors,
        target=target,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    compute_aoa = bool(OmegaConf.select(cfg, "evaluation.compute_aoa", default=True))
    aoa_metrics = None
    if compute_aoa:
        train_loader = get_dataloader(
            zarr_dir=zarr_dir,
            split=cfg.training.data.train_split,
            predictors=predictors,
            target=target,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        train_features = collect_patch_features(train_loader, device=device)
        test_features = collect_patch_features(test_loader, device=device)
        aoa_metrics = compute_aoa_metrics(train_features, test_features, q=0.95)

    checkpoint_override = OmegaConf.select(cfg, "evaluation.checkpoint_path")
    if checkpoint_override:
        checkpoint_path = Path(str(checkpoint_override))
    else:
        checkpoint_path = (
            Path(cfg.training.checkpoint.dir) / f"{run_name_from_cfg(cfg)}.pth"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Pass a custom path with +evaluation.checkpoint_path=/path/to/model.pth"
        )

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    checkpoint_output_channels = _infer_checkpoint_output_channels(state)
    if (
        checkpoint_output_channels is not None
        and checkpoint_output_channels != effective_target_channels
    ):
        raise ValueError(
            "Checkpoint/model channel mismatch: "
            f"checkpoint_output_channels={checkpoint_output_channels}, "
            f"expected_output_channels={effective_target_channels}."
        )

    model = instantiate(cfg.models.active, out_channels=effective_target_channels).to(
        device
    )
    model.load_state_dict(state)

    model.eval()
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    skipped_batches = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"Evaluating [{test_split}]"):
            X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
            y = y.to(device=device, dtype=torch.float32)

            y = y[:, target_channel_indices]

            valid_samples = torch.isfinite(y).any(dim=(1, 2, 3))
            if not bool(valid_samples.any()):
                skipped_batches += 1
                continue
            X = X[valid_samples]
            y = y[valid_samples]

            y_pred = model(X)
            if y_pred.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                )

            valid = torch.isfinite(y_pred) & torch.isfinite(y)
            if not bool(valid.any()):
                skipped_batches += 1
                continue

            all_y_pred.append(y_pred[valid].detach().cpu().numpy())
            all_y_true.append(y[valid].detach().cpu().numpy())

    if all_y_true:
        y_true_vec = np.concatenate(all_y_true)
        y_pred_vec = np.concatenate(all_y_pred)
        n_valid = int(y_true_vec.size)

        r2 = float(r2_score(y_true_vec, y_pred_vec))
        pearson_r = float(pearson_r_metric(y_true_vec, y_pred_vec))
    else:
        n_valid = 0
        r2 = float("nan")
        pearson_r = float("nan")

    results = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "split": test_split,
        "n_valid": n_valid,
        "skipped_batches": skipped_batches,
        "r2": r2,
        "pearson_r": pearson_r,
    }
    if aoa_metrics is not None:
        results.update(aoa_metrics)

    output_path_cfg = OmegaConf.select(cfg, "evaluation.output_path")
    output_path = (
        Path(str(output_path_cfg))
        if output_path_cfg
        else checkpoint_path.with_suffix(".test_metrics.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Evaluation finished")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  split:      {test_split}")
    print(f"  n_valid:    {n_valid}")
    print(f"  r2:         {r2:.6f}")
    print(f"  pearson_r:  {pearson_r:.6f}")
    if aoa_metrics is not None:
        print(f"  aoa_cover:  {aoa_metrics['aoa_coverage']:.6f}")
        print(f"  aoa_thr:    {aoa_metrics['aoa_threshold']:.6f}")
    print(f"  metrics:    {output_path}")

    if run is not None:
        wandb.log(results)
        run.summary["metrics_file"] = str(output_path)
        run.finish()

    return results


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
