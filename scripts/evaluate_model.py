import json
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader
from ptev2.metrics.aoa import collect_patch_features, compute_aoa_metrics
from ptev2.metrics.evaluation import summarize_single_trait_metrics
from ptev2.utils import seed_all


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


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_override = OmegaConf.select(cfg, "checkpoint_path")
    if not checkpoint_override:
        checkpoint_override = OmegaConf.select(cfg, "evaluation.checkpoint_path")
    if not checkpoint_override:
        raise ValueError("Set checkpoint_path to evaluate a single checkpoint.")
    return Path(str(checkpoint_override))


def _load_checkpoint_and_config(
    checkpoint_path: Path, device: torch.device
) -> tuple[dict, DictConfig]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' did not contain a dict state."
        )

    training_cfg = state.get("config")
    if training_cfg is None:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain embedded training config under 'config'."
        )

    return state, OmegaConf.create(training_cfg)


def _build_target_layout_from_train_cfg(
    train_cfg: DictConfig,
    train_store: zarr.Group,
) -> tuple[str, list[str], list[str], list[int], list[int], int]:
    target_cfg = train_cfg.data.targets
    target_dataset = str(target_cfg.dataset)

    zarr_dataset_names = list(train_store["targets"].keys())
    if target_dataset not in zarr_dataset_names:
        raise ValueError(
            f"Dataset '{target_dataset}' not found in zarr. Available: {', '.join(zarr_dataset_names)}"
        )

    zarr_band_names = [str(v) for v in train_store["targets"].attrs["band_names"]]
    band_to_idx = {name: idx for idx, name in enumerate(zarr_band_names)}

    zarr_all_traits = [
        str(f).replace("X", "").replace(".tif", "")
        for f in train_store[f"targets/{target_dataset}"].attrs["files"]
    ]
    traits = (
        [str(v) for v in target_cfg.traits] if target_cfg.traits else zarr_all_traits
    )
    if not traits:
        raise ValueError("data.targets.traits must not be empty after resolution.")
    for trait in traits:
        if trait not in zarr_all_traits:
            raise ValueError(
                f"Trait '{trait}' not found in dataset '{target_dataset}' in zarr."
            )

    cfg_bands = [str(v) for v in target_cfg.bands]
    if not cfg_bands:
        raise ValueError("data.targets.bands must not be empty.")
    for band in cfg_bands:
        if band not in band_to_idx:
            raise ValueError(
                f"Band '{band}' not found in zarr. Available: {', '.join(zarr_band_names)}"
            )

    n_bands = len(zarr_band_names)
    target_indices = [
        trait_pos * n_bands + band_to_idx[band]
        for trait_pos in range(len(traits))
        for band in cfg_bands
    ]
    source_indices = [
        trait_pos * n_bands + band_to_idx["source"] for trait_pos in range(len(traits))
    ]

    return (
        target_dataset,
        traits,
        cfg_bands,
        target_indices,
        source_indices,
        len(traits),
    )


def _resolve_train_zarr_dir(train_cfg: DictConfig) -> Path:
    return (
        Path(str(train_cfg.data.root_dir))
        / f"{train_cfg.data.resolution_km}km"
        / "chips"
        / f"patch{train_cfg.data.patch_size}_stride{train_cfg.data.stride}"
    )


def evaluate_model(cfg: DictConfig) -> dict:
    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")
    print(f"Using device: {device}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state, train_cfg = _load_checkpoint_and_config(checkpoint_path, device)
    seed_all(int(train_cfg.train.seed))

    predictors = [k for k, v in train_cfg.data.predictors.items() if v.use]
    if not predictors:
        raise ValueError("No predictors enabled in checkpoint training config.")

    zarr_dir = _resolve_train_zarr_dir(train_cfg)
    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")
    test_zarr_dir_cfg = OmegaConf.select(cfg, "test_zarr_dir")
    if test_zarr_dir_cfg is None:
        test_zarr_dir_cfg = OmegaConf.select(cfg, "evaluation.test_zarr_dir")
    test_zarr_dir = Path(str(test_zarr_dir_cfg)) if test_zarr_dir_cfg else zarr_dir
    test_split = str(
        OmegaConf.select(cfg, "test_split")
        or OmegaConf.select(cfg, "evaluation.test_split")
        or "test"
    )
    batch_size = int(train_cfg.data_loaders.batch_size)
    num_workers = int(train_cfg.data_loaders.num_workers)

    (
        target_dataset,
        traits,
        selected_bands,
        target_channel_indices,
        source_indices,
        n_traits,
    ) = _build_target_layout_from_train_cfg(
        train_cfg,
        train_store,
    )
    print(
        "Target layout: "
        f"traits={n_traits}, "
        f"selected_bands={selected_bands}, "
        f"output_channels={len(target_channel_indices)}"
    )

    test_loader = get_dataloader(
        zarr_dir=test_zarr_dir,
        split=test_split,
        predictors=predictors,
        target=target_dataset,
        target_indices=target_channel_indices,
        source_indices=source_indices,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    compute_aoa = bool(
        OmegaConf.select(cfg, "compute_aoa")
        if OmegaConf.select(cfg, "compute_aoa") is not None
        else OmegaConf.select(cfg, "evaluation.compute_aoa") or False
    )
    aoa_metrics = None
    if compute_aoa:
        train_split = str(train_cfg.data.get("train_split", "train"))
        train_loader = get_dataloader(
            zarr_dir=zarr_dir,
            split=train_split,
            predictors=predictors,
            target=target_dataset,
            target_indices=target_channel_indices,
            source_indices=source_indices,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        train_features = collect_patch_features(train_loader, device=device)
        test_features = collect_patch_features(test_loader, device=device)
        aoa_metrics = compute_aoa_metrics(train_features, test_features, q=0.95)

    checkpoint_state = state.get("state_dict", state)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Checkpoint state for '{checkpoint_path}' is invalid.")

    checkpoint_output_channels = _infer_checkpoint_output_channels(checkpoint_state)
    if checkpoint_output_channels is not None and checkpoint_output_channels != len(
        target_channel_indices
    ):
        raise ValueError(
            "Checkpoint/model channel mismatch: "
            f"checkpoint_output_channels={checkpoint_output_channels}, "
            f"expected_output_channels={len(target_channel_indices)}."
        )

    total_pred_bands = 0
    for predictor in predictors:
        total_pred_bands += train_store[f"predictors/{predictor}"].shape[1]
    model = instantiate(
        train_cfg.models,
        in_channels=total_pred_bands,
        out_channels=len(target_channel_indices),
    ).to(device)
    model.load_state_dict(checkpoint_state)
    model.eval()

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_src: list[np.ndarray] = []
    skipped_batches = 0

    with torch.no_grad():
        for X, y, src in tqdm(test_loader, desc=f"Evaluating [{checkpoint_path.stem}]"):
            X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
            y = y.to(device=device, dtype=torch.float32)
            src = src.to(device=device, dtype=torch.float32)

            y_pred = model(X)
            if y_pred.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                )

            valid = torch.isfinite(y_pred) & torch.isfinite(y) & (src > 0)
            if not bool(valid.any()):
                skipped_batches += 1
                continue

            all_y_pred.append(y_pred.detach().cpu().numpy())
            all_y_true.append(y.detach().cpu().numpy())
            all_src.append(src.detach().cpu().numpy())

    if all_y_true:
        y_true_stack = np.concatenate(all_y_true, axis=0)
        y_pred_stack = np.concatenate(all_y_pred, axis=0)
        src_stack = np.concatenate(all_src, axis=0)
        metric_summary = summarize_single_trait_metrics(
            y_true=y_true_stack,
            y_pred=y_pred_stack,
            source_mask=src_stack,
            trait_names=traits,
            n_bands=len(selected_bands),
        )
    else:
        metric_summary = {
            "rmse": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "n_valid": 0,
            "cov": float("nan"),
            "cov_mean": float("nan"),
            "cov_median": float("nan"),
            "cov_p95": float("nan"),
            "cov_n": 0.0,
            "trait_metrics": {},
        }

    results_item = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "split": test_split,
        "n_valid": int(metric_summary["n_valid"]),
        "skipped_batches": skipped_batches,
        "rmse": metric_summary["rmse"],
        "r2": metric_summary["r2"],
        "pearson_r": metric_summary["pearson_r"],
        "cov": metric_summary["cov"],
        "cov_mean": metric_summary["cov_mean"],
        "cov_median": metric_summary["cov_median"],
        "cov_p95": metric_summary["cov_p95"],
        "cov_n": metric_summary["cov_n"],
        "trait_metrics": metric_summary["trait_metrics"],
    }
    if aoa_metrics is not None:
        results_item.update(aoa_metrics)

    output_dir_cfg = OmegaConf.select(cfg, "output_dir")
    if output_dir_cfg is None:
        output_dir_cfg = OmegaConf.select(cfg, "evaluation.output_dir")
    output_dir = Path(str(output_dir_cfg)) if output_dir_cfg else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_path.stem}.test_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results_item, handle, indent=2)

    print("Evaluation finished")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  train_zarr:  {zarr_dir}")
    print(f"  test_zarr:   {test_zarr_dir}")
    print(f"  split:      {test_split}")
    print(f"  n_valid:    {metric_summary['n_valid']}")
    print(f"  rmse:       {metric_summary['rmse']:.6f}")
    print(f"  r2:         {metric_summary['r2']:.6f}")
    print(f"  pearson_r:  {metric_summary['pearson_r']:.6f}")
    print(f"  cov_mean:   {metric_summary['cov_mean']:.6f}")
    for trait_name, trait_result in metric_summary["trait_metrics"].items():
        print(
            f"  trait={trait_name}: rmse={trait_result['rmse']:.6f}, r2={trait_result['r2']:.6f}, pearson_r={trait_result['pearson_r']:.6f}, cov={trait_result['cov']:.6f}"
        )
    print(f"  metrics:    {output_path}")

    wandb_enabled_cfg = OmegaConf.select(cfg, "wandb.enabled")
    wandb_enabled_train = OmegaConf.select(train_cfg, "wandb.enabled")
    wandb_enabled = bool(
        wandb_enabled_cfg
        if wandb_enabled_cfg is not None
        else (wandb_enabled_train if wandb_enabled_train is not None else False)
    )

    if wandb_enabled:
        import wandb

        base_run_name = OmegaConf.select(train_cfg, "train.run_name")
        if not base_run_name:
            base_run_name = checkpoint_path.stem
        run_name = f"eval_{base_run_name}"

        wandb_project = OmegaConf.select(cfg, "wandb.project") or OmegaConf.select(
            train_cfg, "wandb.project"
        )
        wandb_entity = OmegaConf.select(cfg, "wandb.entity") or OmegaConf.select(
            train_cfg, "wandb.entity"
        )

        if not wandb_project or not wandb_entity:
            print(
                "W&B logging skipped: missing wandb.project or wandb.entity in config."
            )
            return results_item

        try:
            run = wandb.init(
                project=str(wandb_project),
                entity=str(wandb_entity),
                name=run_name,
                job_type="evaluation",
                config={
                    "checkpoint": str(checkpoint_path),
                    "test_split": test_split,
                    "output_path": str(output_path),
                    "train_config": OmegaConf.to_container(train_cfg, resolve=True),
                    "evaluation_config": OmegaConf.to_container(cfg, resolve=True),
                },
                reinit=True,
            )
            wandb.log(results_item)
            for trait_name, trait_result in metric_summary["trait_metrics"].items():
                wandb.log(
                    {
                        f"trait/{trait_name}/rmse": trait_result["rmse"],
                        f"trait/{trait_name}/r2": trait_result["r2"],
                        f"trait/{trait_name}/pearson_r": trait_result["pearson_r"],
                        f"trait/{trait_name}/cov": trait_result["cov"],
                        f"trait/{trait_name}/cov_mean": trait_result["cov_mean"],
                        f"trait/{trait_name}/cov_median": trait_result["cov_median"],
                        f"trait/{trait_name}/cov_p95": trait_result["cov_p95"],
                    }
                )
            run.summary["metrics_file"] = str(output_path)
            run.summary["checkpoint"] = str(checkpoint_path)
            run.finish()
        except Exception as exc:
            print(f"W&B init/log failed for {checkpoint_path}: {exc}")

    return results_item


@hydra.main(
    version_base=None, config_path="../config", config_name="evaluation/default"
)
def main(cfg: DictConfig) -> None:
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
