import json
from pathlib import Path

import hydra
import numpy as np
import rasterio
import torch
import zarr
from affine import Affine
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rasterio.crs import CRS
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader

from ptev2.data.dataloader import PlantTraitDataset, get_dataloader
from ptev2.metrics.evaluation import summarize_single_trait_metrics
from ptev2.utils import (
    build_target_layout,
    predict_batch,
    resolve_device,
    resolve_eval_source_value,
    resolve_model_cfg,
    resolve_train_zarr_dir,
    seed_all,
)

console = Console()


def _resolve_validity_mask_cfg(
    train_cfg: DictConfig,
    predictors: list[str],
) -> tuple[bool, list[str]]:
    enabled = bool(
        OmegaConf.select(train_cfg, "data.predictor_validity_masks.enabled") or False
    )
    groups = [
        str(v)
        for v in (
            OmegaConf.select(train_cfg, "data.predictor_validity_masks.groups")
            or predictors
        )
    ]
    return enabled, groups


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_override = OmegaConf.select(cfg, "checkpoint_path")
    if checkpoint_override:
        return Path(str(checkpoint_override))

    run_name = OmegaConf.select(cfg, "run_name")
    if run_name:
        return Path(str(cfg.checkpoint_dir)) / f"{str(run_name)}.pth"

    raise ValueError("Set checkpoint_path or run_name.")


def _load_checkpoint_and_train_cfg(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, DictConfig]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' did not contain a dict state."
        )

    embedded_cfg = state.get("config")
    if embedded_cfg is not None:
        return state, OmegaConf.create(embedded_cfg)

    cfg_path = checkpoint_path.with_suffix(".yaml")
    if not cfg_path.exists():
        raise ValueError(
            "Checkpoint has no embedded config and companion yaml is missing: "
            f"{cfg_path}"
        )
    return state, OmegaConf.load(cfg_path)


def _write_all_prediction_tif(
    preds: np.ndarray,
    all_store: zarr.Group,
    out_tif_path: Path,
    band_names: list[str],
) -> None:
    t = all_store.attrs["transform"]
    geo_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs_epsg = int(all_store.attrs["crs_epsg"])
    pixel_w = float(geo_transform.a)
    pixel_h = float(abs(geo_transform.e))

    bounds = all_store["bounds"][:]  # (N, 4): min_x, min_y, max_x, max_y
    n_chips, out_ch, h_out, w_out = preds.shape

    canvas_origin_x = geo_transform.c
    canvas_origin_y = geo_transform.f
    canvas_rows = int(all_store.attrs["raster_height"])
    canvas_cols = int(all_store.attrs["raster_width"])

    _yw = np.minimum(np.arange(1, h_out + 1), np.arange(h_out, 0, -1)).astype(
        np.float32
    )
    _xw = np.minimum(np.arange(1, w_out + 1), np.arange(w_out, 0, -1)).astype(
        np.float32
    )
    weight_map = np.outer(_yw, _xw) ** 2

    canvas = np.zeros((out_ch, canvas_rows, canvas_cols), dtype=np.float32)
    weight_sum = np.zeros((canvas_rows, canvas_cols), dtype=np.float32)

    for i in range(n_chips):
        min_x, _, _, max_y = bounds[i]
        col = round((min_x - canvas_origin_x) / pixel_w)
        row = round((canvas_origin_y - max_y) / pixel_h)

        r0, r1 = max(row, 0), min(row + h_out, canvas_rows)
        c0, c1 = max(col, 0), min(col + w_out, canvas_cols)
        if r0 >= r1 or c0 >= c1:
            continue

        chip = preds[i, :, r0 - row : r1 - row, c0 - col : c1 - col]
        w = weight_map[r0 - row : r1 - row, c0 - col : c1 - col]
        w = np.where(np.isfinite(chip[0]), w, 0.0)
        canvas[:, r0:r1, c0:c1] += chip * w
        weight_sum[r0:r1, c0:c1] += w

    nonzero = weight_sum > 0
    canvas[:, nonzero] /= weight_sum[nonzero]
    canvas[:, ~nonzero] = np.nan

    out_geo_transform = Affine(
        pixel_w, 0, canvas_origin_x, 0, -pixel_h, canvas_origin_y
    )
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_tif_path,
        "w",
        driver="GTiff",
        height=canvas_rows,
        width=canvas_cols,
        count=out_ch,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=out_geo_transform,
        compress="deflate",
        tiled=True,
        nodata=np.nan,
    ) as dst:
        dst.write(canvas)
        for b_idx, name in enumerate(band_names, start=1):
            dst.set_band_description(b_idx, name)


def _resolve_wandb_resume_id(
    cfg: DictConfig,
    train_cfg: DictConfig,
    checkpoint_state: dict,
    checkpoint_path: Path,
) -> str | None:
    checkpoint_wandb_id = checkpoint_state.get("wandb_run_id")
    if checkpoint_wandb_id:
        return str(checkpoint_wandb_id)

    train_summary_wandb_id = OmegaConf.select(train_cfg, "wandb_run_id")
    if train_summary_wandb_id:
        return str(train_summary_wandb_id)

    run_name = OmegaConf.select(cfg, "run_name") or checkpoint_path.stem
    wandb_root = Path(__file__).resolve().parents[1] / "wandb"
    if not wandb_root.exists():
        return None

    for run_dir in sorted(wandb_root.glob("run-*/"), reverse=True):
        output_log = run_dir / "files" / "output.log"
        if not output_log.exists():
            continue
        try:
            content = output_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if f"Run: {run_name}" in content or f"Syncing run {run_name}" in content:
            return run_dir.name.split("-")[-1]

    return None


@hydra.main(version_base=None, config_path="../config/test", config_name="default")
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TEST + FINAL PREDICTION[/bold cyan]")

    device = resolve_device()
    console.print(f"Device: [cyan]{device}[/cyan]")

    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state, train_cfg = _load_checkpoint_and_train_cfg(checkpoint_path, device)
    seed_all(int(train_cfg.train.seed))

    predictors = [k for k, v in train_cfg.data.predictors.items() if bool(v.use)]
    if not predictors:
        raise ValueError("No predictors enabled in checkpoint training config.")
    add_validity_masks, validity_mask_groups = _resolve_validity_mask_cfg(
        train_cfg, predictors=predictors
    )

    zarr_dir = resolve_train_zarr_dir(train_cfg)
    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")

    target_dataset, traits, selected_bands, target_indices, source_indices = (
        build_target_layout(train_cfg, train_store)
    )
    eval_source_value = resolve_eval_source_value(target_dataset)
    eval_source_name = "GBIF" if eval_source_value == 1 else "sPlot"
    console.print(
        f"Eval source for test metrics/masking: [cyan]{eval_source_name} ({eval_source_value})[/cyan]"
    )

    test_split = str(OmegaConf.select(cfg, "test_split") or "test")
    test_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=test_split,
        predictors=predictors,
        target=target_dataset,
        target_indices=target_indices,
        source_indices=source_indices,
        batch_size=int(train_cfg.data_loaders.batch_size),
        num_workers=int(train_cfg.data_loaders.num_workers),
        add_group_validity_masks=add_validity_masks,
        validity_mask_groups=validity_mask_groups,
    )

    model_cfg = resolve_model_cfg(train_cfg)
    checkpoint_state = state.get("state_dict", state)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Checkpoint state for '{checkpoint_path}' is invalid.")

    total_pred_bands = sum(
        train_store[f"predictors/{name}"].shape[1] for name in predictors
    )
    if add_validity_masks:
        total_pred_bands += sum(1 for g in validity_mask_groups if g in predictors)
    model = instantiate(
        model_cfg,
        in_channels=total_pred_bands,
        out_channels=len(target_indices),
    ).to(device)
    model.load_state_dict(checkpoint_state)
    model.eval()

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_src: list[np.ndarray] = []
    skipped_batches = 0

    with torch.no_grad():
        for X, y, src in track(
            test_loader, description=f"Test [{checkpoint_path.stem}]"
        ):
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            src = src.to(device=device, dtype=torch.float32)

            y_pred, valid_x = predict_batch(model, X)
            y = torch.where(valid_x.expand_as(y), y, torch.nan)
            source_eval_mask = src == eval_source_value
            y = torch.where(source_eval_mask, y, torch.nan)

            valid = torch.isfinite(y_pred) & torch.isfinite(y) & source_eval_mask
            if not bool(valid.any()):
                skipped_batches += 1
                continue

            all_y_pred.append(y_pred.detach().cpu().numpy())
            all_y_true.append(y.detach().cpu().numpy())
            all_src.append(src.detach().cpu().numpy())

    if not all_y_true:
        raise RuntimeError("No valid test observations found.")

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    source_mask = np.concatenate(all_src, axis=0)

    metric_summary = summarize_single_trait_metrics(
        y_true=y_true,
        y_pred=y_pred,
        source_mask=source_mask,
        trait_names=traits,
        n_bands=len(selected_bands),
        valid_source_value=eval_source_value,
    )

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem

    compact_metrics = {
        "checkpoint": str(checkpoint_path),
        "split": test_split,
        "n_valid": int(metric_summary["n_valid"]),
        "skipped_batches": int(skipped_batches),
        "rmse": float(metric_summary["rmse"]),
        "r2": float(metric_summary["r2"]),
        "pearson_r": float(metric_summary["pearson_r"]),
        "mae": float(metric_summary["mae"]),
        "macro_rmse": float(metric_summary["macro_rmse"]),
        "macro_r2": float(metric_summary["macro_r2"]),
        "macro_pearson_r": float(metric_summary["macro_pearson_r"]),
        "macro_mae": float(metric_summary["macro_mae"]),
    }

    compact_path = output_dir / f"{stem}.test_metrics_compact.json"
    compact_path.write_text(json.dumps(compact_metrics, indent=2), encoding="utf-8")

    if bool(cfg.save_full_trait_metrics):
        full_path = output_dir / f"{stem}.test_metrics_full.json"
        full_path.write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")

    if bool(cfg.save_arrays):
        np.save(output_dir / f"{stem}.test_preds.npy", y_pred)
        np.save(output_dir / f"{stem}.test_targets.npy", y_true)
        np.save(output_dir / f"{stem}.test_source_mask.npy", source_mask)

    out_tif_path = None
    write_all_map = bool(OmegaConf.select(cfg, "write_all_map") or False)
    if write_all_map:
        # Final full-domain prediction (all.zarr -> GeoTIFF)
        all_zarr_path = zarr_dir / "all.zarr"
        if not all_zarr_path.exists():
            raise FileNotFoundError(f"all.zarr not found: {all_zarr_path}")
        all_store = zarr.open_group(str(all_zarr_path), mode="r")
        all_dataset = PlantTraitDataset(
            all_zarr_path,
            predictors=predictors,
            target=target_dataset,
            target_indices=target_indices,
            source_indices=source_indices,
            add_group_validity_masks=add_validity_masks,
            validity_mask_groups=validity_mask_groups,
        )
        all_loader = DataLoader(
            all_dataset,
            batch_size=int(train_cfg.data_loaders.batch_size),
            shuffle=False,
            num_workers=int(train_cfg.data_loaders.num_workers),
        )

        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for X, _, _ in track(
                all_loader, description=f"All-map [{checkpoint_path.stem}]"
            ):
                X = X.to(device=device, dtype=torch.float32)
                y_pred_map, valid_x = predict_batch(model, X)
                y_pred_map = torch.where(
                    valid_x.expand_as(y_pred_map),
                    y_pred_map,
                    torch.full_like(y_pred_map, float("nan")),
                )
                all_preds.append(y_pred_map.detach().cpu().numpy())

        preds_all = np.concatenate(all_preds, axis=0)
        prediction_out_dir = Path(
            str(OmegaConf.select(cfg, "prediction_out_dir") or output_dir)
        )
        prediction_filename = str(
            OmegaConf.select(cfg, "prediction_tif_name") or f"{stem}.tif"
        )
        out_tif_path = prediction_out_dir / prediction_filename
        band_names = [f"{trait}_{band}" for trait in traits for band in selected_bands]
        _write_all_prediction_tif(preds_all, all_store, out_tif_path, band_names)

    console.print(f"checkpoint: [cyan]{checkpoint_path}[/cyan]")
    console.print(f"split:      [cyan]{test_split}[/cyan]")
    console.print(f"n_valid:    [cyan]{compact_metrics['n_valid']}[/cyan]")
    console.print(f"macro_rmse: [cyan]{compact_metrics['macro_rmse']:.6f}[/cyan]")
    console.print(f"macro_r2:   [cyan]{compact_metrics['macro_r2']:.6f}[/cyan]")
    console.print(f"macro_r:    [cyan]{compact_metrics['macro_pearson_r']:.6f}[/cyan]")
    console.print(f"macro_mae:  [cyan]{compact_metrics['macro_mae']:.6f}[/cyan]")
    console.print(f"metrics:    [cyan]{compact_path}[/cyan]")
    if out_tif_path is not None:
        console.print(f"all-map tif:[cyan]{out_tif_path}[/cyan]")
    else:
        console.print("[yellow]all-map tif skipped (write_all_map=false)[/yellow]")

    wandb_enabled_cfg = OmegaConf.select(cfg, "wandb.enabled")
    wandb_enabled_train = OmegaConf.select(train_cfg, "wandb.enabled")
    wandb_enabled = bool(
        wandb_enabled_cfg
        if wandb_enabled_cfg is not None
        else (wandb_enabled_train if wandb_enabled_train is not None else False)
    )

    if wandb_enabled:
        import wandb

        wandb_project = OmegaConf.select(cfg, "wandb.project") or OmegaConf.select(
            train_cfg, "wandb.project"
        )
        wandb_entity = OmegaConf.select(cfg, "wandb.entity") or OmegaConf.select(
            train_cfg, "wandb.entity"
        )
        if not wandb_project or not wandb_entity:
            console.print("[yellow]W&B skipped: missing entity/project.[/yellow]")
            return

        run_name = OmegaConf.select(cfg, "run_name") or checkpoint_path.stem
        resume_id = _resolve_wandb_resume_id(cfg, train_cfg, state, checkpoint_path)

        if resume_id:
            console.print(
                f"[cyan]Appending test metrics to W&B run id={resume_id}[/cyan]"
            )
            run = wandb.init(
                project=str(wandb_project),
                entity=str(wandb_entity),
                id=str(resume_id),
                resume="allow",
                job_type="test",
                config={
                    "checkpoint": str(checkpoint_path),
                    "test_split": test_split,
                    "output_dir": str(output_dir),
                },
                reinit="finish_previous",
            )
        else:
            console.print(
                "[yellow]Could not resolve matching W&B run; creating a new test run.[/yellow]"
            )
            run = wandb.init(
                project=str(wandb_project),
                entity=str(wandb_entity),
                name=str(run_name),
                job_type="test",
                config={
                    "checkpoint": str(checkpoint_path),
                    "test_split": test_split,
                    "output_dir": str(output_dir),
                },
                reinit="finish_previous",
            )

        log_dict = {
            "test/macro/rmse": compact_metrics["macro_rmse"],
            "test/macro/r2": compact_metrics["macro_r2"],
            "test/macro/pearson_r": compact_metrics["macro_pearson_r"],
            "test/macro/mae": compact_metrics["macro_mae"],
            "test/global/rmse": compact_metrics["rmse"],
            "test/global/r2": compact_metrics["r2"],
            "test/global/pearson_r": compact_metrics["pearson_r"],
            "test/global/mae": compact_metrics["mae"],
            "test/n_valid": compact_metrics["n_valid"],
            "test/skipped_batches": compact_metrics["skipped_batches"],
        }
        wandb.log(log_dict)

        if bool(OmegaConf.select(cfg, "wandb.log_trait_metrics")):
            trait_metrics = metric_summary.get("trait_metrics", {})
            for trait_name, trait_result in trait_metrics.items():
                wandb.log(
                    {
                        f"test/trait/{trait_name}/rmse": trait_result["rmse"],
                        f"test/trait/{trait_name}/r2": trait_result["r2"],
                        f"test/trait/{trait_name}/pearson_r": trait_result["pearson_r"],
                        f"test/trait/{trait_name}/mae": trait_result["mae"],
                    }
                )

        run.summary["checkpoint"] = str(checkpoint_path)
        run.summary["test_metrics_compact"] = str(compact_path)
        if out_tif_path is not None:
            run.summary["all_prediction_tif"] = str(out_tif_path)
        run.finish()


if __name__ == "__main__":
    main()
