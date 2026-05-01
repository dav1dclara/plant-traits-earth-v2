import csv
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
    apply_supervision_bundle,
    apply_supervision_tensor,
    assert_supervision_shape,
    build_target_layout,
    mask_bundle_targets_with_validity,
    move_bundle_to_device,
    predict_batch,
    resolve_device,
    resolve_model_cfg,
    resolve_predictor_validity_config,
    resolve_supervision_config,
    resolve_train_zarr_dir,
    seed_all,
)

console = Console()


def _to_python_scalars(obj):
    if isinstance(obj, dict):
        return {k: _to_python_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_scalars(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


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

    bounds = all_store["bounds"][:]
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
    weight_map = np.outer(_yw, _xw).astype(np.float32) ** 2

    canvas = np.zeros((out_ch, canvas_rows, canvas_cols), dtype=np.float32)
    weight_sum = np.zeros((out_ch, canvas_rows, canvas_cols), dtype=np.float32)

    for i in range(n_chips):
        min_x, _, _, max_y = bounds[i]
        col = round((min_x - canvas_origin_x) / pixel_w)
        row = round((canvas_origin_y - max_y) / pixel_h)

        r0, r1 = max(row, 0), min(row + h_out, canvas_rows)
        c0, c1 = max(col, 0), min(col + w_out, canvas_cols)
        if r0 >= r1 or c0 >= c1:
            continue

        chip = preds[i, :, r0 - row : r1 - row, c0 - col : c1 - col]
        finite = np.isfinite(chip)
        w = weight_map[None, r0 - row : r1 - row, c0 - col : c1 - col] * finite
        canvas[:, r0:r1, c0:c1] += (
            np.nan_to_num(chip, nan=0.0, posinf=0.0, neginf=0.0) * w
        )
        weight_sum[:, r0:r1, c0:c1] += w

    output = np.full_like(canvas, np.nan)
    nonzero = weight_sum > 0
    output[nonzero] = canvas[nonzero] / weight_sum[nonzero]

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
        dst.write(output)
        for b_idx, name in enumerate(band_names, start=1):
            dst.set_band_description(b_idx, name)


def _prepare_batch(
    model: torch.nn.Module,
    X: torch.Tensor,
    bundle: dict[str, dict[str, torch.Tensor]] | None,
    *,
    device: torch.device,
    supervision_mode: str,
    center_crop_size: int,
    predictor_validity_mode: str,
    predictor_min_finite_ratio: float,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]] | None]:
    X = X.to(device=device, dtype=torch.float32)
    if bundle is not None:
        bundle = move_bundle_to_device(bundle, device)
    y_pred, valid_x = predict_batch(
        model,
        X,
        validity_mode=predictor_validity_mode,
        min_finite_ratio=predictor_min_finite_ratio,
    )
    y_pred = apply_supervision_tensor(y_pred, supervision_mode, center_crop_size)
    valid_x = apply_supervision_tensor(valid_x, supervision_mode, center_crop_size)
    assert_supervision_shape(
        y_pred,
        mode=supervision_mode,
        center_crop_size=center_crop_size,
        context=f"{context}/y_pred",
    )
    assert_supervision_shape(
        valid_x,
        mode=supervision_mode,
        center_crop_size=center_crop_size,
        context=f"{context}/valid_x",
    )
    if bundle is not None:
        bundle = apply_supervision_bundle(bundle, supervision_mode, center_crop_size)
        bundle = mask_bundle_targets_with_validity(bundle, valid_x)
        for dataset_name, payload in bundle.items():
            assert_supervision_shape(
                payload["y"],
                mode=supervision_mode,
                center_crop_size=center_crop_size,
                context=f"{context}/{dataset_name}/y",
            )
            assert_supervision_shape(
                payload["source_mask"],
                mode=supervision_mode,
                center_crop_size=center_crop_size,
                context=f"{context}/{dataset_name}/source_mask",
            )
    return y_pred, valid_x, bundle


def _update_source_counts(
    *,
    source_counts_per_trait: dict[str, dict[str, int]],
    source_key: str,
    payload: dict[str, torch.Tensor],
    y_pred: torch.Tensor,
    traits: list[str],
    n_bands: int,
) -> int:
    valid = (
        torch.isfinite(payload["y"])
        & torch.isfinite(y_pred)
        & (payload["source_mask"] > 0)
    )
    total = 0
    for trait_idx, trait_name in enumerate(traits):
        start = trait_idx * n_bands
        stop = start + n_bands
        count = int(valid[:, start:stop, :, :].sum().item())
        source_counts_per_trait[trait_name][source_key] += count
        total += count
    return total


@hydra.main(
    version_base=None, config_path="../../config/22km/test", config_name="default"
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TEST + FINAL PREDICTION[/bold cyan]")

    device = resolve_device()
    console.print(f"Device: [cyan]{device}[/cyan]")

    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state, train_cfg = _load_checkpoint_and_train_cfg(checkpoint_path, device)
    seed_all(int(train_cfg.train.seed))

    zarr_dir_override = OmegaConf.select(cfg, "zarr_dir")
    if zarr_dir_override:
        if OmegaConf.select(train_cfg, "data") is None:
            train_cfg.data = {}
        train_cfg.data.zarr_dir = str(zarr_dir_override)
        console.print(
            f"[yellow]Overriding checkpoint zarr_dir with test config:[/yellow] "
            f"[cyan]{train_cfg.data.zarr_dir}[/cyan]"
        )

    predictors = [k for k, v in train_cfg.data.predictors.items() if bool(v.use)]
    if not predictors:
        raise ValueError("No predictors enabled in checkpoint training config.")

    zarr_dir = resolve_train_zarr_dir(train_cfg)
    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")
    target_layout = build_target_layout(train_cfg, train_store)

    eval_dataset_ckpt = str(target_layout["eval_dataset"])
    eval_dataset_override = OmegaConf.select(cfg, "eval_dataset")
    if (
        eval_dataset_override is not None
        and str(eval_dataset_override) != eval_dataset_ckpt
    ):
        raise ValueError(
            "Test eval_dataset override is not allowed. "
            f"Checkpoint expects '{eval_dataset_ckpt}', got '{eval_dataset_override}'."
        )
    eval_dataset = eval_dataset_ckpt

    test_split = str(OmegaConf.select(cfg, "test_split") or "test")
    test_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=test_split,
        predictors=predictors,
        target_layouts=target_layout["layouts"],
        return_target_bundle=True,
        batch_size=int(train_cfg.data_loaders.batch_size),
        num_workers=int(train_cfg.data_loaders.num_workers),
    )

    model_cfg = resolve_model_cfg(train_cfg)
    checkpoint_state = state.get("state_dict", state)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Checkpoint state for '{checkpoint_path}' is invalid.")

    total_pred_bands = sum(
        train_store[f"predictors/{name}"].shape[1] for name in predictors
    )
    traits = list(target_layout["traits"])
    bands = list(target_layout["bands"])
    out_channels = len(traits) * len(bands)

    model = instantiate(
        model_cfg,
        in_channels=total_pred_bands,
        out_channels=out_channels,
    ).to(device)
    model.load_state_dict(checkpoint_state)
    model.eval()

    supervision_mode, center_crop_size = resolve_supervision_config(train_cfg)
    predictor_validity_mode, predictor_min_finite_ratio = (
        resolve_predictor_validity_config(train_cfg)
    )

    cfg_supervision_mode = OmegaConf.select(cfg, "train.supervision.mode")
    if (
        cfg_supervision_mode is not None
        and str(cfg_supervision_mode) != supervision_mode
    ):
        raise AssertionError(
            "train.supervision.mode must match the checkpoint training config during test."
        )
    cfg_center_crop_size = OmegaConf.select(cfg, "train.supervision.center_crop_size")
    if (
        cfg_center_crop_size is not None
        and int(cfg_center_crop_size) != center_crop_size
    ):
        raise AssertionError(
            "train.supervision.center_crop_size must match the checkpoint training config during test."
        )

    if supervision_mode == "dense":
        console.print(
            "[yellow]Dense supervision with overlapping chips can duplicate target pixels in "
            "loss/metrics. Use only as an ablation unless unique-pixel de-duplication is implemented.[/yellow]"
        )
    if supervision_mode == "center_crop":
        console.print(
            "[yellow]Center-crop supervision uses only the central crop for loss/metrics; "
            "outer patch pixels provide context only.[/yellow]"
        )

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    skipped_batches = 0

    source_counts_per_trait: dict[str, dict[str, int]] = {
        trait: {"n_valid_splot": 0, "n_valid_gbif": 0} for trait in traits
    }
    n_valid_splot = 0
    n_valid_gbif = 0

    with torch.no_grad():
        for X, bundle in track(
            test_loader, description=f"Test [{checkpoint_path.stem}]"
        ):
            y_pred, _, bundle = _prepare_batch(
                model,
                X,
                bundle,
                device=device,
                supervision_mode=supervision_mode,
                center_crop_size=center_crop_size,
                predictor_validity_mode=predictor_validity_mode,
                predictor_min_finite_ratio=predictor_min_finite_ratio,
                context="test",
            )
            if bundle is None:
                skipped_batches += 1
                continue

            splot_payload = bundle.get("splot")
            if splot_payload is not None:
                n_valid_splot += _update_source_counts(
                    source_counts_per_trait=source_counts_per_trait,
                    source_key="n_valid_splot",
                    payload=splot_payload,
                    y_pred=y_pred,
                    traits=traits,
                    n_bands=len(bands),
                )

            gbif_payload = bundle.get("gbif")
            if gbif_payload is not None:
                n_valid_gbif += _update_source_counts(
                    source_counts_per_trait=source_counts_per_trait,
                    source_key="n_valid_gbif",
                    payload=gbif_payload,
                    y_pred=y_pred,
                    traits=traits,
                    n_bands=len(bands),
                )

            eval_payload = bundle.get(eval_dataset)
            if eval_payload is None:
                skipped_batches += 1
                continue

            y_eval = eval_payload["y"]
            src_eval = eval_payload["source_mask"]
            valid_eval = (
                torch.isfinite(y_eval) & torch.isfinite(y_pred) & (src_eval > 0)
            )

            if not bool(valid_eval.any()):
                skipped_batches += 1
                continue

            all_y_pred.append(y_pred.detach().cpu().numpy())
            all_y_true.append(y_eval.detach().cpu().numpy())
            all_valid.append(valid_eval.detach().cpu().numpy())

    if not all_y_true:
        raise RuntimeError("No valid test observations found.")

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    valid_mask = np.concatenate(all_valid, axis=0)

    metric_summary = summarize_single_trait_metrics(
        y_true=y_true,
        y_pred=y_pred,
        source_mask=None,
        valid_mask=valid_mask,
        trait_names=traits,
        n_bands=len(bands),
    )

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem

    if supervision_mode == "center_pixel":
        count_semantics = "Counts refer to supervised center-cell observations."
    elif supervision_mode == "center_crop":
        count_semantics = "Counts refer to supervised center-crop pixel observations."
    else:
        count_semantics = "Counts refer to dense supervised pixel observations; overlapping chips can duplicate targets."

    compact_metrics = {
        "checkpoint": str(checkpoint_path),
        "split": test_split,
        "mode": str(target_layout["mode"]),
        "target_mode": str(target_layout["mode"]),
        "eval_dataset": eval_dataset,
        "supervision_mode": supervision_mode,
        "center_crop_size": int(center_crop_size),
        "predictor_validity_mode": predictor_validity_mode,
        "predictor_min_finite_ratio": float(predictor_min_finite_ratio),
        "count_semantics": count_semantics,
        "n_valid": int(metric_summary["n_valid"]),
        "n_valid_splot": int(n_valid_splot),
        "n_valid_gbif": int(n_valid_gbif),
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
    full_metrics = {
        **compact_metrics,
        "trait_metrics": _to_python_scalars(metric_summary["trait_metrics"]),
        "trait_source_counts": _to_python_scalars(source_counts_per_trait),
    }
    full_path = output_dir / f"{stem}.test_metrics_full.json"
    full_path.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")

    diagnostics_dir = Path(str(OmegaConf.select(cfg, "diagnostics_dir") or output_dir))
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    per_trait_csv = diagnostics_dir / "per_trait_test_metrics.csv"
    if not per_trait_csv.exists():
        with per_trait_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_name",
                    "split",
                    "mode",
                    "eval_dataset",
                    "trait",
                    "n_valid",
                    "n_valid_splot",
                    "n_valid_gbif",
                    "rmse",
                    "r2",
                    "pearson_r",
                    "mae",
                ]
            )
    with per_trait_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for trait_name, metrics in metric_summary["trait_metrics"].items():
            writer.writerow(
                [
                    stem,
                    test_split,
                    str(target_layout["mode"]),
                    eval_dataset,
                    trait_name,
                    metrics["n_valid"],
                    source_counts_per_trait[trait_name]["n_valid_splot"],
                    source_counts_per_trait[trait_name]["n_valid_gbif"],
                    metrics["rmse"],
                    metrics["r2"],
                    metrics["pearson_r"],
                    metrics["mae"],
                ]
            )

    out_tif_path = None
    write_all_map_cfg = OmegaConf.select(cfg, "write_all_map")
    write_all_map = True if write_all_map_cfg is None else bool(write_all_map_cfg)
    if write_all_map:
        all_zarr_path = zarr_dir / "all.zarr"
        if not all_zarr_path.exists():
            raise FileNotFoundError(f"all.zarr not found: {all_zarr_path}")
        all_store = zarr.open_group(str(all_zarr_path), mode="r")

        all_dataset = PlantTraitDataset(
            all_zarr_path,
            predictors=predictors,
            target_layouts=target_layout["layouts"],
            return_target_bundle=True,
        )
        all_loader = DataLoader(
            all_dataset,
            batch_size=int(train_cfg.data_loaders.batch_size),
            shuffle=False,
            num_workers=int(train_cfg.data_loaders.num_workers),
        )

        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for X, _ in track(
                all_loader, description=f"All-map [{checkpoint_path.stem}]"
            ):
                y_pred_map, valid_x, _ = _prepare_batch(
                    model,
                    X,
                    None,
                    device=device,
                    supervision_mode=supervision_mode,
                    center_crop_size=center_crop_size,
                    predictor_validity_mode=predictor_validity_mode,
                    predictor_min_finite_ratio=predictor_min_finite_ratio,
                    context="all_map",
                )
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
        band_names = [f"{trait}_{band}" for trait in traits for band in bands]
        _write_all_prediction_tif(preds_all, all_store, out_tif_path, band_names)

    console.print(f"checkpoint: [cyan]{checkpoint_path}[/cyan]")
    console.print(f"split:      [cyan]{test_split}[/cyan]")
    console.print(f"mode:       [cyan]{target_layout['mode']}[/cyan]")
    console.print(f"eval_data:  [cyan]{eval_dataset}[/cyan]")
    console.print(f"n_valid:    [cyan]{compact_metrics['n_valid']}[/cyan]")
    console.print(f"n_valid_splot: [cyan]{compact_metrics['n_valid_splot']}[/cyan]")
    console.print(f"n_valid_gbif:  [cyan]{compact_metrics['n_valid_gbif']}[/cyan]")
    console.print(f"macro_rmse: [cyan]{compact_metrics['macro_rmse']:.6f}[/cyan]")
    console.print(f"macro_r2:   [cyan]{compact_metrics['macro_r2']:.6f}[/cyan]")
    console.print(f"macro_r:    [cyan]{compact_metrics['macro_pearson_r']:.6f}[/cyan]")
    console.print(f"macro_mae:  [cyan]{compact_metrics['macro_mae']:.6f}[/cyan]")
    console.print(f"metrics:    [cyan]{compact_path}[/cyan]")
    console.print(f"metrics full:[cyan]{full_path}[/cyan]")
    console.print(f"per-trait:  [cyan]{per_trait_csv}[/cyan]")
    if out_tif_path is not None:
        console.print(f"all-map tif:[cyan]{out_tif_path}[/cyan]")
    else:
        console.print("all-map tif:[yellow]skipped (write_all_map=false)[/yellow]")


if __name__ == "__main__":
    main()
