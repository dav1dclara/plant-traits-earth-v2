from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ptev2.metrics.core import mae, pearson_r, r2_score, rmse


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _validate_shapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_mask: np.ndarray,
    n_traits: int,
    n_bands: int,
) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}."
        )
    if y_true.ndim != 4:
        raise ValueError(f"Expected tensors shaped (N, C, H, W), got {y_true.shape}.")
    expected_channels = n_traits * n_bands
    if y_true.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} target channels for {n_traits} traits x {n_bands} bands, got {y_true.shape[1]}."
        )
    if source_mask.ndim != 4:
        raise ValueError(
            f"Expected source_mask shaped (N, T, H, W), got {source_mask.shape}."
        )
    if (
        source_mask.shape[0] != y_true.shape[0]
        or source_mask.shape[2:] != y_true.shape[2:]
    ):
        raise ValueError(
            "source_mask must match batch and spatial dimensions of y_true/y_pred."
        )
    if source_mask.shape[1] != n_traits:
        raise ValueError(
            f"source_mask has {source_mask.shape[1]} trait channels, but {n_traits} traits were requested."
        )


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float(np.nan)
    return float(np.mean(finite))


def summarize_single_trait_metrics(
    y_true: Any,
    y_pred: Any,
    source_mask: Any,
    trait_names: Sequence[str],
    n_bands: int,
) -> dict[str, Any]:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    source_mask_np = _to_numpy(source_mask)

    n_traits = len(trait_names)
    _validate_shapes(y_true_np, y_pred_np, source_mask_np, n_traits, n_bands)

    source_mask_expanded = np.repeat(source_mask_np, n_bands, axis=1)
    global_valid = (
        np.isfinite(y_true_np) & np.isfinite(y_pred_np) & (source_mask_expanded > 0)
    )

    global_y_true = y_true_np[global_valid]
    global_y_pred = y_pred_np[global_valid]

    if global_y_true.size > 0:
        global_rmse = float(rmse(global_y_true, global_y_pred))
        global_r2 = float(r2_score(global_y_true, global_y_pred))
        global_pearson_r = float(pearson_r(global_y_true, global_y_pred))
        global_mae = float(mae(global_y_true, global_y_pred))
        global_n_valid = int(global_y_true.size)
    else:
        global_rmse = float("nan")
        global_r2 = float("nan")
        global_pearson_r = float("nan")
        global_mae = float("nan")
        global_n_valid = 0

    trait_metrics: dict[str, dict[str, float]] = {}
    for trait_idx, trait_name in enumerate(trait_names):
        start = trait_idx * n_bands
        stop = start + n_bands

        y_true_trait = y_true_np[:, start:stop]
        y_pred_trait = y_pred_np[:, start:stop]
        source_trait = source_mask_np[:, trait_idx : trait_idx + 1]

        valid_trait = (
            np.isfinite(y_true_trait) & np.isfinite(y_pred_trait) & (source_trait > 0)
        )
        if np.any(valid_trait):
            yt_trait = y_true_trait[valid_trait]
            yp_trait = y_pred_trait[valid_trait]
            trait_rmse = float(rmse(yt_trait, yp_trait))
            trait_r2 = float(r2_score(yt_trait, yp_trait))
            trait_pearson_r = float(pearson_r(yt_trait, yp_trait))
            trait_mae = float(mae(yt_trait, yp_trait))
            trait_n_valid = int(yt_trait.size)
        else:
            trait_rmse = float("nan")
            trait_r2 = float("nan")
            trait_pearson_r = float("nan")
            trait_mae = float("nan")
            trait_n_valid = 0

        trait_metrics[str(trait_name)] = {
            "n_valid": trait_n_valid,
            "rmse": trait_rmse,
            "r2": trait_r2,
            "pearson_r": trait_pearson_r,
            "mae": trait_mae,
        }

    macro_rmse = _finite_mean([values["rmse"] for values in trait_metrics.values()])
    macro_r2 = _finite_mean([values["r2"] for values in trait_metrics.values()])
    macro_pearson_r = _finite_mean(
        [values["pearson_r"] for values in trait_metrics.values()]
    )
    macro_mae = _finite_mean([values["mae"] for values in trait_metrics.values()])

    return {
        "rmse": global_rmse,
        "r2": global_r2,
        "pearson_r": global_pearson_r,
        "mae": global_mae,
        "n_valid": global_n_valid,
        "macro_rmse": macro_rmse,
        "macro_r2": macro_r2,
        "macro_pearson_r": macro_pearson_r,
        "macro_mae": macro_mae,
        "trait_metrics": trait_metrics,
    }
