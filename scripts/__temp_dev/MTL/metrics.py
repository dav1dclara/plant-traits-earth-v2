"""
Metrics computation for evaluation.
"""

from __future__ import annotations

import numpy as np


def _safe_nanmean(values: np.ndarray) -> float:
    """Compute nanmean without warnings for all-NaN arrays."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def _pearson_r(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute Pearson correlation coefficient, handling NaN."""
    mask = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)
    if mask.sum() < 3:
        return float("nan")
    a_m, b_m = a[mask], b[mask]
    if a_m.std() < 1e-8 or b_m.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a_m, b_m)[0, 1])


def _rmse(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute RMSE, handling NaN."""
    mask = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _r2_score(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute coefficient of determination, handling NaN."""
    mask = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        mask &= valid_mask.astype(bool)
    if mask.sum() < 3:
        return float("nan")

    a_m, b_m = a[mask], b[mask]
    ss_res = np.sum((a_m - b_m) ** 2)
    ss_tot = np.sum((a_m - np.mean(a_m)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def compute_metrics(
    preds: np.ndarray,  # (N, T, H, W)
    targets: np.ndarray,  # (N, T, H, W)
    source_mask: np.ndarray | None = None,  # (N, T, H, W)
    eval_source: str = "all",
) -> dict:
    """
    Compute evaluation metrics.

    Args:
        preds: Predictions (N, T, H, W)
        targets: Targets (N, T, H, W)

    Returns:
        Dictionary with per-trait and overall metrics
    """
    assert preds.shape == targets.shape
    n_samples, n_traits, h, w = preds.shape

    if source_mask is not None:
        assert source_mask.shape == targets.shape

    # Flatten to (N*H*W, T) with traits kept as columns.
    preds_flat = preds.transpose(0, 2, 3, 1).reshape(-1, n_traits)
    targets_flat = targets.transpose(0, 2, 3, 1).reshape(-1, n_traits)
    source_mask_flat = (
        None
        if source_mask is None
        else source_mask.transpose(0, 2, 3, 1).reshape(-1, n_traits)
    )

    if source_mask_flat is None or eval_source == "all":
        eval_mask_flat = None if source_mask_flat is None else source_mask_flat > 0
    elif eval_source == "splot":
        eval_mask_flat = source_mask_flat == 2
    elif eval_source == "gbif":
        eval_mask_flat = source_mask_flat == 1
    else:
        raise ValueError(f"Unknown eval_source: {eval_source}")

    # Per-trait metrics
    per_trait_r = []
    per_trait_r2 = []
    per_trait_rmse = []
    for t in range(n_traits):
        trait_mask = None if eval_mask_flat is None else eval_mask_flat[:, t]
        r = _pearson_r(targets_flat[:, t], preds_flat[:, t], valid_mask=trait_mask)
        r2 = _r2_score(targets_flat[:, t], preds_flat[:, t], valid_mask=trait_mask)
        rmse = _rmse(targets_flat[:, t], preds_flat[:, t], valid_mask=trait_mask)
        per_trait_r.append(r)
        per_trait_r2.append(r2)
        per_trait_rmse.append(rmse)

    # Overall metrics
    overall_mask = None if eval_mask_flat is None else eval_mask_flat.ravel()
    r_all = _pearson_r(
        targets_flat.ravel(), preds_flat.ravel(), valid_mask=overall_mask
    )
    r2_all = _r2_score(
        targets_flat.ravel(), preds_flat.ravel(), valid_mask=overall_mask
    )
    rmse_all = _rmse(targets_flat.ravel(), preds_flat.ravel(), valid_mask=overall_mask)

    # Valid trait metrics (NaN-safe mean)
    per_trait_r = np.array(per_trait_r)
    per_trait_r2 = np.array(per_trait_r2)
    per_trait_rmse = np.array(per_trait_rmse)
    r_mean = _safe_nanmean(per_trait_r)
    r2_mean = _safe_nanmean(per_trait_r2)
    rmse_mean = _safe_nanmean(per_trait_rmse)
    n_eval_pixels = (
        int(np.sum(eval_mask_flat))
        if eval_mask_flat is not None
        else int(np.sum(np.isfinite(targets_flat) & np.isfinite(preds_flat)))
    )

    return {
        "eval_source": eval_source,
        "pearson_r_all": float(r_all),
        "pearson_r_mean": float(r_mean),
        "per_trait_r": per_trait_r.tolist(),
        "r2_all": float(r2_all),
        "r2_mean": float(r2_mean),
        "per_trait_r2": per_trait_r2.tolist(),
        "rmse_all": float(rmse_all),
        "rmse_mean": float(rmse_mean),
        "per_trait_rmse": per_trait_rmse.tolist(),
        "n_eval_pixels": n_eval_pixels,
        "n_valid_pixels": n_eval_pixels,
        "n_samples": n_samples,
        "n_traits": n_traits,
    }
