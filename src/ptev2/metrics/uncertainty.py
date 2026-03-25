from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def pixelwise_mean(pred_stack: Any, axis: int = 0) -> np.ndarray:
    arr = _to_numpy(pred_stack)
    return np.nanmean(arr, axis=axis)


def pixelwise_std(pred_stack: Any, axis: int = 0, ddof: int = 0) -> np.ndarray:
    arr = _to_numpy(pred_stack)
    return np.nanstd(arr, axis=axis, ddof=ddof)


def pixelwise_cov(pred_stack: Any, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    mean_map = pixelwise_mean(pred_stack, axis=axis)
    std_map = pixelwise_std(pred_stack, axis=axis)
    denom = np.where(np.abs(mean_map) > eps, np.abs(mean_map), np.nan)
    return std_map / denom


def summarize_cov(cov_map: Any, mask: Any | None = None) -> dict[str, float]:
    cov = _to_numpy(cov_map)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        cov = cov[m]
    finite = cov[np.isfinite(cov)]
    if finite.size == 0:
        return {
            "cov_mean": float("nan"),
            "cov_median": float("nan"),
            "cov_p95": float("nan"),
            "cov_n": 0.0,
        }
    return {
        "cov_mean": float(np.mean(finite)),
        "cov_median": float(np.median(finite)),
        "cov_p95": float(np.quantile(finite, 0.95)),
        "cov_n": float(finite.size),
    }
