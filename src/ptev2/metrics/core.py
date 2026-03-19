from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mask_finite_pairs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    yt = np.ravel(_to_numpy(y_true))
    yp = np.ravel(_to_numpy(y_pred))
    if yt.shape != yp.shape:
        raise ValueError(
            f"Shape mismatch: y_true.shape={yt.shape}, y_pred.shape={yp.shape}"
        )
    finite = np.isfinite(yt) & np.isfinite(yp)
    return yt[finite], yp[finite]


def pearson_r(y_true: Any, y_pred: Any) -> float:
    yt, yp = _mask_finite_pairs(y_true, y_pred)
    if yt.size < 2:
        return float(np.nan)

    yt_c = yt - yt.mean()
    yp_c = yp - yp.mean()
    denom = np.linalg.norm(yt_c) * np.linalg.norm(yp_c)
    if denom == 0.0:
        return float(np.nan)
    return float(np.dot(yt_c, yp_c) / denom)


def rmse(y_true: Any, y_pred: Any) -> float:
    yt, yp = _mask_finite_pairs(y_true, y_pred)
    if yt.size == 0:
        return float(np.nan)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination computed on finite pairs only."""
    yt, yp = _mask_finite_pairs(y_true, y_pred)
    if yt.size < 2:
        return float(np.nan)

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return float(np.nan)
    return float(1.0 - (ss_res / ss_tot))


def r2_score_manual(y_true: Any, y_pred: Any) -> float:
    """Backward-compatible alias."""
    return r2_score(y_true, y_pred)
