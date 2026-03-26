from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as sk_r2_score


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
    corr = np.corrcoef(yt, yp)[0, 1]
    if not np.isfinite(corr):
        return float(np.nan)
    return float(corr)


def rmse(y_true: Any, y_pred: Any) -> float:
    yt, yp = _mask_finite_pairs(y_true, y_pred)
    if yt.size == 0:
        return float(np.nan)
    return float(mean_squared_error(yt, yp, squared=False))


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination computed on finite pairs only."""
    yt, yp = _mask_finite_pairs(y_true, y_pred)
    if yt.size < 2:
        return float(np.nan)
    score = sk_r2_score(yt, yp)
    if not np.isfinite(score):
        return float(np.nan)
    return float(score)


def r2_score_manual(y_true: Any, y_pred: Any) -> float:
    """Backward-compatible alias."""
    return r2_score(y_true, y_pred)
