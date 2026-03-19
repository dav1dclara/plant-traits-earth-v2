"""Metrics package for evaluation utilities.

Focused on the current single-trait workflow while staying reusable for
multi-band outputs.
"""

from .aoa import collect_patch_features, compute_aoa_metrics
from .core import pearson_r, r2_score, r2_score_manual, rmse
from .uncertainty import pixelwise_cov, pixelwise_mean, pixelwise_std, summarize_cov

__all__ = [
    "collect_patch_features",
    "compute_aoa_metrics",
    "pearson_r",
    "r2_score",
    "r2_score_manual",
    "rmse",
    "pixelwise_cov",
    "pixelwise_mean",
    "pixelwise_std",
    "summarize_cov",
]
