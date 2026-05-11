"""
Power Transform and Standardization utilities for trait predictions and targets.

The training pipeline applies a two-stage normalization:
    raw trait values  →  Yeo-Johnson(λ)  →  z-score(mean, scale)  →  normalized values

This module provides functions to:
1. Transform: raw values → normalized (for preprocessing)
2. Inverse transform: normalized values → raw original units (for denormalization)

Inverse transformation is essential for interpreting metrics (RMSE, MAE, std) in original units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox1p


def inverse_yeo_johnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Inverse Yeo-Johnson transformation.

    Args:
        y: Transformed values (positive or negative branch).
        lmbda: Yeo-Johnson lambda parameter.

    Returns:
        Original (untransformed) values.

    References:
        Yeo, I. K., & Johnson, R. A. (2000).
        A new family of power transformations to improve normality or symmetry.
        Biometrika, 87(4), 954-959.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.empty_like(y)

    pos = y >= 0
    neg = ~pos

    if pos.any():
        x[pos] = inv_boxcox1p(y[pos], lmbda)
    if neg.any():
        x[neg] = -inv_boxcox1p(-y[neg], 2.0 - lmbda)

    return x


def clip_yeo_johnson_to_inverse_domain(
    y: np.ndarray,
    lmbda: float,
    eps: float = 1e-6,
    margin_ratio: float = 0.0,
    margin_abs: float = 0.0,
) -> np.ndarray:
    """
    Clip Yeo-Johnson-space values so inverse transform stays in-domain.

    This guards against invalid `inv_boxcox1p` inputs near branch boundaries.
    It does not change values that are already comfortably in-domain.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if margin_ratio < 0:
        raise ValueError(f"margin_ratio must be >= 0, got {margin_ratio}")
    if margin_abs < 0:
        raise ValueError(f"margin_abs must be >= 0, got {margin_abs}")

    y = np.asarray(y, dtype=np.float64).copy()
    finite = np.isfinite(y)

    # Positive branch uses inv_boxcox1p(y, lmbda). For lmbda < 0 the argument
    # must satisfy y < -1/lmbda.
    if lmbda < 0:
        boundary = -1.0 / lmbda
        margin = max(eps, margin_abs, abs(boundary) * margin_ratio)
        upper = boundary - margin
        pos_mask = finite & (y >= 0)
        y[pos_mask] = np.minimum(y[pos_mask], upper)

    # Negative branch uses -inv_boxcox1p(-y, 2-lmbda). If (2-lmbda) < 0 then
    # -y must satisfy -y < -1/(2-lmbda), i.e. y > 1/(2-lmbda).
    lmbda_neg = 2.0 - lmbda
    if lmbda_neg < 0:
        boundary = 1.0 / lmbda_neg
        margin = max(eps, margin_abs, abs(boundary) * margin_ratio)
        lower = boundary + margin
        neg_mask = finite & (y < 0)
        y[neg_mask] = np.maximum(y[neg_mask], lower)

    return y


def denormalize_predictions(
    normalized_values: np.ndarray,
    trait_id: str,
    params_df: pd.DataFrame,
    *,
    clip_z_abs: float | None = None,
    clip_to_inverse_domain: bool = True,
    domain_eps: float = 1e-6,
    domain_margin_ratio: float = 0.0,
    domain_margin_abs: float = 0.0,
    max_abs_output: float | None = None,
) -> np.ndarray:
    """
    Denormalize trait predictions from normalized space back to original units.

    Pipeline:
        normalized values  →  destandardize (y * scale + mean)  →  inverse Yeo-Johnson  →  original values

    Args:
        normalized_values: Model output (shape: any). NaN values are preserved.
        trait_id: Trait identifier (e.g., 'X3117').
        params_df: DataFrame with columns [yeo_johnson_lambda, standardize_mean, standardize_scale].
                   Indexed by trait_id.
        clip_z_abs: Optional absolute clipping in normalized z-space before
            de-standardization. Helps suppress extreme outliers from dominating
            inverse-transform outputs. Example: 3.0.
        clip_to_inverse_domain: If True, clip Yeo-Johnson-space values to stay
            safely inside the inverse domain for the trait lambda.
        domain_eps: Small positive safety margin used for domain clipping.
        domain_margin_ratio: Additional relative margin away from inverse
            boundary. Useful to avoid huge amplification near singularities.
        domain_margin_abs: Additional absolute margin away from inverse boundary.
        max_abs_output: Optional absolute clipping in original units after
            inverse transform (final safety cap).

    Returns:
        Denormalized values in original units (same shape as input).

    Raises:
        KeyError: If trait_id not found in params_df.
        ValueError: If params_df missing required columns.
    """
    if trait_id not in params_df.index:
        raise KeyError(
            f"Trait '{trait_id}' not found in transformation parameters. "
            f"Available: {list(params_df.index)}"
        )

    required_cols = {"yeo_johnson_lambda", "standardize_mean", "standardize_scale"}
    missing = required_cols - set(params_df.columns)
    if missing:
        raise ValueError(f"Parameter DataFrame missing required columns: {missing}")

    row = params_df.loc[trait_id]
    lmbda = float(row["yeo_johnson_lambda"])
    mean_ = float(row["standardize_mean"])
    scale_ = float(row["standardize_scale"])

    normalized = np.asarray(normalized_values, dtype=np.float64).copy()
    invalid_mask = ~np.isfinite(normalized)

    if clip_z_abs is not None:
        clip_z_abs = float(clip_z_abs)
        if clip_z_abs <= 0:
            raise ValueError(f"clip_z_abs must be > 0 when provided, got {clip_z_abs}.")
        np.clip(normalized, -clip_z_abs, clip_z_abs, out=normalized)

    # Step 1: Destandardize
    y = normalized * scale_ + mean_

    if clip_to_inverse_domain:
        y = clip_yeo_johnson_to_inverse_domain(
            y,
            lmbda,
            eps=domain_eps,
            margin_ratio=domain_margin_ratio,
            margin_abs=domain_margin_abs,
        )

    # Step 2: Inverse Yeo-Johnson
    x = inverse_yeo_johnson(y, lmbda)

    if max_abs_output is not None:
        max_abs_output = float(max_abs_output)
        if max_abs_output <= 0:
            raise ValueError(
                f"max_abs_output must be > 0 when provided, got {max_abs_output}."
            )
        np.clip(x, -max_abs_output, max_abs_output, out=x)

    # Preserve original invalid mask and avoid propagating inf values.
    x[invalid_mask] = np.nan
    x[~np.isfinite(x)] = np.nan

    return x.astype(np.float32)


def load_power_transformer_params(
    csv_path: str
    | Path = "/scratch3/plant-traits-v2/data/power_transformer_params.csv",
) -> pd.DataFrame:
    """
    Load power transformer parameters from CSV.

    Args:
        csv_path: Path to power_transformer_params.csv (default: project root).

    Returns:
        DataFrame indexed by trait_id with columns:
            - yeo_johnson_lambda
            - standardize_mean
            - standardize_scale
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Power transformer CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "trait" not in df.columns:
        raise ValueError(
            f"Power transformer CSV must contain 'trait' column. "
            f"Found columns: {list(df.columns)}"
        )

    required_cols = {
        "trait",
        "yeo_johnson_lambda",
        "standardize_mean",
        "standardize_scale",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Power transformer CSV missing required columns: {missing}")

    df["trait"] = df["trait"].astype(str).str.replace(r"^X", "", regex=True)
    df = df.set_index("trait")

    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(
            "Power transformer CSV contains duplicate normalized trait IDs: "
            f"{duplicates}"
        )

    return df
