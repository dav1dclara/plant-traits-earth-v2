from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from scipy.special import inv_boxcox1p
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def as_list(value) -> list:
    """Normalize optional scalar or sequence values to a Python list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def get_scale_offset(filepath: str | Path) -> tuple[float | None, float]:
    """Read scale_factor and add_offset from raster tags if present."""
    filepath = Path(filepath)
    with rasterio.open(filepath) as src:
        tags = {**src.tags(), **src.tags(1)}

    def get_tag(keys: Iterable[str], default=None):
        for key in keys:
            for tag_key, tag_value in tags.items():
                if tag_key.lower() == key.lower():
                    try:
                        return float(tag_value)
                    except (TypeError, ValueError):
                        return tag_value
        return default

    scale = get_tag(["scale_factor", "scale"])
    offset = get_tag(["add_offset", "offset"], default=0.0)
    return scale, float(offset or 0.0)


def read_raster(filepath: str | Path) -> tuple[np.ndarray, dict, float | None]:
    """Read a single-band raster as float64 and return data, profile, and nodata."""
    filepath = Path(filepath)
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata
    return data, profile, nodata


def mask_nodata(
    data: np.ndarray,
    nodata: float | None,
    extra_nodata=None,
) -> np.ndarray:
    """Apply nodata masking and convert invalid entries to NaN."""
    arr = np.asarray(data, dtype=np.float64).copy()

    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            arr[np.isnan(arr)] = np.nan
        else:
            arr[np.isclose(arr, nodata)] = np.nan

    for value in as_list(extra_nodata):
        if value is None:
            continue
        arr[np.isclose(arr, value)] = np.nan

    return arr


def load_as_physical(
    filepath: str | Path,
    extra_nodata=None,
    manual_scale: float | None = None,
    manual_offset: float = 0.0,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> tuple[np.ndarray, dict, float | None, float]:
    """Load a raster and convert values into physical units."""
    data, profile, nodata = read_raster(filepath)
    data = mask_nodata(data, nodata, extra_nodata=extra_nodata)

    if manual_scale is not None:
        scale_used = float(manual_scale)
        offset_used = float(manual_offset)
    else:
        scale_used, offset_used = get_scale_offset(filepath)

    if scale_used is not None:
        data = data * float(scale_used) + float(offset_used)

    if valid_min is not None:
        data[data < float(valid_min)] = np.nan
    if valid_max is not None:
        data[data > float(valid_max)] = np.nan

    profile.update(dtype="float32", nodata=np.nan, count=1)
    return data.astype(np.float32), profile, scale_used, float(offset_used)


def dist_stats(arr: np.ndarray) -> dict:
    """Compute simple descriptive stats on finite values only."""
    valid = np.asarray(arr, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return {
            "n": 0,
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "std": np.nan,
        }

    return {
        "n": int(valid.size),
        "min": float(valid.min()),
        "median": float(np.median(valid)),
        "mean": float(valid.mean()),
        "max": float(valid.max()),
        "std": float(valid.std()),
    }


def fit_yj_transform(
    data: np.ndarray,
    max_fit_pixels: int | None = 500_000,
    transform_chunk_size: int = 1_000_000,
    random_seed: int = 42,
) -> tuple[np.ndarray, PowerTransformer]:
    """Fit a Yeo-Johnson transform on valid pixels and apply it in chunks."""
    arr = np.asarray(data, dtype=np.float32)
    flat = arr.reshape(-1)
    valid_idx = np.flatnonzero(np.isfinite(flat))

    if valid_idx.size == 0:
        raise ValueError("No valid pixels found; cannot fit transform.")

    fit_values = flat[valid_idx].astype(np.float64)
    if max_fit_pixels is not None and fit_values.size > int(max_fit_pixels):
        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(
            fit_values.size, size=int(max_fit_pixels), replace=False
        )
        fit_values = fit_values[sample_idx]

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(fit_values.reshape(-1, 1))

    transformed_flat = np.full(flat.shape, np.nan, dtype=np.float32)
    for start in range(0, valid_idx.size, int(transform_chunk_size)):
        end = min(start + int(transform_chunk_size), valid_idx.size)
        idx = valid_idx[start:end]
        chunk = flat[idx].astype(np.float64).reshape(-1, 1)
        transformed_flat[idx] = pt.transform(chunk).reshape(-1).astype(np.float32)

    return transformed_flat.reshape(arr.shape), pt


def save_tif(
    data: np.ndarray,
    profile: dict,
    out_path: str | Path,
    overwrite: bool = False,
) -> bool:
    """Write a single-band raster to disk and return whether a file was written."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    if out_path.exists() and not overwrite:
        return False

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.asarray(data, dtype=np.float32), 1)
    return True


def inverse_yeo_johnson(y: np.ndarray, lmbda: float) -> np.ndarray:
    """Inverse Yeo-Johnson transform for positive and negative branches."""
    y = np.asarray(y, dtype=np.float64)
    x = np.empty_like(y)

    pos = y >= 0
    neg = ~pos

    if pos.any():
        x[pos] = inv_boxcox1p(y[pos], lmbda)
    if neg.any():
        x[neg] = -inv_boxcox1p(-y[neg], 2 - lmbda)

    return x


def back_transform_values(
    values: np.ndarray,
    lmbda: float,
    mean_: float,
    scale_: float,
) -> np.ndarray:
    """Undo z-score standardization followed by Yeo-Johnson transform."""
    y = np.asarray(values, dtype=np.float64) * float(scale_) + float(mean_)
    return inverse_yeo_johnson(y, float(lmbda))


def extract_trait_id(filepath: str | Path, valid_traits: Iterable[str]) -> str | None:
    """Extract the trait identifier by exact or substring match."""
    filepath = Path(filepath)
    stem = filepath.stem
    trait_set = list(valid_traits)
    if stem in trait_set:
        return stem
    for trait_id in trait_set:
        if trait_id in stem:
            return trait_id
    return None


def load_params_table(csv_path: str | Path) -> pd.DataFrame:
    """Load the transformation parameter table and normalize the trait index."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "trait" in df.columns:
        trait_col = "trait"
    elif "variable" in df.columns:
        trait_col = "variable"
    else:
        raise ValueError(
            "Parameter CSV must contain either a 'trait' or 'variable' column."
        )

    required = [
        trait_col,
        "yeo_johnson_lambda",
        "standardize_mean",
        "standardize_scale",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parameter CSV: {missing}")

    return df.set_index(trait_col)
