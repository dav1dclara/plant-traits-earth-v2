"""Dataset and dataloader for chips zarr formats (chips_centered and chips_luca).

Implements Ideas 1+2:
  Idea 1 — SPLOT-only primary training loss.
            Val/test chips are 100% SPLOT-center, so zero source mismatch in eval.
  Idea 2 — Linear GBIF→SPLOT calibration applied at load time.
            Per-trait shift+scale computed once from the training split, then
            applied in __getitem__ wherever GBIF labels are present.

Supported zarr formats (auto-detected by get_chips_dataloader):

  1. chips_centered/patch{N}_stride1 (supervision_splot_only / supervision_gbif_only):
       predictors/{name}               (N, C, P, P)  float32
       targets/supervision_splot_only  (N, 74, P, P) float32   interleaved [val_t, src_t, ...]
       targets/supervision_gbif_only   (N, 74, P, P) float32   interleaved [val_t, src_t, ...]
       center/gbif_mean / center/gbif_valid / center/splot_mean / center/splot_valid
       CRS: EPSG:3857 (Web Mercator), patch_size 3/5/7, stride 1

  2. chips_luca/patch7_stride3 (splot / gbif with 6 channels per trait):
       predictors/{name}  (N, C, P, P)  float32   (same 5 predictor groups)
       targets/splot      (N, 222, P, P) float32   6 channels per trait: [mean, n, std, q25, q75, ...]
       targets/gbif       (N, 222, P, P) float32   NaN where no observation
       bounds             (N, 4)         float64
       CRS: EPSG:6933 (Equal-Earth), patch_size 7, stride 3

  3. patch15_stride10 (comb format — legacy):
       targets/comb  (N, 259, P, P)  7 channels per trait: [mean, std, median, q05, q95, count, src]
       src: 1=GBIF, 2=SPLOT, NaN=none

Source encoding (for loss functions):
    0 = no data
    1 = GBIF
    2 = SPLOT

Trait ordering (37 traits, alphabetical by trait ID):
    X1080 X13 X138 X14 X144 X145 X146 X15 X163 X169 X21 X223 X224 X237 X26 X27
    X281 X282 X289 X297 X3106 X3107 X3112 X3113 X3114 X3117 X3120 X351 X4 X46
    X47 X50 X55 X6 X614 X78 X95
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRAITS = 37

TRAIT_NAMES: list[str] = [
    "X1080",
    "X13",
    "X138",
    "X14",
    "X144",
    "X145",
    "X146",
    "X15",
    "X163",
    "X169",
    "X21",
    "X223",
    "X224",
    "X237",
    "X26",
    "X27",
    "X281",
    "X282",
    "X289",
    "X297",
    "X3106",
    "X3107",
    "X3112",
    "X3113",
    "X3114",
    "X3117",
    "X3120",
    "X351",
    "X4",
    "X46",
    "X47",
    "X50",
    "X55",
    "X6",
    "X614",
    "X78",
    "X95",
]

PREDICTORS: list[str] = [
    "canopy_height",
    "modis",
    "soil_grids",
    "vodca",
    "worldclim",
]

# Minimum number of valid samples required per trait to fit calibration params.
_MIN_CALIB_SAMPLES = 20

# EPSG:6933 approximate global extents (used for lat/lon encoding of patch15 data).
_EPSG6933_X_HALF = 17_367_530.445
_EPSG6933_Y_HALF = 7_342_215.437


def _latlon_channels(
    bounds: np.ndarray,  # (N, 4) float64 [min_x, min_y, max_x, max_y]
    patch_size: int,
    crs_epsg: int = 3857,
) -> np.ndarray:  # (N, 4, P, P) float32
    """Compute per-pixel sinusoidal geographic encoding for each chip.

    Returns 4 channels: [sin_lon, cos_lon, sin_lat, cos_lat].
    All channels are in [-1, 1] and are consistent across CRS:
      - sin/cos of longitude encode periodicity (east/west wrap)
      - sin/cos of latitude encode north/south position

    Supports:
      EPSG:3857 (Web Mercator) — used by chips_centered zarrs
      EPSG:6933 (EASE-2 Equal-Area) — used by patch15_stride10 zarrs

    Reversible: just do not pass add_latlon=True at inference/training.
    """
    N = bounds.shape[0]
    P = patch_size

    # Pixel dimensions (may vary per chip, but in practice constant)
    pw = (bounds[:, 2] - bounds[:, 0]) / P  # (N,) pixel width  in CRS units
    ph = (bounds[:, 3] - bounds[:, 1]) / P  # (N,) pixel height in CRS units

    # Column-wise x centers: x[n, col] = min_x[n] + (col+0.5)*pw[n]
    col_idx = np.arange(P, dtype=np.float64)  # (P,)
    x_c = bounds[:, 0:1] + (col_idx[None, :] + 0.5) * pw[:, None]  # (N, P)

    # Row-wise y centers (top-down): y[n, row] = max_y[n] - (row+0.5)*ph[n]
    row_idx = np.arange(P, dtype=np.float64)  # (P,)
    y_c = bounds[:, 3:4] - (row_idx[None, :] + 0.5) * ph[:, None]  # (N, P)

    # Convert to lon/lat radians
    if crs_epsg == 3857:
        R = 6_378_137.0
        lon_rad = x_c / R  # (N, P) exact
        lat_rad = 2.0 * np.arctan(np.exp(y_c / R)) - np.pi / 2  # (N, P) exact
    else:
        # EPSG:6933 EASE-2: approximate linear mapping to lon/lat space
        lon_rad = (x_c / _EPSG6933_X_HALF) * np.pi  # (N, P)
        lat_rad = (y_c / _EPSG6933_Y_HALF) * (np.pi / 2)  # (N, P)

    # Broadcast to (N, P, P):
    #   lon varies with columns → broadcast over rows
    #   lat varies with rows   → broadcast over columns
    sin_lon = np.broadcast_to(np.sin(lon_rad)[:, None, :], (N, P, P)).copy()  # (N,P,P)
    cos_lon = np.broadcast_to(np.cos(lon_rad)[:, None, :], (N, P, P)).copy()
    sin_lat = np.broadcast_to(np.sin(lat_rad)[:, :, None], (N, P, P)).copy()  # (N,P,P)
    cos_lat = np.broadcast_to(np.cos(lat_rad)[:, :, None], (N, P, P)).copy()

    return np.stack([sin_lon, cos_lon, sin_lat, cos_lat], axis=1).astype(
        np.float32
    )  # (N,4,P,P)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


class GBIFCalibration:
    """Per-trait linear calibration parameters mapping GBIF → SPLOT transform space.

    Calibration formula applied per trait t:
        gbif_corrected[t] = (gbif_raw[t] - gbif_mean[t]) / gbif_std[t]
                            * splot_std[t] + splot_mean[t]

    This is a mathematically exact shift+scale that guarantees:
        E[gbif_corrected[t]] == splot_mean[t]
        Std[gbif_corrected[t]] == splot_std[t]

    Pearson correlation between GBIF pixels is preserved (rank invariant).

    Attributes:
        gbif_mean:  (37,) per-trait GBIF mean in training split
        gbif_std:   (37,) per-trait GBIF std in training split (>= 1e-6)
        splot_mean: (37,) per-trait SPLOT mean in training split
        splot_std:  (37,) per-trait SPLOT std in training split (>= 1e-6)
        scale:      (37,) splot_std / gbif_std — multiplicative factor
        shift:      (37,) splot_mean - gbif_mean * scale — additive offset
    """

    def __init__(
        self,
        gbif_mean: np.ndarray,
        gbif_std: np.ndarray,
        splot_mean: np.ndarray,
        splot_std: np.ndarray,
    ) -> None:
        self.gbif_mean = gbif_mean.astype(np.float32)
        self.gbif_std = np.maximum(gbif_std, 1e-6).astype(np.float32)
        self.splot_mean = splot_mean.astype(np.float32)
        self.splot_std = np.maximum(splot_std, 1e-6).astype(np.float32)
        # Pre-compute combined affine params for fast __getitem__ application.
        # gbif_corrected = raw * scale + shift
        self.scale = (self.splot_std / self.gbif_std).astype(np.float32)
        self.shift = (self.splot_mean - self.gbif_mean * self.scale).astype(np.float32)

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scale, shift) as (37, 1, 1) float32 tensors for broadcasting."""
        scale = torch.from_numpy(self.scale).view(N_TRAITS, 1, 1)
        shift = torch.from_numpy(self.shift).view(N_TRAITS, 1, 1)
        return scale, shift

    def delta_means(self) -> np.ndarray:
        """Return post-calibration Δmean per trait (should be ~0 for all traits)."""
        return self.gbif_mean * self.scale + self.shift - self.splot_mean


def compute_gbif_calibration(train_zarr_path: str | Path) -> GBIFCalibration:
    """Compute per-trait GBIF→SPLOT linear calibration params from the training split.

    MUST be called on train.zarr only (never val/test) to prevent data leakage.

    Args:
        train_zarr_path: Path to the train.zarr of chips_centered.

    Returns:
        GBIFCalibration with fitted parameters for all 37 traits.
    """
    z = zarr.open_group(str(train_zarr_path), mode="r")

    gbif_m = z["center/gbif_mean"][:]  # (N, 37) float32
    splot_m = z["center/splot_mean"][:]  # (N, 37) float32
    gbif_v = z["center/gbif_valid"][:]  # (N, 37) uint8 — 1 where GBIF valid
    splot_v = z["center/splot_valid"][:]  # (N, 37) uint8 — 1 where SPLOT valid

    gbif_mean = np.zeros(N_TRAITS, dtype=np.float64)
    gbif_std = np.ones(N_TRAITS, dtype=np.float64)
    splot_mean = np.zeros(N_TRAITS, dtype=np.float64)
    splot_std = np.ones(N_TRAITS, dtype=np.float64)

    for i in range(N_TRAITS):
        gv = gbif_m[gbif_v[:, i] == 1, i]
        sv = splot_m[splot_v[:, i] == 1, i]

        if len(gv) >= _MIN_CALIB_SAMPLES:
            gbif_mean[i] = float(gv.mean())
            gbif_std[i] = float(max(gv.std(), 1e-6))

        if len(sv) >= _MIN_CALIB_SAMPLES:
            splot_mean[i] = float(sv.mean())
            splot_std[i] = float(max(sv.std(), 1e-6))

    return GBIFCalibration(
        gbif_mean=gbif_mean.astype(np.float32),
        gbif_std=gbif_std.astype(np.float32),
        splot_mean=splot_mean.astype(np.float32),
        splot_std=splot_std.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ChipsCenteredDataset(Dataset):
    """PyTorch Dataset for chips_centered zarr (patch3/5/7_stride1 formats).

    Patch size is auto-detected from the zarr's `patch_size` attribute.
    The center pixel index is `patch_size // 2` (e.g. 1 for 3×3, 2 for 5×5).

    Returns per sample (5-tuple):
        X               (150, P, P) float32  — stacked EO predictors
        y_splot_vals    (37,  P, P) float32  — SPLOT label values (0 where no data)
        y_splot_src     (37,  P, P) float32  — SPLOT source mask  (2=SPLOT, 0=no data)
        y_gbif_vals     (37,  P, P) float32  — GBIF labels, linearly calibrated to SPLOT space
        y_gbif_src      (37,  P, P) float32  — GBIF source mask   (1=GBIF,  0=no data)

    P = patch_size (3, 5, or 7). center_idx = P // 2.

    Speed note: All zarr arrays are pre-loaded into numpy RAM at __init__ time.
    This eliminates per-sample zarr decompression overhead, which is the main
    bottleneck for small chips. Memory cost ≈ 600 MB for 3×3 train split,
    ~1.7 GB for 7×7 train split. On Linux (fork-based DataLoader), workers share
    this memory via CoW — no per-worker duplication beyond pages written.

    Args:
        zarr_path:   Path to one of {train,val,test,all}.zarr
        calibration: GBIFCalibration fitted on the training split.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        calibration: GBIFCalibration,
        add_latlon: bool = False,
    ) -> None:
        store = zarr.open_group(str(zarr_path), mode="r")

        # Auto-detect patch size (3, 5, or 7) from zarr metadata.
        self.patch_size = int(store.attrs.get("patch_size", 3))
        self.center_idx = self.patch_size // 2  # 1 for 3×3, 2 for 5×5, 3 for 7×7

        # Pre-load all arrays into numpy RAM. Each zarr read decompresses the
        # entire array once; __getitem__ then does pure numpy indexing.
        self._X_arrs: list[np.ndarray] = [
            store[f"predictors/{p}"][:]  # (N, C, 3, 3) float32
            for p in PREDICTORS
        ]
        self._sv_splot: np.ndarray = store["targets/supervision_splot_only"][
            :
        ]  # (N,74,3,3)
        self._sv_gbif: np.ndarray = store["targets/supervision_gbif_only"][
            :
        ]  # (N,74,3,3)

        self.n = self._sv_splot.shape[0]
        self.in_channels = sum(a.shape[1] for a in self._X_arrs)

        # Optional lat/lon sinusoidal encoding — 4 extra channels per pixel.
        # Reversible: train without add_latlon to restore original 150-channel input.
        self._latlon: np.ndarray | None = None
        if add_latlon:
            crs_epsg = int(store.attrs.get("crs_epsg", 3857))  # chips_centered → 3857
            _bounds = store["bounds"][:].astype(np.float64)  # (N, 4)
            self._latlon = _latlon_channels(_bounds, self.patch_size, crs_epsg)
            self.in_channels += 4

        # Calibration as (37, 1, 1) tensors for broadcasting in __getitem__.
        self._calib_scale, self._calib_shift = calibration.as_tensors()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,  # X              (150, 3, 3)
        torch.Tensor,  # y_splot_vals   (37,  3, 3)
        torch.Tensor,  # y_splot_src    (37,  3, 3)
        torch.Tensor,  # y_gbif_vals    (37,  3, 3)  calibrated
        torch.Tensor,  # y_gbif_src     (37,  3, 3)
    ]:
        # ---- Predictors (pure numpy indexing — no disk I/O) ----
        X = torch.from_numpy(
            np.concatenate([a[idx] for a in self._X_arrs], axis=0)
        ).float()  # (150, 3, 3)
        if self._latlon is not None:
            X = torch.cat(
                [X, torch.from_numpy(self._latlon[idx])], dim=0
            )  # (154, P, P)

        # ---- SPLOT supervision ----
        # Interleaved: [val_t0, src_t0, val_t1, src_t1, ...] → (74, 3, 3)
        sv_splot = torch.from_numpy(self._sv_splot[idx]).float()
        y_splot_vals = sv_splot[0::2]  # (37, 3, 3) values
        y_splot_src = sv_splot[1::2]  # (37, 3, 3) sources — {0, 2}

        # ---- GBIF supervision + linear calibration ----
        sv_gbif = torch.from_numpy(self._sv_gbif[idx]).float()
        y_gbif_vals_raw = sv_gbif[0::2]  # (37, 3, 3) raw GBIF values
        y_gbif_src = sv_gbif[1::2]  # (37, 3, 3) sources — {0, 1}

        gbif_has_data = y_gbif_src > 0  # (37, 3, 3) bool
        y_gbif_vals_calib = torch.where(
            gbif_has_data,
            y_gbif_vals_raw * self._calib_scale + self._calib_shift,
            torch.zeros_like(y_gbif_vals_raw),
        )

        return X, y_splot_vals, y_splot_src, y_gbif_vals_calib, y_gbif_src


# ---------------------------------------------------------------------------
# Patch15Dataset — for patch15_stride10 zarr (comb target format)
# ---------------------------------------------------------------------------


class Patch15Dataset(Dataset):
    """PyTorch Dataset for patch15_stride10 zarr (global grid chips).

    Target format differs from ChipsCenteredDataset: instead of separate
    supervision_splot_only / supervision_gbif_only arrays, targets are stored
    as a single ``comb`` array with 7 channels per trait (37 traits × 7 = 259):
        [mean, std, median, q05, q95, count, source]
    Source encoding: 1 = GBIF, 2 = SPLOT, NaN = no data.

    Returns the same 5-tuple as ChipsCenteredDataset so train_chips.py works
    without modification:
        X               (150, P, P) float32  — stacked EO predictors
        y_splot_vals    (37,  P, P) float32  — SPLOT mean values (0 where no data)
        y_splot_src     (37,  P, P) float32  — SPLOT source mask  (2=SPLOT, 0=no data)
        y_gbif_vals     (37,  P, P) float32  — GBIF mean values, calibrated to SPLOT space
        y_gbif_src      (37,  P, P) float32  — GBIF source mask   (1=GBIF,  0=no data)
    """

    def __init__(
        self,
        zarr_path: str | Path,
        calibration: GBIFCalibration,
        add_latlon: bool = False,
    ) -> None:
        store = zarr.open_group(str(zarr_path), mode="r")

        self.patch_size = int(store.attrs.get("patch_size", 15))
        self.center_idx = self.patch_size // 2

        # Pre-load predictors
        self._X_arrs: list[np.ndarray] = [
            store[f"predictors/{p}"][:] for p in PREDICTORS
        ]
        self.in_channels = sum(a.shape[1] for a in self._X_arrs)

        # Optional lat/lon sinusoidal encoding — 4 extra channels per pixel.
        self._latlon: np.ndarray | None = None
        if add_latlon:
            crs_epsg = int(store.attrs.get("crs_epsg", 6933))  # patch15 → 6933
            _bounds = store["bounds"][:].astype(np.float64)  # (N, 4)
            self._latlon = _latlon_channels(_bounds, self.patch_size, crs_epsg)
            self.in_channels += 4

        # Pre-load comb and convert to splot/gbif supervision arrays.
        # comb: (N, 37*7, P, P)  channels per trait: [mean, std, median, q05, q95, count, source]
        comb = store["targets/comb"][:]  # (N, 259, P, P) float32
        N, _, P, _ = comb.shape
        self.n = N

        sv_splot = np.zeros((N, N_TRAITS * 2, P, P), dtype=np.float32)  # (N, 74, P, P)
        sv_gbif = np.zeros((N, N_TRAITS * 2, P, P), dtype=np.float32)

        for t in range(N_TRAITS):
            mean_ch = comb[:, t * 7 + 0, :, :]  # (N, P, P)
            src_ch = comb[:, t * 7 + 6, :, :]  # (N, P, P) — 1=GBIF, 2=SPLOT, NaN=none

            splot_mask = src_ch == 2
            gbif_mask = src_ch == 1

            sv_splot[:, t * 2 + 0, :, :] = np.where(splot_mask, mean_ch, 0.0)
            sv_splot[:, t * 2 + 1, :, :] = np.where(splot_mask, 2.0, 0.0)

            sv_gbif[:, t * 2 + 0, :, :] = np.where(gbif_mask, mean_ch, 0.0)
            sv_gbif[:, t * 2 + 1, :, :] = np.where(gbif_mask, 1.0, 0.0)

        del comb  # free ~1.4 GB

        self._sv_splot = sv_splot
        self._sv_gbif = sv_gbif
        self._calib_scale, self._calib_shift = calibration.as_tensors()

    def __len__(self) -> int:
        return self.n

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(
            np.concatenate([a[idx] for a in self._X_arrs], axis=0)
        ).float()
        if self._latlon is not None:
            X = torch.cat([X, torch.from_numpy(self._latlon[idx])], dim=0)

        sv_splot = torch.from_numpy(self._sv_splot[idx]).float()
        y_splot_vals = sv_splot[0::2]
        y_splot_src = sv_splot[1::2]

        sv_gbif = torch.from_numpy(self._sv_gbif[idx]).float()
        y_gbif_vals_raw = sv_gbif[0::2]
        y_gbif_src = sv_gbif[1::2]

        gbif_has_data = y_gbif_src > 0
        y_gbif_vals_calib = torch.where(
            gbif_has_data,
            y_gbif_vals_raw * self._calib_scale + self._calib_shift,
            torch.zeros_like(y_gbif_vals_raw),
        )

        return X, y_splot_vals, y_splot_src, y_gbif_vals_calib, y_gbif_src


# ---------------------------------------------------------------------------
# LucaChipsDataset — for chips_luca/patch7_stride3 zarr
# ---------------------------------------------------------------------------


class LucaChipsDataset(Dataset):
    """PyTorch Dataset for chips_luca/patch7_stride3 zarr format.

    Target format: separate ``splot`` and ``gbif`` arrays with 6 channels per
    trait (37 traits × 6 = 222 channels):
        [mean, n_obs, std, q25, q75, ...]
    Channel 0 of each 6-group is the mean value used for training.
    NaN in channel 0 encodes no observation for that trait at that pixel.

    Predictors are stored as 5 named sub-arrays (same names as chips_centered).

    Returns the same 5-tuple as ChipsCenteredDataset so train_chips.py works
    without modification:
        X               (150, P, P) float32  — stacked EO predictors
        y_splot_vals    (37,  P, P) float32  — SPLOT mean values (0 where no data)
        y_splot_src     (37,  P, P) float32  — SPLOT source mask  (2=SPLOT, 0=no data)
        y_gbif_vals     (37,  P, P) float32  — GBIF mean values, calibrated to SPLOT space
        y_gbif_src      (37,  P, P) float32  — GBIF source mask   (1=GBIF,  0=no data)
    """

    # Number of channels per trait in the splot/gbif target arrays.
    CHANNELS_PER_TRAIT = 6

    def __init__(
        self,
        zarr_path: str | Path,
        calibration: GBIFCalibration,
        add_latlon: bool = False,
    ) -> None:
        store = zarr.open(str(zarr_path), mode="r")

        self.patch_size = int(store.attrs.get("patch_size", 7))
        self.center_idx = self.patch_size // 2  # 3 for patch7

        # Pre-load predictors (same 5 named groups as chips_centered)
        self._X_arrs: list[np.ndarray] = [
            store[f"predictors/{p}"][:] for p in PREDICTORS
        ]
        self.in_channels = sum(a.shape[1] for a in self._X_arrs)

        # Optional lat/lon sinusoidal encoding (4 extra channels per pixel).
        self._latlon: np.ndarray | None = None
        if add_latlon:
            crs_epsg = int(store.attrs.get("crs_epsg", 6933))  # chips_luca → 6933
            _bounds = store["bounds"][:].astype(np.float64)  # (N, 4)
            self._latlon = _latlon_channels(_bounds, self.patch_size, crs_epsg)
            self.in_channels += 4

        # Pre-load mean channels only (ch0 of each 6-group) to save RAM.
        # splot/gbif: (N, 222, P, P) — extract every 6th channel starting at 0.
        C = self.CHANNELS_PER_TRAIT
        splot_raw = store["targets/splot"][:]  # (N, 222, P, P)
        gbif_raw = store["targets/gbif"][:]  # (N, 222, P, P)
        splot_means = splot_raw[:, 0::C, :, :]  # (N, 37, P, P)  mean values
        gbif_means = gbif_raw[:, 0::C, :, :]  # (N, 37, P, P)
        del splot_raw, gbif_raw

        self.n = splot_means.shape[0]
        self._splot_means: np.ndarray = splot_means  # (N, 37, P, P)  NaN where no data
        self._gbif_means: np.ndarray = gbif_means  # (N, 37, P, P)  NaN where no data
        self._calib_scale, self._calib_shift = calibration.as_tensors()

    def __len__(self) -> int:
        return self.n

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,  # X              (150, P, P)
        torch.Tensor,  # y_splot_vals   (37,  P, P)
        torch.Tensor,  # y_splot_src    (37,  P, P)
        torch.Tensor,  # y_gbif_vals    (37,  P, P)  calibrated
        torch.Tensor,  # y_gbif_src     (37,  P, P)
    ]:
        # ---- Predictors ----
        X = torch.from_numpy(
            np.concatenate([a[idx] for a in self._X_arrs], axis=0)
        ).float()
        if self._latlon is not None:
            X = torch.cat([X, torch.from_numpy(self._latlon[idx])], dim=0)

        # ---- SPLOT supervision ----
        splot_t = torch.from_numpy(self._splot_means[idx]).float()  # (37, P, P)
        splot_valid = torch.isfinite(splot_t)
        y_splot_vals = torch.where(splot_valid, splot_t, torch.zeros_like(splot_t))
        y_splot_src = torch.where(
            splot_valid,
            torch.full_like(splot_t, 2.0),  # 2 = SPLOT
            torch.zeros_like(splot_t),
        )

        # ---- GBIF supervision + linear calibration ----
        gbif_t = torch.from_numpy(self._gbif_means[idx]).float()  # (37, P, P)
        gbif_valid = torch.isfinite(gbif_t)
        y_gbif_vals_raw = torch.where(gbif_valid, gbif_t, torch.zeros_like(gbif_t))
        y_gbif_src = torch.where(
            gbif_valid,
            torch.ones_like(gbif_t),  # 1 = GBIF
            torch.zeros_like(gbif_t),
        )

        gbif_has_data = y_gbif_src > 0
        y_gbif_vals_calib = torch.where(
            gbif_has_data,
            y_gbif_vals_raw * self._calib_scale + self._calib_shift,
            torch.zeros_like(y_gbif_vals_raw),
        )

        return X, y_splot_vals, y_splot_src, y_gbif_vals_calib, y_gbif_src


def compute_gbif_calibration_patch15(train_zarr_path: str | Path) -> GBIFCalibration:
    """Compute per-trait GBIF→SPLOT calibration params from a patch15_stride10 train zarr.

    Uses only the center pixel of each chip to avoid spatial autocorrelation leakage.

    Args:
        train_zarr_path: Path to train.zarr of patch15_stride10 format.

    Returns:
        GBIFCalibration with fitted parameters for all 37 traits.
    """
    z = zarr.open_group(str(train_zarr_path), mode="r")
    comb = z["targets/comb"][:]  # (N, 259, P, P)
    P = comb.shape[2]
    ci = P // 2  # center pixel index

    gbif_mean = np.zeros(N_TRAITS, dtype=np.float64)
    gbif_std = np.ones(N_TRAITS, dtype=np.float64)
    splot_mean = np.zeros(N_TRAITS, dtype=np.float64)
    splot_std = np.ones(N_TRAITS, dtype=np.float64)

    for t in range(N_TRAITS):
        mean_c = comb[:, t * 7 + 0, ci, ci]  # (N,)
        src_c = comb[:, t * 7 + 6, ci, ci]  # (N,)

        gv = mean_c[(src_c == 1) & np.isfinite(mean_c)]  # GBIF center values
        sv = mean_c[(src_c == 2) & np.isfinite(mean_c)]  # SPLOT center values

        if len(gv) >= _MIN_CALIB_SAMPLES:
            gbif_mean[t] = float(gv.mean())
            gbif_std[t] = float(max(gv.std(), 1e-6))

        if len(sv) >= _MIN_CALIB_SAMPLES:
            splot_mean[t] = float(sv.mean())
            splot_std[t] = float(max(sv.std(), 1e-6))

    return GBIFCalibration(
        gbif_mean=gbif_mean.astype(np.float32),
        gbif_std=gbif_std.astype(np.float32),
        splot_mean=splot_mean.astype(np.float32),
        splot_std=splot_std.astype(np.float32),
    )


def compute_gbif_calibration_luca(train_zarr_path: str | Path) -> GBIFCalibration:
    """Compute per-trait GBIF→SPLOT calibration params from a chips_luca train zarr.

    The chips_luca format stores 6 channels per trait in separate splot/gbif arrays.
    Channel 0 of each 6-group is the mean trait value; NaN encodes no data.
    Calibration uses only center-pixel means to avoid spatial autocorrelation.

    Args:
        train_zarr_path: Path to train.zarr of chips_luca format.

    Returns:
        GBIFCalibration with fitted parameters for all 37 traits.
    """
    z = zarr.open(str(train_zarr_path), mode="r")
    P = int(z.attrs.get("patch_size", 7))
    ci = P // 2  # center pixel index

    # Load center-pixel means for all traits at once (cheap: (N, 222) slice)
    splot_center_all = z["targets/splot"][:, :, ci, ci]  # (N, 222)
    gbif_center_all = z["targets/gbif"][:, :, ci, ci]  # (N, 222)

    # Channel 0 of each 6-group = mean trait value
    splot_c = splot_center_all[:, 0::6]  # (N, 37)
    gbif_c = gbif_center_all[:, 0::6]  # (N, 37)
    del splot_center_all, gbif_center_all

    gbif_mean = np.zeros(N_TRAITS, dtype=np.float64)
    gbif_std = np.ones(N_TRAITS, dtype=np.float64)
    splot_mean = np.zeros(N_TRAITS, dtype=np.float64)
    splot_std = np.ones(N_TRAITS, dtype=np.float64)

    for t in range(N_TRAITS):
        gv = gbif_c[:, t]
        gv = gv[np.isfinite(gv)]
        sv = splot_c[:, t]
        sv = sv[np.isfinite(sv)]

        if len(gv) >= _MIN_CALIB_SAMPLES:
            gbif_mean[t] = float(gv.mean())
            gbif_std[t] = float(max(gv.std(), 1e-6))

        if len(sv) >= _MIN_CALIB_SAMPLES:
            splot_mean[t] = float(sv.mean())
            splot_std[t] = float(max(sv.std(), 1e-6))

    return GBIFCalibration(
        gbif_mean=gbif_mean.astype(np.float32),
        gbif_std=gbif_std.astype(np.float32),
        splot_mean=splot_mean.astype(np.float32),
        splot_std=splot_std.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------


def get_chips_dataloader(
    zarr_path: str | Path,
    calibration: GBIFCalibration,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    pin_memory: bool = True,
    add_latlon: bool = False,
) -> DataLoader:
    """Return a DataLoader, auto-detecting zarr format.

    Detects format by inspecting the ``targets/`` group keys:
      - ``supervision_splot_only`` → ChipsCenteredDataset  (chips_centered format)
      - ``splot`` (without supervision_splot_only) → LucaChipsDataset  (chips_luca format)
      - ``comb`` → Patch15Dataset  (patch15_stride10 legacy format)

    Args:
        zarr_path:   Path to split zarr (train.zarr / val.zarr / test.zarr).
        calibration: GBIFCalibration fitted on the training split.
        batch_size:  Samples per batch.
        num_workers: Parallel data-loading workers.
        shuffle:     Shuffle before each epoch (use True for training).
        pin_memory:  Pin memory for faster GPU transfer.
    """
    _store = zarr.open(str(zarr_path), mode="r")
    _target_keys = list(_store["targets"].keys())
    if "supervision_splot_only" in _target_keys:
        ds: Dataset = ChipsCenteredDataset(
            zarr_path, calibration, add_latlon=add_latlon
        )
    elif "splot" in _target_keys:
        ds = LucaChipsDataset(zarr_path, calibration, add_latlon=add_latlon)
    elif "comb" in _target_keys:
        ds = Patch15Dataset(zarr_path, calibration, add_latlon=add_latlon)
    else:
        raise ValueError(
            f"Unrecognised target format in {zarr_path}. "
            f"Expected 'supervision_splot_only', 'splot', or 'comb' under targets/. "
            f"Found: {_target_keys}"
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
