"""Evaluation script for checkpoints produced by train_chips.py.

Computes center-pixel SPLOT Pearson-r on the test split (test.zarr).
Uses the same data pipeline as train_chips.py — no data leakage.

The test split is completely untouched during training; this script is the
first (and only) time test.zarr is read for evaluation.

Usage:
    # Evaluate best-r checkpoint for STL
    python scripts/test_chips.py --checkpoint scripts/Checkpoints_Scores/checkpoints/chips_stl_best_r.pth

    # Evaluate best-r checkpoint for MTL
    python scripts/test_chips.py --checkpoint scripts/Checkpoints_Scores/checkpoints/chips_mtl_best_r.pth

    # Evaluate best-r checkpoint for MMoE V3
    python scripts/test_chips.py --checkpoint scripts/Checkpoints_Scores/checkpoints/chips_mmoe_v3_best_r.pth

    # With W&B logging
    python scripts/test_chips.py --checkpoint <path> --wandb

    # Custom output directory
    python scripts/test_chips.py --checkpoint <path> --output_dir results/chips/

    # Best-loss checkpoints (secondary metric, optional comparison)
    python scripts/test_chips.py --checkpoint scripts/Checkpoints_Scores/checkpoints/chips_stl.pth
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import zarr as _zarr_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

try:
    from rich.console import Console
    from rich.progress import track
except ImportError:

    class Console:
        def print(self, *args, **kwargs):
            print(*args)

        def rule(self, text: str):
            print(f"\n{'─' * 20} {text} {'─' * 20}")

    def track(it, description=""):
        return it


from ptev2.data.chips_dataset import (
    N_TRAITS,
    PREDICTORS,
    TRAIT_NAMES,
    GBIFCalibration,
    compute_gbif_calibration,
    compute_gbif_calibration_luca,
    compute_gbif_calibration_patch15,
    get_chips_dataloader,
)
from ptev2.utils import seed_all

console = Console()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, object]:
    """Load checkpoint and embedded OmegaConf config saved by train_chips.py."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a dict: {checkpoint_path}")
    cfg_raw = state.get("config")
    if cfg_raw is None:
        raise ValueError(f"Checkpoint has no embedded 'config': {checkpoint_path}")
    cfg = OmegaConf.create(cfg_raw)
    return state, cfg


def compute_pearson_r(
    preds_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    masks_list: list[torch.Tensor],
) -> list[float]:
    """Compute per-trait center-pixel SPLOT Pearson-r from collected tensors.

    Uses float64 mean-centering for numerical stability (no running-sum formula).
    """
    if not preds_list:
        return [float("nan")] * N_TRAITS

    P = torch.cat(preds_list, dim=0)  # (N_test, 37)
    G = torch.cat(labels_list, dim=0)  # (N_test, 37)
    M = torch.cat(masks_list, dim=0)  # (N_test, 37) bool

    per_trait_r: list[float] = []
    for t in range(N_TRAITS):
        valid_t = M[:, t]
        n = valid_t.sum().item()
        if n > 1:
            p_t = P[valid_t, t].double()
            g_t = G[valid_t, t].double()
            p_m = p_t - p_t.mean()
            g_m = g_t - g_t.mean()
            num_r = (p_m * g_m).sum()
            den_r = torch.sqrt(p_m.pow(2).sum() * g_m.pow(2).sum())
            r = float(num_r / (den_r + 1e-12)) if den_r > 1e-12 else 0.0
            per_trait_r.append(max(-1.0, min(1.0, r)))
        else:
            per_trait_r.append(float("nan"))
    return per_trait_r


def _generate_trait_maps(
    map_preds_list: list,
    chip_lon: np.ndarray,
    chip_lat: np.ndarray,
    per_trait_r: list,
    output_dir: Path,
    stem: str,
    console: object,
) -> None:
    """Generate a grid of predicted-trait scatter maps and save as a PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[yellow]matplotlib not installed — skipping map generation.[/yellow]"
        )
        return

    all_preds = torch.cat(map_preds_list, dim=0).numpy()  # (N_chips, 37)
    all_preds = np.where(np.isfinite(all_preds), all_preds, np.nan)

    N_COLS = 7
    N_ROWS = math.ceil(N_TRAITS / N_COLS)  # 6 rows for 37 traits
    fig, axes = plt.subplots(
        N_ROWS, N_COLS, figsize=(N_COLS * 3, N_ROWS * 2.2), dpi=100
    )
    axes = axes.flatten()

    for t in range(N_TRAITS):
        ax = axes[t]
        vals = all_preds[:, t]
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            ax.set_visible(False)
            continue
        v_min = float(np.nanpercentile(vals[finite], 2))
        v_max = float(np.nanpercentile(vals[finite], 98))
        sc = ax.scatter(
            chip_lon[finite],
            chip_lat[finite],
            c=vals[finite],
            s=1.2,
            linewidths=0,
            cmap="viridis",
            vmin=v_min,
            vmax=v_max,
            rasterized=True,
        )
        r_val = per_trait_r[t]
        r_str = f"{r_val:+.3f}" if math.isfinite(r_val) else "N/A"
        ax.set_title(f"{TRAIT_NAMES[t]}  r={r_str}", fontsize=7, pad=2)
        ax.tick_params(labelsize=5)
        fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)

    for i in range(N_TRAITS, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Predicted trait maps — {stem}", fontsize=10)
    fig.tight_layout()
    map_path = output_dir / f"{stem}_predicted_maps.png"
    fig.savefig(map_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  Trait maps saved: [cyan]{map_path}[/cyan]")


def _generate_global_tif(
    model: object,
    all_zarr_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    console: object,
) -> None:
    """Mosaic global trait predictions from all.zarr into a continuous GeoTIF.

    Two modes, selected by the zarr's `patch_size` attribute:

    patch_size == 3  (chips_centered all.zarr, SPLOT-location chips):
        Direct inference — each chip is a 3×3 predictor patch.  Output covers
        only SPLOT plot locations (~68k chips), so the global map has gaps.

    patch_size == 15  (patch15_stride10 all.zarr, global uniform grid):
        Dense sub-window inference — for each 15×15 chip we slide a 3×3 window
        at stride 1 across all 13×13 = 169 positions.  We run the model on each
        sub-window and use only the CENTER PIXEL prediction (the position the
        model was trained to predict accurately).  With stride-10 chips, adjacent
        chips overlap by 5 pixels, so every canvas pixel receives predictions
        from at least one sub-window → fully continuous global map.
        Total predictions: 7000 chips × 169 sub-windows ≈ 1.18M (batched).
    """
    try:
        import rasterio
        from affine import Affine
        from rasterio.crs import CRS
    except ImportError:
        console.print(
            "[red]rasterio / affine not installed — cannot write GeoTIF.\n"
            "Install with:  pip install rasterio affine[/red]"
        )
        return

    console.print(f"\nLoading global chips from [cyan]{all_zarr_path}[/cyan]...")
    store = _zarr_module.open_group(str(all_zarr_path), mode="r")

    # ---- geographic metadata -----------------------------------------------
    t_list = list(store.attrs["transform"])
    geo_tf = Affine(t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5])
    crs_epsg = int(store.attrs["crs_epsg"])
    canvas_rows = int(store.attrs["raster_height"])
    canvas_cols = int(store.attrs["raster_width"])
    pixel_w = float(geo_tf.a)
    pixel_h = float(abs(geo_tf.e))
    origin_x = float(geo_tf.c)  # left edge (easting  of pixel 0,0 top-left)
    origin_y = float(geo_tf.f)  # top  edge (northing of pixel 0,0 top-left)

    patch_size = int(store.attrs.get("patch_size", 15))
    bounds = store["bounds"][:]  # (N, 4) [min_x, min_y, max_x, max_y]
    N_chips = bounds.shape[0]

    console.print(
        f"  patch_size={patch_size}  chips={N_chips:,}  "
        f"canvas={canvas_rows}×{canvas_cols}  EPSG:{crs_epsg}"
    )

    # ---- load full predictor arrays into RAM --------------------------------
    console.print("  Loading predictor arrays...")
    X_parts = []
    for pname in PREDICTORS:
        X_parts.append(store[f"predictors/{pname}"][:].astype(np.float32))
    X_all = np.concatenate(X_parts, axis=1)  # (N, C, H, W)  H=W=patch_size
    del X_parts
    console.print(f"  Predictors loaded: {X_all.shape}  ({X_all.nbytes / 1e9:.2f} GB)")

    # ---- canvas accumulators -----------------------------------------------
    canvas = np.zeros((N_TRAITS, canvas_rows, canvas_cols), dtype=np.float64)
    weight_sum = np.zeros((canvas_rows, canvas_cols), dtype=np.float64)

    model.eval()

    # ================================================================
    # MODE A: 3×3 chips (chips_centered all.zarr)
    # Each chip → one 3×3 prediction block placed at chip's canvas position.
    # Coverage is limited to SPLOT plot locations.
    # ================================================================
    if patch_size == 3:
        console.print("  Mode: direct 3×3 inference (SPLOT-location chips)...")
        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, N_chips, batch_size):
                x = torch.from_numpy(X_all[i : i + batch_size]).to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x = torch.clamp(x, -1e4, 1e4)
                all_preds.append(model(x).cpu().numpy())  # (B, 37, 3, 3)
        preds_all = np.concatenate(all_preds, axis=0)  # (N, 37, 3, 3)
        del all_preds

        console.print("  Mosaicing...")
        for i in range(N_chips):
            min_x, _, _, max_y = bounds[i]
            col = int(round((min_x - origin_x) / pixel_w))
            row = int(round((origin_y - max_y) / pixel_h))
            r0 = max(row, 0)
            r1 = min(row + 3, canvas_rows)
            c0 = max(col, 0)
            c1 = min(col + 3, canvas_cols)
            lr = r0 - row
            lc = c0 - col
            if r0 >= r1 or c0 >= c1:
                continue
            chip = preds_all[i, :, lr : lr + (r1 - r0), lc : lc + (c1 - c0)]
            fm = np.isfinite(chip[0]).astype(np.float64)
            canvas[:, r0:r1, c0:c1] += chip * fm
            weight_sum[r0:r1, c0:c1] += fm

    # ================================================================
    # MODE B: 15×15 chips (patch15_stride10 all.zarr)
    # Slide 3×3 window at stride 1 → 13×13 = 169 sub-windows per chip.
    # Only keep the CENTER PIXEL [1,1] of each sub-window prediction.
    # This gives a fully continuous global map:
    #   chip at (r,c) → predictions at canvas pixels (r+1..r+13, c+1..c+13)
    #   stride-10 chips overlap by 5 pixels → every pixel covered ≥ once.
    # ================================================================
    else:
        SW = patch_size - 2  # number of stride-1 sub-window positions per axis
        #  patch_size=15 → SW=13 → 13×13=169 sub-windows per chip
        n_sw = SW * SW
        console.print(
            f"  Mode: dense sub-window inference  "
            f"({SW}×{SW}={n_sw} sub-windows per chip, "
            f"~{N_chips * n_sw:,} total predictions)..."
        )

        # Pre-compute per-chip canvas top-left (row, col)
        chip_rows = np.round((origin_y - bounds[:, 3]) / pixel_h).astype(
            np.int32
        )  # max_y → top
        chip_cols = np.round((bounds[:, 0] - origin_x) / pixel_w).astype(
            np.int32
        )  # min_x → left

        # Batch all sub-windows across all chips; flush to canvas every flush_every chips
        flush_every = max(1, batch_size // n_sw)  # chips per flush

        sw_X: list[np.ndarray] = []  # accumulated sub-window predictors
        sw_rc: list[tuple[int, int]] = []  # canvas (row, col) for center pixel

        def _flush(sw_X_acc, sw_rc_acc):
            if not sw_X_acc:
                return
            x_batch = torch.from_numpy(
                np.stack(sw_X_acc, axis=0)  # (M, 150, 3, 3)
            ).to(device)
            x_batch = torch.nan_to_num(x_batch, nan=0.0, posinf=0.0, neginf=0.0)
            x_batch = torch.clamp(x_batch, -1e4, 1e4)
            with torch.no_grad():
                preds = model(x_batch)[:, :, 1, 1].cpu().numpy()  # (M, 37) center pixel

            for k, (r, c) in enumerate(sw_rc_acc):
                if not (0 <= r < canvas_rows and 0 <= c < canvas_cols):
                    continue
                p = preds[k]  # (37,)
                fm = float(np.isfinite(p).all())
                if fm == 0:
                    continue
                canvas[:, r, c] += p
                weight_sum[r, c] += 1.0

        with torch.no_grad():
            for i in range(N_chips):
                chip_data = X_all[i]  # (C, 15, 15)
                cr = int(chip_rows[i])
                cc = int(chip_cols[i])

                for dr in range(SW):
                    for dc in range(SW):
                        sw = chip_data[:, dr : dr + 3, dc : dc + 3]  # (C, 3, 3)
                        sw_X.append(sw)
                        # center pixel of this sub-window sits at canvas (cr+dr+1, cc+dc+1)
                        sw_rc.append((cr + dr + 1, cc + dc + 1))

                if (i + 1) % flush_every == 0 or i == N_chips - 1:
                    _flush(sw_X, sw_rc)
                    sw_X.clear()
                    sw_rc.clear()
                    if (i + 1) % max(flush_every * 20, 200) == 0:
                        pct = 100 * (i + 1) / N_chips
                        console.print(f"    {i + 1:,}/{N_chips:,} chips  ({pct:.0f}%)")

    # ---- normalise canvas --------------------------------------------------
    nonzero = weight_sum > 0
    canvas[:, nonzero] /= weight_sum[nonzero]
    canvas[:, ~nonzero] = np.nan
    covered_pct = 100.0 * nonzero.sum() / nonzero.size
    console.print(
        f"  Canvas coverage: {covered_pct:.1f}%  "
        f"({nonzero.sum():,}/{nonzero.size:,} pixels)"
    )

    # ---- write GeoTIF ------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_geo = Affine(pixel_w, 0.0, origin_x, 0.0, -pixel_h, origin_y)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=canvas_rows,
        width=canvas_cols,
        count=N_TRAITS,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=out_geo,
        compress="deflate",
        tiled=True,
        nodata=float("nan"),
    ) as dst:
        dst.write(canvas.astype(np.float32))
        for b_idx, trait_name in enumerate(TRAIT_NAMES, start=1):
            dst.set_band_description(b_idx, trait_name)

    console.print(f"  Global GeoTIF saved: [cyan]{output_path}[/cyan]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a train_chips.py checkpoint on the test split."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to a .pth checkpoint saved by train_chips.py (best_r or best_loss).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/metrics",
        type=str,
        help="Directory to write JSON results (default: results/chips/).",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Override batch size (default: use checkpoint training config).",
    )
    parser.add_argument(
        "--num_workers",
        default=None,
        type=int,
        help="Override num_workers (default: use checkpoint training config).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to W&B (requires wandb installed and configured).",
    )
    parser.add_argument(
        "--maps",
        action="store_true",
        help="Generate predicted trait maps (scatter plots) saved as a PNG in output_dir.",
    )
    parser.add_argument(
        "--tif",
        action="store_true",
        help=(
            "Generate a global GeoTIF prediction map using all.zarr "
            "(center 3×3 crop from 15×15 global patches, EPSG:6933). "
            "Requires rasterio and affine to be installed."
        ),
    )
    parser.add_argument(
        "--all_zarr",
        default=None,
        type=str,
        help=(
            "Path to the global all.zarr (patch15_stride10 format). "
            "Default: auto-detect at <data.root_dir>/chips/patch15_stride10/all.zarr."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device override (e.g. 'cpu', 'cuda:0'). Default: auto-detect.",
    )
    parser.add_argument(
        "--no_gbif_calib",
        action="store_true",
        help="Ablation: skip GBIF→SPLOT linear calibration (scale=1, shift=0 for all traits).",
    )
    parser.add_argument(
        "--eval_all_splot",
        action="store_true",
        help="Evaluate Pearson-r at ALL SPLOT pixels in each patch (not just center). "
        "Gives more test samples; may double-count SPLOT locs shared across chips.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ------------------------------------------------------------------ #
    # Device                                                              #
    # ------------------------------------------------------------------ #
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    console.rule("[bold cyan]TEST CHIPS — Evaluation[/bold cyan]")
    console.print(f"Checkpoint: [cyan]{checkpoint_path}[/cyan]")
    console.print(f"Device:     [cyan]{device}[/cyan]")

    # ------------------------------------------------------------------ #
    # Load checkpoint + embedded config                                   #
    # ------------------------------------------------------------------ #
    state, cfg = load_checkpoint(checkpoint_path, device)
    seed_all(int(cfg.train.seed))

    in_channels = int(state.get("in_channels", 150))
    out_channels = int(state.get("out_channels", N_TRAITS))
    run_name = str(OmegaConf.select(cfg, "train.run_name") or checkpoint_path.stem)

    console.print(f"Run name:   [cyan]{run_name}[/cyan]")
    console.print(f"Epoch:      [cyan]{state.get('epoch', '?')}[/cyan]")
    console.print(f"Val r (train): [cyan]{state.get('val_mean_r', 'N/A')}[/cyan]")

    # ------------------------------------------------------------------ #
    # Data paths (derived from embedded config)                           #
    # ------------------------------------------------------------------ #
    zarr_dir = Path(cfg.data.root_dir) / cfg.data.zarr_dir
    train_zarr = zarr_dir / "train.zarr"
    test_zarr = zarr_dir / "test.zarr"

    for p in (train_zarr, test_zarr):
        if not p.exists():
            raise FileNotFoundError(f"Required zarr not found: {p}")

    console.print(f"\n[bold]Data[/bold]")
    console.print(f"  train.zarr (calibration only): [dim]{train_zarr}[/dim]")
    console.print(f"  test.zarr  (evaluation):       [cyan]{test_zarr}[/cyan]")

    # ------------------------------------------------------------------ #
    # GBIF calibration (same params as training — no leakage)            #
    # ------------------------------------------------------------------ #
    if args.no_gbif_calib:
        console.print(
            "\n[yellow]ABLATION: GBIF calibration disabled (identity — scale=1, shift=0)[/yellow]"
        )
        _zeros = np.zeros(N_TRAITS, dtype=np.float32)
        _ones = np.ones(N_TRAITS, dtype=np.float32)
        calibration = GBIFCalibration(
            gbif_mean=_zeros,
            gbif_std=_ones,
            splot_mean=_zeros,
            splot_std=_ones,
        )
    else:
        console.print("\nComputing GBIF calibration from train.zarr...")
        import zarr as _zarr_detect

        _train_target_keys = list(
            _zarr_detect.open(str(train_zarr), mode="r")["targets"].keys()
        )
        if (
            "comb" in _train_target_keys
            and "supervision_splot_only" not in _train_target_keys
        ):
            calibration = compute_gbif_calibration_patch15(train_zarr)
        elif (
            "splot" in _train_target_keys
            and "supervision_splot_only" not in _train_target_keys
        ):
            calibration = compute_gbif_calibration_luca(train_zarr)
        else:
            calibration = compute_gbif_calibration(train_zarr)

    # ------------------------------------------------------------------ #
    # Dataloader                                                          #
    # ------------------------------------------------------------------ #
    batch_size = args.batch_size or int(cfg.data_loaders.batch_size)
    num_workers = args.num_workers or int(cfg.data_loaders.num_workers)
    add_latlon = bool(OmegaConf.select(cfg, "data.add_latlon", default=False))

    test_loader = get_chips_dataloader(
        test_zarr,
        calibration,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        add_latlon=add_latlon,
    )
    console.print(f"  Test chips: [cyan]{len(test_loader.dataset):,}[/cyan]")
    console.print(f"  Batch size: [cyan]{batch_size}[/cyan]")
    # Center pixel index — 1 for 3×3, 2 for 5×5, 3 for 7×7
    center_idx = test_loader.dataset.center_idx
    console.print(
        f"  Patch size: [cyan]{test_loader.dataset.patch_size}[/cyan]  (center pixel [{center_idx},{center_idx}])"
    )

    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    model = instantiate(
        cfg.models,
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    model.load_state_dict(state["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[bold]Model[/bold]")
    console.print(f"  {cfg.models._target_}  ({n_params:,} params)")

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Running inference on test split...[/bold]")

    # Load chip center coordinates for map generation (bounds are in EPSG:3857)
    chip_lon: np.ndarray | None = None
    chip_lat: np.ndarray | None = None
    if args.maps:
        _store = _zarr_module.open_group(str(test_zarr), mode="r")
        _b = _store["bounds"][:]  # (N, 4) float64: [x_min, y_min, x_max, y_max]
        _cx = (_b[:, 0] + _b[:, 2]) / 2.0  # center easting
        _cy = (_b[:, 1] + _b[:, 3]) / 2.0  # center northing
        _R = 6378137.0  # WGS84 semi-major axis
        chip_lon = np.degrees(_cx / _R)
        chip_lat = np.degrees(2.0 * np.arctan(np.exp(_cy / _R)) - np.pi / 2.0)
        del _store, _b, _cx, _cy

    test_preds_list: list[torch.Tensor] = []
    test_labels_list: list[torch.Tensor] = []
    test_masks_list: list[torch.Tensor] = []
    map_preds_list: list[torch.Tensor] = []  # all center preds (for map)
    test_loss_num = 0.0
    test_loss_den = 0.0

    from ptev2.loss import WeightedMaskedDenseLoss

    splot_loss_fn = WeightedMaskedDenseLoss(
        error_type="smooth_l1", huber_delta=1.0, w_gbif=0.0, w_splot=1.0
    ).to(device)

    with torch.no_grad():
        for X, y_sv, y_ss, _y_gv, _y_gs in track(test_loader, description="Test"):
            X = X.to(device, dtype=torch.float32)
            y_sv = y_sv.to(device, dtype=torch.float32)
            y_ss = y_ss.to(device, dtype=torch.float32)

            pred_finite = torch.isfinite(X).all(dim=1, keepdim=True)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.clamp(X, -1e4, 1e4)
            y_ss = torch.where(
                pred_finite.expand_as(y_ss), y_ss, torch.zeros_like(y_ss)
            )

            pred = model(X)  # (B, 37, P, P)

            # Test loss (unweighted — no quality weight at test time)
            v_num, v_den = splot_loss_fn.loss_components(pred, y_sv, y_ss)
            if torch.isfinite(v_num) and v_den > 0:
                test_loss_num += v_num.item()
                test_loss_den += v_den.item()

            # Center-pixel collection for Pearson-r (or all SPLOT pixels with --eval_all_splot)
            if args.eval_all_splot:
                _B, _T, _P, _ = pred.shape
                pred_c = pred.permute(0, 2, 3, 1).reshape(
                    _B * _P * _P, _T
                )  # (B*P*P, 37)
                y_c = y_sv.permute(0, 2, 3, 1).reshape(_B * _P * _P, _T)
                src_c = y_ss.permute(0, 2, 3, 1).reshape(_B * _P * _P, _T)
            else:
                ci = center_idx
                pred_c = pred[:, :, ci, ci]  # (B, 37)
                y_c = y_sv[:, :, ci, ci]  # (B, 37)
                src_c = y_ss[:, :, ci, ci]  # (B, 37)
            msk = (src_c == 2) & torch.isfinite(y_c) & torch.isfinite(pred_c)

            if msk.any():
                test_preds_list.append(pred_c.cpu())
                test_labels_list.append(y_c.cpu())
                test_masks_list.append(msk.cpu())

            if args.maps:
                map_preds_list.append(pred_c.cpu())

    if not test_preds_list:
        raise RuntimeError(
            "No valid test observations found — check test.zarr and SPLOT coverage."
        )

    # ------------------------------------------------------------------ #
    # Metrics                                                             #
    # ------------------------------------------------------------------ #
    test_loss = test_loss_num / test_loss_den if test_loss_den > 0 else float("nan")
    per_trait_r = compute_pearson_r(test_preds_list, test_labels_list, test_masks_list)

    valid_r = [r for r in per_trait_r if math.isfinite(r)]
    macro_r = sum(valid_r) / len(valid_r) if valid_r else float("nan")

    # Also compute RMSE and MAE on center pixels for completeness
    P = torch.cat(test_preds_list, dim=0)
    G = torch.cat(test_labels_list, dim=0)
    M = torch.cat(test_masks_list, dim=0)

    per_trait_rmse: list[float] = []
    per_trait_mae: list[float] = []
    per_trait_n: list[int] = []
    for t in range(N_TRAITS):
        vt = M[:, t]
        n = int(vt.sum().item())
        per_trait_n.append(n)
        if n > 0:
            diff = P[vt, t].double() - G[vt, t].double()
            per_trait_rmse.append(float(diff.pow(2).mean().sqrt()))
            per_trait_mae.append(float(diff.abs().mean()))
        else:
            per_trait_rmse.append(float("nan"))
            per_trait_mae.append(float("nan"))

    valid_rmse = [v for v in per_trait_rmse if math.isfinite(v)]
    valid_mae = [v for v in per_trait_mae if math.isfinite(v)]
    macro_rmse = sum(valid_rmse) / len(valid_rmse) if valid_rmse else float("nan")
    macro_mae = sum(valid_mae) / len(valid_mae) if valid_mae else float("nan")

    # ------------------------------------------------------------------ #
    # Print results                                                       #
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold green]Test Results[/bold green]")
    console.print(f"  test_loss:  [cyan]{test_loss:.5f}[/cyan]")
    console.print(
        f"  macro_r:    [bold cyan]{macro_r:.4f}[/bold cyan]  (n_valid_traits={len(valid_r)}/{N_TRAITS})"
    )
    console.print(f"  macro_rmse: [cyan]{macro_rmse:.5f}[/cyan]")
    console.print(f"  macro_mae:  [cyan]{macro_mae:.5f}[/cyan]")

    console.print("\n  Per-trait Pearson-r:")
    for t, (name, r) in enumerate(zip(TRAIT_NAMES, per_trait_r)):
        n_samp = per_trait_n[t]
        marker = " ←LOW" if math.isfinite(r) and r < 0.3 else ""
        console.print(f"    {name:8s}: {r:+.3f}  (n={n_samp:,}){marker}")

    # ------------------------------------------------------------------ #
    # Save JSON                                                           #
    # ------------------------------------------------------------------ #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem

    results = {
        "checkpoint": str(checkpoint_path),
        "run_name": run_name,
        "epoch": state.get("epoch"),
        "val_mean_r_at_train": state.get("val_mean_r"),
        "val_loss_at_train": state.get("val_loss"),
        "test_split": str(test_zarr),
        "n_test_chips": len(test_loader.dataset),
        "test_loss": float(test_loss),
        "macro_r": float(macro_r),
        "macro_rmse": float(macro_rmse),
        "macro_mae": float(macro_mae),
        "n_valid_traits": len(valid_r),
        "per_trait": {
            TRAIT_NAMES[t]: {
                "r": per_trait_r[t],
                "rmse": per_trait_rmse[t],
                "mae": per_trait_mae[t],
                "n": per_trait_n[t],
            }
            for t in range(N_TRAITS)
        },
    }

    out_path = output_dir / f"{stem}.test_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"\n  Results saved: [cyan]{out_path}[/cyan]")

    # ---- Scores JSON: trait_id → Pearson-r for easy model comparison ----
    scores = {TRAIT_NAMES[t]: float(per_trait_r[t]) for t in range(N_TRAITS)}
    scores_path = output_dir / f"{stem}.scores.json"
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    console.print(f"  Scores saved:  [cyan]{scores_path}[/cyan]")

    # ---- Predicted trait maps (optional: --maps flag) ------------------
    if args.maps and map_preds_list and chip_lon is not None:
        console.print("\nGenerating trait maps...")
        _generate_trait_maps(
            map_preds_list,
            chip_lon,
            chip_lat,
            per_trait_r,
            output_dir,
            stem,
            console,
        )

    # ---- Global GeoTIF (optional: --tif flag) --------------------------
    if args.tif:
        if args.all_zarr:
            all_zarr_path = Path(args.all_zarr)
        else:
            # Auto-detect: first check same dir as train/val/test (chips_centered 3×3),
            # then fall back to the legacy patch15 path.
            candidate_3x3 = zarr_dir / "all.zarr"
            candidate_15 = (
                Path(cfg.data.root_dir) / "chips" / "patch15_stride10" / "all.zarr"
            )
            if candidate_3x3.exists():
                all_zarr_path = candidate_3x3
            elif candidate_15.exists():
                all_zarr_path = candidate_15
                console.print(
                    "[yellow]Using patch15 all.zarr — run make_all_zarr.py for full coverage.[/yellow]"
                )
            else:
                all_zarr_path = candidate_3x3  # will trigger "not found" error below

        if not all_zarr_path.exists():
            console.print(
                f"[red]all.zarr not found at {all_zarr_path}.\n"
                f"Provide the correct path with --all_zarr.[/red]"
            )
        else:
            tif_path = output_dir / f"{stem}_global_map.tif"
            _generate_global_tif(
                model,
                all_zarr_path,
                tif_path,
                device,
                batch_size,
                console,
            )

    # ------------------------------------------------------------------ #
    # W&B (optional)                                                      #
    # ------------------------------------------------------------------ #
    if args.wandb:
        try:
            import wandb
        except ImportError:
            console.print(
                "[yellow]W&B requested but not installed — skipping.[/yellow]"
            )
        else:
            group = str(OmegaConf.select(cfg, "train.group") or "chips")
            wandb.init(
                project=str(cfg.wandb.project),
                entity=str(cfg.wandb.entity),
                name=f"{stem}_test",
                group=group,
                config={
                    "checkpoint": str(checkpoint_path),
                    "run_name": run_name,
                    "eval_split": "test",
                },
                reinit=True,
            )
            wandb.log(
                {
                    "test/macro_r": macro_r,
                    "test/macro_rmse": macro_rmse,
                    "test/macro_mae": macro_mae,
                    "test/loss": test_loss,
                    "test/n_valid_traits": len(valid_r),
                    "test/per_trait_r": {
                        TRAIT_NAMES[t]: per_trait_r[t]
                        for t in range(N_TRAITS)
                        if math.isfinite(per_trait_r[t])
                    },
                }
            )
            wandb.finish()
            console.print("  W&B logged.")


if __name__ == "__main__":
    main()
