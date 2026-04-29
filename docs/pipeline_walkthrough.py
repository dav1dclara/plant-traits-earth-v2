"""
pipeline_walkthrough.py
=======================
A self-contained, heavily-commented walkthrough of the full plant trait
prediction pipeline. Run any section independently to see what it does.

This file is meant to be READ alongside the doc — every major design choice
is explained inline. Nothing here trains a real model; it demonstrates the
shapes, logic and computations with real data where possible.

Usage:
    conda activate PTEV2
    python docs/pipeline_walkthrough.py

Sections:
    1. Data layout     — what's in the zarr files
    2. Label format    — how SPLOT/GBIF labels are stored
    3. GBIF calibration — how we align GBIF to SPLOT space
    4. Dataset          — how ChipsCenteredDataset works
    5. Loss function    — how the masked Huber loss works
    6. Models           — shapes through each architecture
    7. Evaluation metric — how Pearson-r is computed
"""

import math
import sys

sys.path.insert(0, "src")

import numpy as np
import torch
import zarr

TRAIN_ZARR = (
    "/scratch3/plant-traits-v2/data/22km/chips_centered/patch3_stride1/train.zarr"
)
VAL_ZARR = "/scratch3/plant-traits-v2/data/22km/chips_centered/patch3_stride1/val.zarr"
TEST_ZARR = (
    "/scratch3/plant-traits-v2/data/22km/chips_centered/patch3_stride1/test.zarr"
)

DIVIDER = "\n" + "=" * 70 + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — What's Inside the zarr Files
# ─────────────────────────────────────────────────────────────────────────────
def section1_data_layout():
    print(DIVIDER + "SECTION 1 — Data layout inside zarr\n")

    store = zarr.open_group(TRAIN_ZARR, mode="r")

    # ── Top-level groups ──────────────────────────────────────────────────
    # Each zarr file has these top-level groups:
    #   predictors/ — the 5 satellite/environmental data sources
    #   targets/    — the plant trait labels (SPLOT and GBIF)
    #   center/     — summary statistics at the center pixel only
    #   bounds      — bounding box of each chip in EPSG:3857 (for making maps)
    #   row, col    — grid indices of each chip in the global raster

    print("Top-level groups:", list(store.keys()))

    # ── Predictors ────────────────────────────────────────────────────────
    # 5 data sources stacked as separate arrays.
    # Each has shape (N_chips, C_channels, 3, 3).
    # "3×3" = a 3-pixel-wide spatial window centered on the observation.
    print("\nPredictor arrays:")
    predictor_info = {
        "canopy_height": " 2 ch — tree canopy height from lidar-derived product",
        "modis": "72 ch — vegetation reflectance time series (72 time steps)",
        "soil_grids": "61 ch — soil properties at multiple depths (SoilGrids)",
        "vodca": " 9 ch — vegetation optical depth (microwave, C/X/Ku bands)",
        "worldclim": " 6 ch — long-term climate statistics (temp, precip...)",
    }
    total_channels = 0
    for name, desc in predictor_info.items():
        arr = store[f"predictors/{name}"]
        total_channels += arr.shape[1]
        print(f"  {name:15s}: shape={arr.shape}  # {desc}")
    print(f"\n  Total input channels: {total_channels}  (= 2+72+61+9+6)")

    # ── Targets ───────────────────────────────────────────────────────────
    print("\nTarget arrays:")
    for key in ["supervision_splot_only", "supervision_gbif_only", "supervision"]:
        arr = store[f"targets/{key}"]
        print(f"  targets/{key}: shape={arr.shape}")
    print("  → 74 channels = 37 traits × 2 (value + source_mask interleaved)")

    # ── Center statistics ──────────────────────────────────────────────────
    print("\nCenter statistics (per chip, per trait):")
    for key in sorted(store["center"].keys()):
        arr = store[f"center/{key}"]
        print(f"  center/{key}: shape={arr.shape} dtype={arr.dtype}")

    # ── Dataset sizes ──────────────────────────────────────────────────────
    print("\nDataset sizes:")
    for name, path in [("TRAIN", TRAIN_ZARR), ("VAL", VAL_ZARR), ("TEST", TEST_ZARR)]:
        s = zarr.open_group(path, mode="r")
        n = s["targets/supervision_splot_only"].shape[0]
        # Count chips that have any SPLOT data at the center pixel
        splot_valid = s["center/splot_valid"][:]
        has_splot = int((splot_valid.sum(axis=1) > 0).sum())
        print(
            f"  {name}: {n:,} chips, {has_splot:,} with SPLOT center ({100 * has_splot / n:.0f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — The Label Format (Interleaved val+source)
# ─────────────────────────────────────────────────────────────────────────────
def section2_label_format():
    print(DIVIDER + "SECTION 2 — How labels are stored (interleaved format)\n")

    store = zarr.open_group(TRAIN_ZARR, mode="r")

    # Grab the first chip that has SPLOT data at center
    splot_valid = store["center/splot_valid"][:]  # (N, 37)
    idx = int(np.where(splot_valid.sum(axis=1) > 0)[0][0])
    print(f"Using chip index: {idx}")

    # Load the SPLOT supervision for this chip
    sv = store["targets/supervision_splot_only"][idx]  # (74, 3, 3)
    print(f"\nsv shape: {sv.shape}  (74 = 37 traits × 2)")

    # The 74 channels are interleaved: [val_t0, src_t0, val_t1, src_t1, ...]
    y_values = sv[0::2]  # channels 0, 2, 4, ..., 72  → shape (37, 3, 3)
    y_source = sv[1::2]  # channels 1, 3, 5, ..., 73  → shape (37, 3, 3)

    print(f"\ny_values (trait values): shape={y_values.shape}")
    print(f"y_source (source mask):  shape={y_source.shape}")

    # Source codes:
    #   0 = no data at this pixel
    #   1 = GBIF observation
    #   2 = SPLOT observation
    print("\nSource mask at center pixel [1,1] (first 5 traits):")
    for t in range(5):
        src = int(y_source[t, 1, 1])
        val = float(y_values[t, 1, 1])
        label = {0: "no data", 1: "GBIF", 2: "SPLOT"}.get(src, "?")
        print(f"  trait {t}: source={src} ({label}), value={val:.4f}")

    # Show the 3x3 spatial pattern of sources for one trait
    print(f"\nSource mask for trait 0 (3×3 spatial window):")
    for row in range(3):
        print("  ", y_source[0, row, :].tolist())
    print("  (2=SPLOT at nearby pixels, 0=no data)")

    # This matters for the loss: the model is only trained on pixels where source > 0
    n_splot = (y_source == 2).sum()
    n_gbif = (y_source == 1).sum()
    n_none = (y_source == 0).sum()
    print(f"\nFor this chip: SPLOT pixels={n_splot}, GBIF={n_gbif}, no-data={n_none}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GBIF Calibration (Idea 2)
# ─────────────────────────────────────────────────────────────────────────────
def section3_gbif_calibration():
    print(DIVIDER + "SECTION 3 — GBIF calibration (aligning GBIF to SPLOT space)\n")

    from ptev2.data.chips_dataset import TRAIT_NAMES, compute_gbif_calibration

    # The problem: at co-located pixels, GBIF and SPLOT measure the same trait
    # but GBIF values have a different mean and scale (r≈0.33 before calibration).
    # We fit a per-trait linear transform: GBIF_calibrated = scale * GBIF_raw + shift
    # such that GBIF_calibrated has the same mean and std as SPLOT.

    print("Fitting GBIF calibration on training data...")
    print("(This reads the entire train.zarr center/ arrays — ~11s one-time cost)")
    calib = compute_gbif_calibration(TRAIN_ZARR)

    # Show first few traits
    print(f"\nPer-trait linear calibration (first 5 traits):")
    print(f"{'Trait':10s}  {'scale':8s}  {'shift':8s}")
    for t in range(5):
        scale = float(calib.scale[t])
        shift = float(calib.shift[t])
        print(f"  {TRAIT_NAMES[t]:10s}  {scale:+.4f}    {shift:+.4f}")

    print("\nInterpretation:")
    print("  scale ≈ splot_std / gbif_std — stretches/compresses GBIF variance")
    print("  shift = splot_mean - scale * gbif_mean — shifts mean to match SPLOT")
    print("  After calibration: max Δmean across 37 traits ≈ 1.5e-8 (essentially 0)")
    print(
        "  GBIF is still noisy — calibration aligns distribution, not individual values"
    )
    print("\nGBIF is used with weight=0.15 in training:")
    print("  total_loss = splot_loss + 0.15 * gbif_loss")
    print("  → GBIF provides spatial coverage; SPLOT drives target distribution")

    return calib


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Dataset and DataLoader
# ─────────────────────────────────────────────────────────────────────────────
def section4_dataset(calib):
    print(DIVIDER + "SECTION 4 — ChipsCenteredDataset\n")

    import time

    from ptev2.data.chips_dataset import N_TRAITS, get_chips_dataloader

    # The dataset pre-loads ALL zarr arrays into RAM at __init__.
    # This is the key speedup: no disk I/O during training.
    print("Creating DataLoader (will pre-load ~600 MB into RAM)...")
    t0 = time.time()
    dl = get_chips_dataloader(
        TRAIN_ZARR, calib, batch_size=128, num_workers=4, shuffle=True
    )
    t1 = time.time()
    print(f"Init time: {t1 - t0:.1f}s  (one-off cost at startup)")
    print(f"Dataset size: {len(dl.dataset):,} chips")
    print(f"Batches per epoch: {len(dl)}")

    # Get one batch and inspect shapes
    print("\nFetching one batch...")
    t0 = time.time()
    X, y_sv, y_ss, y_gv, y_gs = next(iter(dl))
    t1 = time.time()
    print(f"First batch time: {(t1 - t0) * 1000:.0f} ms")

    print(f"\nBatch tensor shapes:")
    print(
        f"  X    (predictors):           {tuple(X.shape)}  — (batch, 150 channels, 3×3)"
    )
    print(
        f"  y_sv (SPLOT values):         {tuple(y_sv.shape)}  — (batch, 37 traits, 3×3)"
    )
    print(
        f"  y_ss (SPLOT source mask):    {tuple(y_ss.shape)}  — (batch, 37 traits, 3×3)"
    )
    print(
        f"  y_gv (GBIF values, calib):   {tuple(y_gv.shape)}  — (batch, 37 traits, 3×3)"
    )
    print(
        f"  y_gs (GBIF source mask):     {tuple(y_gs.shape)}  — (batch, 37 traits, 3×3)"
    )

    # SPLOT coverage in this batch
    n_splot = (y_ss == 2).sum().item()
    n_total = y_ss.numel()
    print(
        f"\nSPLOT pixels in batch: {n_splot:,} / {n_total:,} = {100 * n_splot / n_total:.1f}%"
    )
    print("(Expect ~34% of chips to have SPLOT at center, spread across all 9 pixels)")

    # Time 20 batches to estimate epoch time
    print("\nTiming 20 batches to estimate epoch speed...")
    t0 = time.time()
    for i, _ in enumerate(dl):
        if i >= 20:
            break
    t1 = time.time()
    print(f"20 batches: {t1 - t0:.2f}s = {(t1 - t0) / 20 * 1000:.0f}ms/batch")
    print(
        f"Projected epoch: {(t1 - t0) / 20 * len(dl) / 60:.2f} min  (was 3-4 min before pre-loading)"
    )

    return dl


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — The Loss Function
# ─────────────────────────────────────────────────────────────────────────────
def section5_loss():
    print(DIVIDER + "SECTION 5 — WeightedMaskedDenseLoss\n")

    from ptev2.loss import WeightedMaskedDenseLoss

    # ── What is smooth L1 / Huber loss? ───────────────────────────────────
    # For a single prediction-label pair:
    #   if |pred - label| < delta:   loss = 0.5 * (pred - label)^2  (quadratic)
    #   else:                         loss = delta * (|pred - label| - 0.5*delta)  (linear)
    # This is more robust than MSE — doesn't blow up for occasional large errors.

    print("Smooth L1 (Huber) loss with delta=1.0:")
    errors = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    delta = 1.0
    smooth_l1 = torch.where(
        errors.abs() < delta, 0.5 * errors.pow(2), delta * (errors.abs() - 0.5 * delta)
    )
    for e, l in zip(errors.tolist(), smooth_l1.tolist()):
        print(f"  error={e:+.1f} → loss={l:.4f}")

    # ── What is the masking? ───────────────────────────────────────────────
    print("\nMasked loss concept:")
    print("  Only pixels with source_mask > 0 contribute to the loss.")
    print("  A pixel with no label (source=0) is completely ignored.")
    print("  This is critical because most pixels in each chip have no label.")

    # Demonstrate with a small example
    splot_loss = WeightedMaskedDenseLoss(
        error_type="smooth_l1", huber_delta=1.0, w_gbif=0.0, w_splot=1.0
    )

    # Fake batch: 2 samples, 2 traits, 2×2 spatial
    B, T, H, W = 2, 2, 2, 2
    pred = torch.zeros(B, T, H, W)
    target = torch.ones(B, T, H, W)  # all labels = 1.0, predictions = 0.0
    source = torch.tensor(
        [
            # sample 0: trait0 has SPLOT at (0,0), trait1 has no data
            # sample 1: both traits have SPLOT everywhere
            [[[2, 0], [0, 0]], [[0, 0], [0, 0]]],  # sample 0
            [[[2, 2], [2, 2]], [[2, 2], [2, 2]]],  # sample 1
        ],
        dtype=torch.float32,
    )

    num, den = splot_loss.loss_components(pred, target, source)
    print(f"\n  Example: pred=0, target=1 everywhere, but source varies")
    print(
        f"  loss = {num.item():.4f} / {den.item():.0f} = {num.item() / den.item():.4f}"
    )
    print(f"  (den = number of valid SPLOT pixels = 1 + 8 = 9)")
    print(f"  (error = |0 - 1| = 1.0 → smooth_l1(1.0, delta=1.0) = 0.5 × 1² = 0.5)")
    print(
        f"  Expected: 9 × 0.5 / 9 = 0.5 ✓"
        if abs(num.item() / den.item() - 0.5) < 1e-4
        else "  ✗ check logic"
    )

    # ── The combined loss ─────────────────────────────────────────────────
    print("\nCombined training loss:")
    print("  total_loss = splot_loss + 0.15 × gbif_loss")
    print("  Both use WeightedMaskedDenseLoss but on different supervision arrays.")
    print(
        "  GBIF weight 0.15: enough to provide spatial gradient, not enough to distort SPLOT distribution."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Models
# ─────────────────────────────────────────────────────────────────────────────
def section6_models():
    print(DIVIDER + "SECTION 6 — Model architectures\n")

    from ptev2.data.chips_dataset import N_TRAITS
    from ptev2.models.multitask_v2 import GatedMMoEModelV3, MTLModel, STLModel

    IN_CH = 150  # 2+72+61+9+6 predictor channels

    def summarise(model, name):
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.zeros(1, IN_CH, 3, 3)
        with torch.no_grad():
            y = model(x)
        print(f"  {name}:")
        print(f"    params:       {n_params:,}")
        print(f"    input shape:  {tuple(x.shape)}")
        print(f"    output shape: {tuple(y.shape)}  (batch, 37 traits, 3×3)")

    # ── STL ────────────────────────────────────────────────────────────────
    print("STL — Single Task Learning:")
    print("  Architecture: SharedEncoder → single 1×1 conv head for all 37 traits")
    print("  SharedEncoder: 4 ResidualBlocks (150→32→32→64→128 channels)")
    stl = STLModel(in_channels=IN_CH, out_channels=N_TRAITS)
    summarise(stl, "STL")

    # ── MTL ────────────────────────────────────────────────────────────────
    print("\nMTL — Multi Task Learning:")
    print("  Architecture: SharedEncoder → shared 3×3 conv → 37 separate 1×1 heads")
    print(
        "  Each trait gets its own prediction head but shares all upstream computation"
    )
    mtl = MTLModel(in_channels=IN_CH, out_channels=N_TRAITS)
    summarise(mtl, "MTL")

    # ── MMoE V3 ────────────────────────────────────────────────────────────
    print("\nMMoE V3 — Mixture of Experts:")
    print("  Architecture: SharedEncoder → 8 Expert networks + gating → 37 heads")
    print(
        "  Gating: 4 trait groups (biologically motivated) each have a gating network"
    )
    print(
        "  that learns to route to the most relevant experts for that group of traits"
    )
    print("  Expert networks: small MLPs operating on 128-dim encoder features")
    mmoe = GatedMMoEModelV3(in_channels=IN_CH, out_channels=N_TRAITS)
    summarise(mmoe, "MMoE V3")

    # ── Forward pass walkthrough (STL) ─────────────────────────────────────
    print("\nSTL forward pass — shape trace:")
    x = torch.zeros(4, IN_CH, 3, 3)
    print(f"  Input:            {tuple(x.shape)}")
    z = stl.encoder.block1(stl.encoder(x) if False else x)  # skip for brevity
    # Walk manually
    x_ = x
    if stl.encoder.input_pca is not None:
        x_ = stl.encoder.input_pca(x_)
    z1 = stl.encoder.block1(x_)
    print(f"  After ResBlock1:  {tuple(z1.shape)}  (150→32)")
    z2 = stl.encoder.block2(z1)
    print(f"  After ResBlock2:  {tuple(z2.shape)}  (32→32)")
    z3 = stl.encoder.block3(z2)
    print(f"  After ResBlock3:  {tuple(z3.shape)}  (32→64)")
    z4 = stl.encoder.block4(z3)
    print(f"  After ResBlock4:  {tuple(z4.shape)}  (64→128)")
    out = stl.head(z4)
    print(f"  After head:       {tuple(out.shape)}  (128→37)")
    print("  Note: spatial size stays 3×3 throughout (stride=1 everywhere)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Evaluation Metric (Pearson-r)
# ─────────────────────────────────────────────────────────────────────────────
def section7_metric():
    print(DIVIDER + "SECTION 7 — Evaluation: center-pixel Pearson-r\n")

    print("What we measure:")
    print(
        "  For each test chip, take pred[:, 1, 1] — the center pixel prediction (37 values)."
    )
    print(
        "  Only use chips where center pixel has a valid SPLOT label (source_mask == 2)."
    )
    print(
        "  For each of the 37 traits, compute Pearson correlation over all valid chips."
    )
    print("  Average across traits → macro-r (our primary reported number).")

    print("\nWhy Pearson-r and not RMSE?")
    print("  Pearson-r measures whether the model correctly ranks locations")
    print(
        "  (which places have higher/lower trait values) regardless of absolute scale."
    )
    print(
        "  This is what matters for global maps — we want spatial patterns to be right."
    )
    print(
        "  RMSE is sensitive to scale, which could be off if trait units aren't matched."
    )

    print("\nThe formula (what we compute in code):")
    print("  r = (Σ (pred_i - mean_pred)(label_i - mean_label))")
    print("      / (sqrt(Σ(pred_i - mean_pred)²) * sqrt(Σ(label_i - mean_label)²))")

    # Demonstrate with a simple example
    N = 1000
    torch.manual_seed(42)
    labels = torch.randn(N)
    noise = torch.randn(N) * 0.5
    preds_good = labels + noise  # correlated predictions
    preds_bad = torch.randn(N)  # random predictions

    def pearson_r(p, g):
        p_m = p - p.mean()
        g_m = g - g.mean()
        return float(
            (p_m * g_m).sum()
            / (p_m.pow(2).sum().sqrt() * g_m.pow(2).sum().sqrt() + 1e-12)
        )

    print(f"\n  Example (N={N} samples):")
    print(
        f"  Good predictions (label + small noise): r = {pearson_r(preds_good, labels):+.4f}"
    )
    print(
        f"  Random predictions:                     r = {pearson_r(preds_bad, labels):+.4f}"
    )

    print("\nWhy float64? — Catastrophic cancellation bug (now fixed)")
    print("  The naive running-sum formula: r = (N Σxy - Σx Σy) / sqrt(...)")
    print(
        "  suffers from large cancellation when Σxy ≈ Σx·Σy (which happens at convergence)."
    )
    print(
        "  We use the list-accumulation approach: collect all pred/label values, then compute"
    )
    print("  mean-centered dot products in float64. This is numerically stable.")

    print("\nVal vs Test:")
    print(
        "  Val  set: 100% SPLOT center → used during training for early stopping + checkpointing"
    )
    print(
        "  Test set: 100% SPLOT center → NEVER touched during training; only evaluated once at the end"
    )
    print("  Train set: 34% SPLOT center → primary signal; rest is GBIF-only chips")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run all sections
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPlant Trait Prediction Pipeline — Walkthrough")
    print("=" * 70)
    print("Each section explains one part of the pipeline with real data/shapes.")

    section1_data_layout()
    section2_label_format()
    calib = section3_gbif_calibration()
    dl = section4_dataset(calib)
    section5_loss()
    section6_models()
    section7_metric()

    print(DIVIDER + "DONE — all sections completed.\n")
    print("For training, run:")
    print(
        "  python scripts/train_chips.py train.run_name=chips_mmoe wandb.enabled=true"
    )
    print("\nFor evaluation, run:")
    print("  python scripts/test_chips.py --checkpoint <path> --maps")
