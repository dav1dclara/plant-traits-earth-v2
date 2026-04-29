"""Training script for Ideas 1+2: SPLOT-only primary loss + calibrated GBIF auxiliary.

Uses chips_centered/patch3_stride1 zarr (3×3 spatial patches at 22km = 66km context).

Key differences from trainv2.py:
    1. Data: chips_centered zarr (18,487 SPLOT train chips vs 166 in patch15 — 111×)
    2. Primary loss: SPLOT-only supervision (supervision_splot_only). Zero source mismatch.
    3. Aux loss: linearly-calibrated GBIF supervision at weight aux_gbif_weight (default 0.15).
       Calibration parameters are computed once from train.zarr (no data leakage).
    4. Val metric: center-pixel SPLOT Pearson-r (center pixel index [1,1]).
       All val chips are 100% SPLOT-center, so this is always well-defined.
    5. Model: same GatedMMoEModelV3 / MTLModel / STLModel (in_channels=150, out_channels=37).
       No architecture changes needed — stride_blocks=[1,1,1,1] keeps spatial dim 3×3.

Usage:
    # MMoE V3 (default)
    python scripts/train_chips.py

    # STL baseline
    python scripts/train_chips.py models=respatch_v2 train.run_name=chips_stl

    # MTL
    python scripts/train_chips.py models=mtl_v2 train.run_name=chips_mtl

    # with W&B + group
    python scripts/train_chips.py wandb.enabled=true train.group=chips_ideas12

    # override aux weight
    python scripts/train_chips.py train.aux_gbif_weight=0.0 train.run_name=chips_splot_only
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Rich console / progress (optional dep, graceful fallback)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.progress import track
except ImportError:

    class Console:  # type: ignore[override]
        def print(self, *args, **kwargs) -> None:
            print(*args)

        def rule(self, text: str) -> None:
            print(f"\n{text}")

    def track(iterable, description: str = ""):
        if description:
            print(description)
        return iterable


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


# ---------------------------------------------------------------------------
# PCA helper (reused pattern from trainv2.py)
# ---------------------------------------------------------------------------


def _fit_pca(
    train_loader,
    in_channels: int,
    n_components: int,
    device: torch.device,
    max_pixels: int = 3_000_000,
):
    """Fit PCA on training predictor data via incremental covariance accumulation."""
    console.print(
        f"Fitting PCA: {in_channels} → {n_components} components (max {max_pixels:,} pixels)..."
    )
    cov_sum = torch.zeros(in_channels, in_channels, dtype=torch.float64)
    mean_sum = torch.zeros(in_channels, dtype=torch.float64)
    n_total = 0

    for X, *_ in train_loader:
        B, C, H, W = X.shape
        pixels = X.permute(0, 2, 3, 1).reshape(-1, C).double()
        valid = torch.isfinite(pixels).all(dim=1)
        pixels = pixels[valid]
        pixels = torch.clamp(pixels, -1e4, 1e4)
        if pixels.shape[0] == 0:
            continue
        mean_sum += pixels.sum(dim=0)
        cov_sum += pixels.T @ pixels
        n_total += pixels.shape[0]
        if n_total >= max_pixels:
            break

    if n_total < n_components + 1:
        raise RuntimeError(
            f"PCA fitting failed: only {n_total} valid pixels collected."
        )

    data_mean = mean_sum / n_total
    cov = cov_sum / n_total - data_mean.unsqueeze(1) * data_mean.unsqueeze(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    components = eigenvectors[:, -n_components:].T.contiguous().float()
    explained = (
        eigenvalues[-n_components:].sum() / eigenvalues.clamp(min=0).sum()
    ).item()
    console.print(
        f"  PCA done: top-{n_components} explain {100 * explained:.1f}% "
        f"({n_total:,} pixels used)"
    )
    return components, data_mean.float(), explained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="training/chips_v1", version_base=None)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TRAIN CHIPS — Ideas 1+2[/bold cyan]")

    # ------------------------------------------------------------------ #
    # 1. Setup                                                            #
    # ------------------------------------------------------------------ #
    seed_all(cfg.train.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    console.print(f"Device: [cyan]{device}[/cyan]")

    # ------------------------------------------------------------------ #
    # 2. Data paths                                                       #
    # ------------------------------------------------------------------ #
    console.print("[bold]\nData[/bold]")
    zarr_dir = Path(cfg.data.root_dir) / cfg.data.zarr_dir
    assert zarr_dir.exists(), f"Zarr directory does not exist: {zarr_dir}"
    console.print(f"Zarr dir: [cyan]{zarr_dir}[/cyan]")

    train_zarr = zarr_dir / "train.zarr"
    val_zarr = zarr_dir / "val.zarr"

    for p in (train_zarr, val_zarr):
        assert p.exists(), f"Split zarr missing: {p}"

    # ------------------------------------------------------------------ #
    # 3. GBIF calibration (train split only — no leakage)                #
    # ------------------------------------------------------------------ #
    # Auto-detect zarr format by inspecting target keys.
    import zarr as _zarr_detect

    _train_target_keys = list(
        _zarr_detect.open(str(train_zarr), mode="r")["targets"].keys()
    )
    _is_patch15 = (
        "comb" in _train_target_keys
        and "supervision_splot_only" not in _train_target_keys
    )
    _is_luca = (
        "splot" in _train_target_keys
        and "supervision_splot_only" not in _train_target_keys
        and "comb" not in _train_target_keys
    )

    gbif_calib_enabled = bool(OmegaConf.select(cfg, "train.gbif_calib", default=True))
    if gbif_calib_enabled:
        console.print("\n[bold]Computing GBIF\u2192SPLOT calibration...[/bold]")
        if _is_patch15:
            calibration = compute_gbif_calibration_patch15(train_zarr)
        elif _is_luca:
            calibration = compute_gbif_calibration_luca(train_zarr)
        else:
            calibration = compute_gbif_calibration(train_zarr)
        deltas = calibration.delta_means()
        max_delta = float(np.abs(deltas).max()) if len(deltas) else 0.0
        console.print(
            f"  Calibration fitted. Max \u0394mean across 37 traits: [cyan]{max_delta:.2e}[/cyan]"
            + (" \u2713" if max_delta < 1e-4 else " (warning: larger than expected)")
        )
    else:
        console.print(
            "\n[yellow]ABLATION: GBIF calibration disabled (identity \u2014 scale=1, shift=0)[/yellow]"
        )
        _zeros = np.zeros(N_TRAITS, dtype=np.float32)
        _ones = np.ones(N_TRAITS, dtype=np.float32)
        calibration = GBIFCalibration(
            gbif_mean=_zeros,
            gbif_std=_ones,
            splot_mean=_zeros,
            splot_std=_ones,
        )

    # ------------------------------------------------------------------ #
    # 4. Dataloaders                                                      #
    # ------------------------------------------------------------------ #
    batch_size = cfg.data_loaders.batch_size
    num_workers = cfg.data_loaders.num_workers
    add_latlon = bool(OmegaConf.select(cfg, "data.add_latlon", default=False))
    eval_all_splot = bool(OmegaConf.select(cfg, "train.eval_all_splot", default=False))
    console.print(f"\n[bold]Dataloaders[/bold]")
    console.print(f"  Batch size:  [cyan]{batch_size}[/cyan]")
    console.print(f"  Num workers: [cyan]{num_workers}[/cyan]")
    console.print(f"  add_latlon:  [cyan]{add_latlon}[/cyan]")
    console.print(f"  eval_all_splot: [cyan]{eval_all_splot}[/cyan]")

    train_loader = get_chips_dataloader(
        train_zarr,
        calibration,
        batch_size,
        num_workers,
        shuffle=True,
        add_latlon=add_latlon,
    )
    val_loader = get_chips_dataloader(
        val_zarr,
        calibration,
        batch_size,
        num_workers,
        shuffle=False,
        add_latlon=add_latlon,
    )

    # Quick shape check from the first batch
    _X, _ys, _yss, _yg, _ygs = next(iter(train_loader))
    in_channels = _X.shape[1]
    assert in_channels == train_loader.dataset.in_channels, (
        f"Channel mismatch: got {in_channels}, expected {train_loader.dataset.in_channels}"
    )
    assert _ys.shape[1] == N_TRAITS, (
        f"SPLOT values channels: got {_ys.shape[1]}, expected {N_TRAITS}"
    )
    assert _yss.shape[1] == N_TRAITS, (
        f"SPLOT src channels: got {_yss.shape[1]}, expected {N_TRAITS}"
    )
    console.print(f"  Predictor shape (C,H,W): [cyan]{tuple(_X.shape[1:])}[/cyan]")
    console.print(f"  Label shape    (T,H,W):  [cyan]{tuple(_ys.shape[1:])}[/cyan]")
    console.print(f"  Train chips: [cyan]{len(train_loader.dataset):,}[/cyan]")
    console.print(f"  Val chips:   [cyan]{len(val_loader.dataset):,}[/cyan]")
    # Center pixel index — 1 for 3×3, 2 for 5×5, 3 for 7×7
    center_idx = train_loader.dataset.center_idx
    console.print(
        f"  Patch size: [cyan]{train_loader.dataset.patch_size}[/cyan]  (center pixel index: [{center_idx},{center_idx}])"
    )
    del _X, _ys, _yss, _yg, _ygs

    # ------------------------------------------------------------------ #
    # 5. Model                                                            #
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Model[/bold]")
    model = instantiate(
        cfg.models,
        in_channels=in_channels,
        out_channels=N_TRAITS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Target:      [cyan]{cfg.models._target_}[/cyan]")
    console.print(f"  In channels: [cyan]{in_channels}[/cyan]")
    console.print(f"  Out channels:[cyan]{N_TRAITS}[/cyan]")
    console.print(f"  Parameters:  [cyan]{n_params:,}[/cyan]")

    # Optional PCA projection
    pca_n_components = OmegaConf.select(cfg, "models.pca_n_components", default=None)
    if (
        pca_n_components is not None
        and hasattr(model, "encoder")
        and getattr(model.encoder, "input_pca", None) is not None
    ):
        components, data_mean, explained = _fit_pca(
            train_loader, in_channels, int(pca_n_components), device
        )
        model.encoder.input_pca.init_from_pca(
            components.to(device), data_mean.to(device)
        )
        console.print(
            f"  PCA: {in_channels} → {pca_n_components} "
            f"({100 * explained:.1f}% variance)"
        )

    # ------------------------------------------------------------------ #
    # 6. Loss functions                                                   #
    # ------------------------------------------------------------------ #
    # SPLOT primary: only source=2 pixels contribute.
    # If uncertainty weighting is enabled, use UncertaintyWeightedMTLLoss
    # instead of the fixed WeightedMaskedDenseLoss. This adds 37 learnable
    # log-variance params so harder traits are automatically down-weighted.
    uncertainty_enabled = bool(
        OmegaConf.select(cfg, "train.uncertainty.enabled", default=False)
    )
    if uncertainty_enabled:
        from ptev2.loss import UncertaintyWeightedMTLLoss

        init_log_sigma_sq = float(
            OmegaConf.select(cfg, "train.uncertainty.init_log_sigma_sq", default=0.0)
        )
        splot_loss_fn = UncertaintyWeightedMTLLoss(
            n_traits=N_TRAITS,
            w_gbif=0.0,
            w_splot=1.0,
            init_log_sigma_sq=init_log_sigma_sq,
        ).to(device)
        console.print(
            f"  SPLOT primary:  [bold green]UncertaintyWeightedMTLLoss[/bold green]  (37 learnable σ²)"
        )
    else:
        splot_loss_fn = instantiate(cfg.train.splot_loss).to(device)
        console.print(
            f"  SPLOT primary:  [cyan]{cfg.train.splot_loss._target_}[/cyan]  (w_splot={cfg.train.splot_loss.w_splot})"
        )
    # GBIF auxiliary: only source=1 pixels contribute (after calibration).
    gbif_loss_fn = instantiate(cfg.train.gbif_loss).to(device)
    aux_gbif_weight = float(cfg.train.aux_gbif_weight)

    console.print(
        f"  GBIF auxiliary: [cyan]{cfg.train.gbif_loss._target_}[/cyan]   (w_gbif={cfg.train.gbif_loss.w_gbif})"
    )
    console.print(f"  aux_gbif_weight: [cyan]{aux_gbif_weight}[/cyan]")

    # ------------------------------------------------------------------ #
    # 7. Optimizer & Scheduler                                            #
    # ------------------------------------------------------------------ #
    # If uncertainty loss is active, include its learnable σ² params in optimizer.
    trainable_params = list(model.parameters())
    if uncertainty_enabled and hasattr(splot_loss_fn, "log_sigma_sq"):
        trainable_params.append(splot_loss_fn.log_sigma_sq)
    optimizer = instantiate(cfg.train.optimizer, params=trainable_params)
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    grad_clip_norm = float(cfg.train.gradient_clip_norm)

    console.print(
        f"  Optimizer:  [cyan]{cfg.train.optimizer._target_}[/cyan]  lr={cfg.train.optimizer.lr}"
    )
    console.print(f"  Scheduler:  [cyan]{cfg.train.scheduler._target_}[/cyan]")
    console.print(f"  Grad clip:  [cyan]{grad_clip_norm}[/cyan]")

    # ------------------------------------------------------------------ #
    # 8. Early stopping & checkpointing                                   #
    # ------------------------------------------------------------------ #
    es_enabled = bool(cfg.train.early_stopping.enabled)
    es_patience = int(cfg.train.early_stopping.patience)
    es_min_delta = float(cfg.train.early_stopping.min_delta)
    console.print(
        f"  Early stopping: [cyan]{es_enabled}[/cyan]  patience={es_patience}"
    )

    # ------------------------------------------------------------------ #
    # 9. MMoE auxiliary losses (no-ops for STL/MTL)                      #
    # ------------------------------------------------------------------ #
    lb_weight = float(OmegaConf.select(cfg, "mmoe.load_balance_weight", default=0.0))
    gc_weight = float(
        OmegaConf.select(cfg, "mmoe.group_consistency_weight", default=0.0)
    )

    # Build trait-group index list for group-consistency loss
    trait_group_indices: list[list[int]] | None = None
    trait_groups_ids = OmegaConf.select(cfg, "mmoe.trait_groups", default=None)
    if trait_groups_ids is not None:
        trait_id_to_idx = {t.lstrip("X"): i for i, t in enumerate(TRAIT_NAMES)}
        groups_mapped: list[list[int]] = []
        for group in trait_groups_ids:
            idxs = [
                trait_id_to_idx[str(tid)]
                for tid in group
                if str(tid) in trait_id_to_idx
            ]
            if len(idxs) > 1:
                groups_mapped.append(idxs)
        trait_group_indices = groups_mapped if groups_mapped else None

    if lb_weight > 0:
        console.print(f"  Load-balance loss weight: [cyan]{lb_weight}[/cyan]")
    if gc_weight > 0 and trait_group_indices:
        console.print(
            f"  Group-consistency loss weight: [cyan]{gc_weight}[/cyan] ({len(trait_group_indices)} groups)"
        )

    # ------------------------------------------------------------------ #
    # 10. W&B                                                             #
    # ------------------------------------------------------------------ #
    requested_run_name = OmegaConf.select(cfg, "train.run_name")
    requested_run_group = OmegaConf.select(cfg, "train.group")
    if requested_run_name is not None:
        requested_run_name = str(requested_run_name)
    if requested_run_group is not None:
        requested_run_group = str(requested_run_group)

    wandb_module = None
    if cfg.wandb.enabled:
        try:
            import wandb as wandb_module
        except Exception as exc:
            raise RuntimeError(
                f"W&B enabled but import failed: {exc}. "
                "Set wandb.enabled=false or fix the environment."
            ) from exc
        run = wandb_module.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=requested_run_name,
            group=requested_run_group,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        run_name = run.name
        console.print(f"W&B run: [cyan]{run_name}[/cyan]")
    else:
        run_name = requested_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        console.print("[yellow]W&B logging disabled.[/yellow]")

    # ------------------------------------------------------------------ #
    # 11. Training loop                                                   #
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, checkpoint_dir / f"{run_name}.yaml")

    best_mean_r = float("-inf")
    best_val_loss = float("inf")
    best_r_epoch = -1
    best_loss_epoch = -1
    epochs_without_improvement = 0

    best_r_path = checkpoint_dir / f"{run_name}_best_r.pth"
    best_loss_path = checkpoint_dir / f"{run_name}.pth"

    for epoch in range(cfg.train.epochs):
        console.rule(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        # -------------------------------------------------------------- #
        # Training                                                        #
        # -------------------------------------------------------------- #
        model.train()
        train_splot_num = 0.0
        train_splot_den = 0.0
        train_gbif_num = 0.0
        train_gbif_den = 0.0

        for X, y_sv, y_ss, y_gv, y_gs in track(train_loader, description="Training"):
            X = X.to(device, dtype=torch.float32)
            y_sv = y_sv.to(device, dtype=torch.float32)  # (B, 37, 3, 3) SPLOT values
            y_ss = y_ss.to(device, dtype=torch.float32)  # (B, 37, 3, 3) SPLOT src
            y_gv = y_gv.to(
                device, dtype=torch.float32
            )  # (B, 37, 3, 3) GBIF calib values
            y_gs = y_gs.to(device, dtype=torch.float32)  # (B, 37, 3, 3) GBIF src

            # Sanitize predictors — clamp extreme values, zero-fill NaN
            pred_finite = torch.isfinite(X).all(dim=1, keepdim=True)  # (B, 1, H, W)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.clamp(X, -1e4, 1e4)

            # Where predictors are invalid, mask the targets too
            y_sv = torch.where(
                pred_finite.expand_as(y_sv), y_sv, torch.zeros_like(y_sv)
            )
            y_ss = torch.where(
                pred_finite.expand_as(y_ss), y_ss, torch.zeros_like(y_ss)
            )
            y_gv = torch.where(
                pred_finite.expand_as(y_gv), y_gv, torch.zeros_like(y_gv)
            )
            y_gs = torch.where(
                pred_finite.expand_as(y_gs), y_gs, torch.zeros_like(y_gs)
            )

            # Skip batch if no labels at all (very rare)
            if not ((y_ss > 0).any() or (y_gs > 0).any()):
                continue

            optimizer.zero_grad(set_to_none=True)

            pred = model(X)  # (B, 37, 3, 3)

            if not torch.isfinite(pred).all():
                continue

            # ---- SPLOT primary loss ----
            splot_num, splot_den = splot_loss_fn.loss_components(pred, y_sv, y_ss)
            loss = torch.zeros(1, device=device, dtype=torch.float32)

            if torch.isfinite(splot_num) and splot_den > 0:
                loss = loss + splot_num / splot_den
                train_splot_num += splot_num.detach().item()
                train_splot_den += splot_den.detach().item()

            # ---- GBIF auxiliary loss (calibrated, unweighted by quality) ----
            if aux_gbif_weight > 0:
                gbif_num, gbif_den = gbif_loss_fn.loss_components(pred, y_gv, y_gs)
                if torch.isfinite(gbif_num) and gbif_den > 0:
                    loss = loss + aux_gbif_weight * (gbif_num / gbif_den)
                    train_gbif_num += gbif_num.detach().item()
                    train_gbif_den += gbif_den.detach().item()

            # ---- MMoE auxiliary losses (no-ops for STL/MTL) ----
            if lb_weight > 0 and hasattr(model, "get_load_balancing_loss"):
                lb_loss = model.get_load_balancing_loss()
                if torch.isfinite(lb_loss):
                    loss = loss + lb_weight * lb_loss

            if (
                gc_weight > 0
                and trait_group_indices
                and hasattr(model, "get_group_consistency_loss")
            ):
                gc_loss = model.get_group_consistency_loss(trait_group_indices)
                if torch.isfinite(gc_loss):
                    loss = loss + gc_weight * gc_loss

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        train_splot_loss = (
            train_splot_num / train_splot_den if train_splot_den > 0 else float("nan")
        )
        train_gbif_loss = (
            train_gbif_num / train_gbif_den if train_gbif_den > 0 else float("nan")
        )
        console.print(
            f"  train_splot_loss={train_splot_loss:.5f}"
            + (
                f"  train_gbif_loss={train_gbif_loss:.5f}"
                if aux_gbif_weight > 0
                else ""
            )
            + f"  lr={current_lr:.2e}"
        )

        # -------------------------------------------------------------- #
        # Validation                                                      #
        # -------------------------------------------------------------- #
        model.eval()
        val_splot_num = 0.0
        val_splot_den = 0.0

        # Collect center-pixel predictions and labels for Pearson-r.
        # List-accumulation is numerically stable (avoids catastrophic cancellation
        # in the n*Σx² − (Σx)² running-sum formula when n is large).
        # Each val chip contributes exactly one center-pixel sample per trait.
        # Memory cost: ≈2 × 6956 × 37 × 4 bytes ≈ 2 MB — fine.
        val_preds_list: list[torch.Tensor] = []  # (B, 37) float32 on CPU
        val_labels_list: list[torch.Tensor] = []  # (B, 37) float32 on CPU
        val_masks_list: list[torch.Tensor] = []  # (B, 37) bool on CPU

        with torch.no_grad():
            for X, y_sv, y_ss, y_gv, y_gs in track(
                val_loader, description="Validation"
            ):
                X = X.to(device, dtype=torch.float32)
                y_sv = y_sv.to(device, dtype=torch.float32)
                y_ss = y_ss.to(device, dtype=torch.float32)

                pred_finite = torch.isfinite(X).all(dim=1, keepdim=True)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = torch.clamp(X, -1e4, 1e4)
                # Where predictors were invalid, zero out the supervision masks
                # (values are kept as-is; the mask below filters them anyway).
                y_ss = torch.where(
                    pred_finite.expand_as(y_ss), y_ss, torch.zeros_like(y_ss)
                )
                # y_sv left as-is — the Pearson mask handles invalid positions.

                pred = model(X)  # (B, 37, 3, 3)

                # Val loss: uses the full 3×3 SPLOT supervision.
                # WeightedMaskedDenseLoss handles non-finite predictions internally
                # via its own valid mask, so no need to skip the batch here.
                v_num, v_den = splot_loss_fn.loss_components(pred, y_sv, y_ss)
                if torch.isfinite(v_num) and v_den > 0:
                    val_splot_num += v_num.item()
                    val_splot_den += v_den.item()

                # Pearson-r collection: center pixel only (default) or all SPLOT pixels.
                # eval_all_splot=True gives more samples per batch but may double-count
                # SPLOT locations that appear in multiple chips as non-center pixels.
                if eval_all_splot:
                    _B, _T, _P, _ = pred.shape
                    pred_c = pred.permute(0, 2, 3, 1).reshape(
                        _B * _P * _P, _T
                    )  # (B*P*P, 37)
                    y_c = y_sv.permute(0, 2, 3, 1).reshape(_B * _P * _P, _T)
                    src_c = y_ss.permute(0, 2, 3, 1).reshape(_B * _P * _P, _T)
                else:
                    ci = center_idx
                    pred_c = pred[:, :, ci, ci]  # (B, 37)
                    y_c = y_sv[:, :, ci, ci]
                    src_c = y_ss[:, :, ci, ci]
                msk = (src_c == 2) & torch.isfinite(y_c) & torch.isfinite(pred_c)

                if msk.any():
                    val_preds_list.append(pred_c.cpu())
                    val_labels_list.append(y_c.cpu())
                    val_masks_list.append(msk.cpu())

        val_loss_avg = (
            val_splot_num / val_splot_den if val_splot_den > 0 else float("nan")
        )

        # Compute per-trait Pearson-r from collected tensors.
        # Mean-centering before dot-product is numerically stable.
        per_trait_r: list[float] = []
        if val_preds_list:
            P = torch.cat(val_preds_list, dim=0)  # (N_val, 37)
            G = torch.cat(val_labels_list, dim=0)  # (N_val, 37)
            M = torch.cat(val_masks_list, dim=0)  # (N_val, 37) bool

            for t in range(N_TRAITS):
                valid_t = M[:, t]
                n = valid_t.sum().item()
                if n > 1:
                    p_t = P[valid_t, t].double()  # float64 for precision
                    g_t = G[valid_t, t].double()
                    p_m = p_t - p_t.mean()
                    g_m = g_t - g_t.mean()
                    num_r = (p_m * g_m).sum()
                    den_r = torch.sqrt((p_m.pow(2).sum()) * (g_m.pow(2).sum()))
                    r = float(num_r / (den_r + 1e-12)) if den_r > 1e-12 else 0.0
                    per_trait_r.append(float(max(-1.0, min(1.0, r))))
                else:
                    per_trait_r.append(float("nan"))
        else:
            per_trait_r = [float("nan")] * N_TRAITS

        valid_r = [r for r in per_trait_r if math.isfinite(r)]
        val_macro_r = sum(valid_r) / len(valid_r) if valid_r else float("nan")

        console.print(
            f"  val_loss={val_loss_avg:.5f}  "
            f"val_mean_r={val_macro_r:.4f}  "
            f"(n_valid_traits={len(valid_r)}/{N_TRAITS})"
        )

        # Per-trait summary every 10 epochs
        if (epoch + 1) % 10 == 0:
            console.print("  Per-trait Pearson-r:")
            for t, (name, r) in enumerate(zip(TRAIT_NAMES, per_trait_r)):
                marker = " ←LOW" if math.isfinite(r) and r < 0.3 else ""
                console.print(f"    {name:8s}: {r:+.3f}{marker}")

        # -------------------------------------------------------------- #
        # Checkpointing & early stopping                                  #
        # -------------------------------------------------------------- #
        val_r_valid = math.isfinite(val_macro_r)
        val_loss_valid = math.isfinite(val_loss_avg)

        ckpt_payload = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "out_channels": N_TRAITS,
            "trait_names": TRAIT_NAMES,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        # Primary: best by val_mean_r (SPLOT center-pixel) — used for early stopping
        is_best_r = val_r_valid and val_macro_r > best_mean_r + es_min_delta
        if is_best_r:
            best_mean_r = val_macro_r
            best_r_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {**ckpt_payload, "val_loss": val_loss_avg, "val_mean_r": best_mean_r},
                best_r_path,
            )
            console.print(
                f"  [green]Best r checkpoint saved[/green] "
                f"(val_mean_r={best_mean_r:.4f}, epoch={best_r_epoch})"
            )
        elif val_r_valid:
            epochs_without_improvement += 1

        # Secondary: best by val_loss
        is_best_loss = val_loss_valid and val_loss_avg < best_val_loss - es_min_delta
        if is_best_loss:
            best_val_loss = val_loss_avg
            best_loss_epoch = epoch + 1
            torch.save({**ckpt_payload, "val_loss": best_val_loss}, best_loss_path)
            console.print(
                f"  [dim]Best loss checkpoint saved[/dim] (val_loss={best_val_loss:.5f})"
            )

        # W&B logging
        if cfg.wandb.enabled:
            log_dict: dict = {
                "epoch": epoch + 1,
                "train/splot_loss": train_splot_loss,
                "train/gbif_loss": train_gbif_loss,
                "val/loss": val_loss_avg,
                "val/mean_r": val_macro_r,
                "train/lr": current_lr,
            }
            if is_best_r:
                log_dict["val/mean_r_best"] = best_mean_r
            if is_best_loss:
                log_dict["val/loss_best"] = best_val_loss
            if val_r_valid:
                log_dict["val/per_trait_r"] = {
                    TRAIT_NAMES[t]: per_trait_r[t]
                    for t in range(N_TRAITS)
                    if math.isfinite(per_trait_r[t])
                }
            if wandb_module is not None:
                wandb_module.log(log_dict)

        # Early stopping
        if es_enabled and epochs_without_improvement >= es_patience:
            console.print(
                f"[yellow]Early stopping triggered[/yellow] "
                f"(patience={es_patience}, best at epoch {best_r_epoch})"
            )
            break

    # ------------------------------------------------------------------ #
    # Done                                                                #
    # ------------------------------------------------------------------ #
    console.rule("[bold cyan]DONE[/bold cyan]")
    if best_r_epoch > 0:
        console.print(f"Best val_mean_r={best_mean_r:.4f} at epoch {best_r_epoch}")
        console.print(f"  Primary checkpoint:   [cyan]{best_r_path}[/cyan]")
    if best_loss_epoch > 0:
        console.print(f"Best val_loss={best_val_loss:.5f} at epoch {best_loss_epoch}")
        console.print(f"  Secondary checkpoint: [cyan]{best_loss_path}[/cyan]")

    if cfg.wandb.enabled:
        run.summary.update(
            {
                "best_val_mean_r": float(best_mean_r)
                if math.isfinite(best_mean_r)
                else None,
                "best_val_loss": float(best_val_loss)
                if math.isfinite(best_val_loss)
                else None,
                "best_r_epoch": int(best_r_epoch),
                "best_loss_epoch": int(best_loss_epoch),
                "checkpoint_best_r_path": str(best_r_path),
                "checkpoint_best_loss_path": str(best_loss_path),
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
