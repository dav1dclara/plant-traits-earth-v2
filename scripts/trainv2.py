import itertools
import math
from datetime import datetime
from pathlib import Path

import hydra
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def _fit_pca(
    train_loader,
    in_channels: int,
    n_components: int,
    device: torch.device,
    max_pixels: int = 3_000_000,
):
    """Fit PCA on training predictor data via incremental covariance accumulation.

    Returns:
        components: (n_components, in_channels) top-K eigenvectors
        data_mean:  (in_channels,) channel mean over training pixels
        explained:  fraction of variance explained by top-K components
    """
    console.print(
        f"Fitting PCA: {in_channels} → {n_components} components (max {max_pixels:,} pixels)..."
    )
    cov_sum = torch.zeros(in_channels, in_channels, dtype=torch.float64)
    mean_sum = torch.zeros(in_channels, dtype=torch.float64)
    n_total = 0

    for X, _, _ in train_loader:
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
    # E[X^T X] - mean^T mean  (outer product of mean)
    cov = cov_sum / n_total - data_mean.unsqueeze(1) * data_mean.unsqueeze(0)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # ascending order
    components = eigenvectors[:, -n_components:].T.contiguous().float()  # (K, C)
    explained = (
        eigenvalues[-n_components:].sum() / eigenvalues.clamp(min=0).sum()
    ).item()
    console.print(
        f"  PCA done: top-{n_components} components explain {100 * explained:.1f}% variance "
        f"({n_total:,} pixels used)"
    )
    return components, data_mean.float(), explained


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


from ptev2.data.dataloader import get_dataloader
from ptev2.utils import seed_all

console = Console()


@hydra.main(config_path="../config", config_name="training/mtl_v2", version_base=None)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TRAINING[/bold cyan]")

    # Set random seed
    seed_all(cfg.train.seed)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    console.print(f"Device: [cyan]{device}[/cyan]")

    # Data configuration
    console.print("[bold]\nData[/bold]")
    root_dir = Path(cfg.data.root_dir)
    resolution_km = cfg.data.resolution_km
    patch_size = cfg.data.patch_size
    stride = cfg.data.stride
    zarr_dir = (
        root_dir / f"{resolution_km}km" / "chips" / f"patch{patch_size}_stride{stride}"
    )
    assert zarr_dir.exists(), f"Zarr directory does not exist: {zarr_dir}"

    console.print(f"Resolution (km): [cyan]{resolution_km}[/cyan]")
    console.print(f"Patch size: [cyan]{patch_size}[/cyan]")
    console.print(f"Stride: [cyan]{stride}[/cyan]")
    console.print(f"Zarr directory: [cyan]{zarr_dir}[/cyan]")

    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")

    # Predictors
    console.print("[bold]\nPredictors:[/bold]")
    predictors = [
        name
        for name, predictor_cfg in cfg.data.predictors.items()
        if bool(predictor_cfg.use)
    ]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    total_pred_bands = 0
    for predictor in predictors:
        n_bands = train_store[f"predictors/{predictor}"].shape[1]
        total_pred_bands += n_bands
        console.print(f"  - [cyan]{predictor}[/cyan] ({n_bands} bands)")
    console.print(f"  Total: [cyan]{total_pred_bands} bands[/cyan]")

    # Targets
    console.print("\n[bold]Targets:[/bold]")

    # What's available in the zarr store
    zarr_dataset_names = list(train_store["targets"].keys())
    zarr_band_names = train_store["targets"].attrs["band_names"]
    band_to_idx = {name: idx for idx, name in enumerate(zarr_band_names)}

    # What's requested in the config
    target_cfg = cfg.data.targets
    target_dataset = str(target_cfg.dataset)
    cfg_bands = [str(v) for v in target_cfg.bands]

    # Validate config against zarr store
    if target_dataset not in zarr_dataset_names:
        raise ValueError(
            f"Dataset '{target_dataset}' not found in zarr. Available: {', '.join(zarr_dataset_names)}"
        )
    zarr_all_traits = [
        f.replace("X", "").replace(".tif", "")
        for f in train_store[f"targets/{target_dataset}"].attrs["files"]
    ]
    traits = (
        [str(v) for v in target_cfg.traits] if target_cfg.traits else zarr_all_traits
    )
    for trait in traits:
        if trait not in zarr_all_traits:
            raise ValueError(
                f"Trait '{trait}' not found in dataset '{target_dataset}' in zarr."
            )
    for band in cfg_bands:
        if band not in band_to_idx:
            raise ValueError(
                f"Band '{band}' not found in zarr. Available: {', '.join(zarr_band_names)}"
            )
    bands = cfg_bands

    # Print what we're using
    console.print(f"Dataset: [cyan]{target_dataset}[/cyan]")
    console.print(f"Traits ([cyan]{len(traits)}[/cyan]):")
    for trait in traits:
        console.print(f"  - {trait}")
    console.print(f"Bands ([cyan]{len(bands)}[/cyan]):")
    for band in bands:
        console.print(f"  - {band}")

    n_bands = len(zarr_band_names)
    target_indices = [
        trait_pos * n_bands + band_to_idx[band]
        for trait_pos in range(len(traits))
        for band in bands
    ]
    source_indices = [
        trait_pos * n_bands + band_to_idx["source"] for trait_pos in range(len(traits))
    ]

    # Primary output size (mean band only — matches test.py evaluation)
    primary_out_channels = len(target_indices)  # = len(traits) * len(bands)

    # Auxiliary supervision bands (q05 / q95) — loaded alongside primary but
    # only used during training. NOT included in checkpoint out_channels.
    aux_bands_cfg = [
        str(b) for b in OmegaConf.select(cfg, "train.aux_bands", default=[])
    ]
    aux_target_indices: list[int] = []
    aux_bands_available: list[str] = []
    for ab in aux_bands_cfg:
        if ab in band_to_idx:
            aux_target_indices.extend(
                [t * n_bands + band_to_idx[ab] for t in range(len(traits))]
            )
            aux_bands_available.append(ab)
        else:
            console.print(f"[yellow]Aux band '{ab}' not in zarr — skipped.[/yellow]")

    # Combined indices passed to the dataloader (primary first, aux appended)
    all_target_indices = target_indices + aux_target_indices

    # --- Trait group mapping for MMoE group-consistency loss ---
    # Maps trait IDs in cfg.mmoe.trait_groups → channel indices in model output.
    # If the config doesn't have mmoe.trait_groups, this stays None.
    trait_group_indices: list[list[int]] | None = None
    trait_groups_ids = OmegaConf.select(cfg, "mmoe.trait_groups", default=None)
    if trait_groups_ids is not None:
        trait_id_to_idx = {t: i for i, t in enumerate(traits)}
        groups_mapped = []
        for group in trait_groups_ids:
            idxs = [
                trait_id_to_idx[str(tid)]
                for tid in group
                if str(tid) in trait_id_to_idx
            ]
            if len(idxs) > 1:
                groups_mapped.append(idxs)
        trait_group_indices = groups_mapped if groups_mapped else None
        if trait_group_indices:
            console.print(
                f"Trait group indices loaded: [cyan]{len(trait_group_indices)} groups[/cyan]"
            )

    # Trait masking: exclude certain traits from the loss (model still predicts them).
    # Useful for ablation — remove unlearnable traits (223, 95, 138, 289, 224).
    mask_trait_ids = [
        str(t) for t in OmegaConf.select(cfg, "train.mask_traits", default=[])
    ]
    excluded_channels: list[int] = []
    if mask_trait_ids:
        trait_id_to_ch = {t: i for i, t in enumerate(traits)}
        excluded_channels = [
            trait_id_to_ch[tid] for tid in mask_trait_ids if tid in trait_id_to_ch
        ]
        missing = [tid for tid in mask_trait_ids if tid not in trait_id_to_ch]
        console.print(
            f"Masking [cyan]{len(excluded_channels)}[/cyan] traits from loss: "
            f"{[traits[c] for c in excluded_channels]}"
        )
        if missing:
            console.print(
                f"[yellow]mask_traits not found in dataset (skipped): {missing}[/yellow]"
            )

    # MMoE auxiliary loss weights (0 = disabled; safe for STL/MTL which lack these methods)
    lb_weight = float(OmegaConf.select(cfg, "mmoe.load_balance_weight", default=0.0))
    gc_weight = float(
        OmegaConf.select(cfg, "mmoe.group_consistency_weight", default=0.0)
    )
    if lb_weight > 0:
        console.print(f"Load-balance loss weight: [cyan]{lb_weight}[/cyan]")
    if gc_weight > 0:
        console.print(f"Group-consistency loss weight: [cyan]{gc_weight}[/cyan]")

    # Dataloader configuration
    console.print("\n[bold]Data loaders:[/bold]")
    batch_size = cfg.data_loaders.batch_size
    num_workers = cfg.data_loaders.num_workers

    console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
    console.print(f"Num workers: [cyan]{num_workers}[/cyan]")

    dataloader_cfg = {
        "zarr_dir": zarr_dir,
        "predictors": predictors,
        "target": target_dataset,
        "target_indices": all_target_indices,
        "source_indices": source_indices,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    train_loader = get_dataloader(split="train", **dataloader_cfg)
    val_loader = get_dataloader(split="val", **dataloader_cfg)

    X_t, y_t, src_t = next(iter(train_loader))
    X_v, y_v, src_v = next(iter(val_loader))

    assert X_t.shape[1:] == X_v.shape[1:], (
        f"X channel/spatial mismatch: train={tuple(X_t.shape[1:])} vs val={tuple(X_v.shape[1:])}"
    )
    assert y_t.shape[1:] == y_v.shape[1:], (
        f"y channel/spatial mismatch: train={tuple(y_t.shape[1:])} vs val={tuple(y_v.shape[1:])}"
    )
    assert src_t.shape[1:] == src_v.shape[1:], (
        f"source_mask channel/spatial mismatch: train={tuple(src_t.shape[1:])} vs val={tuple(src_v.shape[1:])}"
    )

    console.print(f"Predictor shape (C,H,W): [cyan]{tuple(X_t.shape[1:])}[/cyan]")
    console.print(f"Target shape (C,H,W): [cyan]{tuple(y_t.shape[1:])}[/cyan]")
    console.print(f"Source mask shape (C,H,W): [cyan]{tuple(src_t.shape[1:])}[/cyan]")

    # Model
    console.print("\n[bold]Model and training configuration[/bold]")
    model = instantiate(
        cfg.models, in_channels=total_pred_bands, out_channels=primary_out_channels
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model:         [cyan]{cfg.models._target_}[/cyan]")
    console.print(f"  In channels:  [cyan]{total_pred_bands}[/cyan]")
    console.print(f"  Out channels: [cyan]{primary_out_channels}[/cyan]")
    console.print(f"  Parameters:   [cyan]{n_params:,}[/cyan]")

    # PCA input projection: fit on training data and initialise encoder.input_pca weights.
    # Triggered when models.pca_n_components is set in the model config (non-null).
    # The PCA layer is baked into the model state_dict, so test.py loads it transparently.
    pca_n_components = OmegaConf.select(cfg, "models.pca_n_components", default=None)
    if (
        pca_n_components is not None
        and hasattr(model, "encoder")
        and getattr(model.encoder, "input_pca", None) is not None
    ):
        components, data_mean, explained = _fit_pca(
            train_loader, total_pred_bands, int(pca_n_components), device
        )
        model.encoder.input_pca.init_from_pca(
            components.to(device), data_mean.to(device)
        )
        console.print(
            f"  PCA projection: [cyan]{total_pred_bands} → {pca_n_components}[/cyan] "
            f"([cyan]{100 * explained:.1f}%[/cyan] variance)"
        )

    # Aux quantile head (q05/q95 supervision, separate from main model checkpoint)
    aux_weight = float(OmegaConf.select(cfg, "train.aux_weight", default=0.0))
    aux_head = None
    if aux_weight > 0 and aux_target_indices:
        from ptev2.models.multitask_v2 import AuxQuantileHead

        aux_head = AuxQuantileHead(n_traits=len(traits)).to(device)
        aux_n_params = sum(p.numel() for p in aux_head.parameters())
        console.print(
            f"Aux quantile head: weight={aux_weight}, bands={aux_bands_available}, "
            f"params={aux_n_params:,}"
        )
    elif aux_bands_available:
        console.print(
            f"[yellow]Aux bands configured ({aux_bands_available}) but aux_weight=0 — skipped.[/yellow]"
        )

    # Optimizer, loss, scheduler
    loss_fn = instantiate(cfg.train.loss).to(device)
    all_opt_params = list(model.parameters()) + list(loss_fn.parameters())
    if aux_head is not None:
        all_opt_params += list(aux_head.parameters())
    optimizer = instantiate(cfg.train.optimizer, params=all_opt_params)
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)
    # scheduler_metric_name = str(cfg.train.scheduler_step.metric)  # TODO: what does this do?
    grad_clip_norm = float(cfg.train.gradient_clip_norm)
    console.print(f"Optimizer:             [cyan]{cfg.train.optimizer._target_}[/cyan]")
    console.print(f"Loss:                  [cyan]{cfg.train.loss._target_}[/cyan]")
    console.print(f"Scheduler:             [cyan]{cfg.train.scheduler._target_}[/cyan]")
    # console.print(f"Scheduler step metric: [cyan]{scheduler_metric_name}[/cyan]")
    console.print(f"Gradient clip norm:    [cyan]{grad_clip_norm}[/cyan]")

    # Early stopping configuration
    early_stopping_enabled = bool(cfg.train.early_stopping.enabled)
    early_stopping_patience = int(cfg.train.early_stopping.patience)
    early_stopping_min_delta = float(cfg.train.early_stopping.min_delta)
    console.print(f"Early stopping enabled: [cyan]{early_stopping_enabled}[/cyan]")
    if early_stopping_enabled:
        console.print(f"  Patience (epochs):   [cyan]{early_stopping_patience}[/cyan]")
        console.print(f"  Min delta:           [cyan]{early_stopping_min_delta}[/cyan]")

    requested_run_name = OmegaConf.select(cfg, "train.run_name")
    if requested_run_name is not None:
        requested_run_name = str(requested_run_name)
    requested_run_group = OmegaConf.select(cfg, "train.group")
    if requested_run_group is not None:
        requested_run_group = str(requested_run_group)

    wandb_module = None

    # W&B
    if cfg.wandb.enabled:
        try:
            import wandb as wandb_module  # Lazy import for debugger stability.
        except Exception as exc:
            raise RuntimeError(
                "W&B is enabled but import failed. "
                "Set wandb.enabled=false for debugging or fix the environment. "
                f"Original error: {exc}"
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
        console.print(f"W&B logging enabled. Run: [cyan]{run_name}[/cyan]")
    else:
        run_name = requested_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        console.print("[yellow]W&B logging disabled.[/yellow]")

    # Checkpoint and early stopping state
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_mean_r = float("-inf")
    best_loss_epoch = -1
    best_r_epoch = -1
    epochs_without_improvement = 0
    best_loss_path = checkpoint_dir / f"{run_name}.pth"
    best_r_path = checkpoint_dir / f"{run_name}_best_r.pth"
    config_path = checkpoint_dir / f"{run_name}.yaml"
    OmegaConf.save(cfg, config_path)

    for epoch in range(cfg.train.epochs):
        console.rule(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        # Training loop
        model.train()
        train_num_total = 0.0
        train_den_total = 0.0
        # train_total_pixels = 0
        # train_valid_pixels = 0

        for X, y, src in track(train_loader, description="Training"):
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            src = src.to(device, dtype=torch.float32)

            # Split primary (mean) vs aux (q05/q95) targets
            y_mean = y[:, :primary_out_channels]
            y_aux = y[:, primary_out_channels:]  # empty tensor if no aux bands

            # Mask primary targets where predictors invalid or source unlabeled
            valid = torch.isfinite(X).all(dim=1, keepdim=True)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.clamp(X, min=-1e4, max=1e4)
            y_mean = torch.where(valid.expand_as(y_mean), y_mean, torch.nan)
            y_mean = torch.where(src > 0, y_mean, torch.nan)

            if not torch.isfinite(y_mean).any():
                continue

            # Zero out excluded traits so they don't contribute to loss
            if excluded_channels:
                y_mean[:, excluded_channels] = float("nan")

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(X)
            if not torch.isfinite(y_pred).all():
                continue
            batch_num, batch_den = loss_fn.loss_components(y_pred, y_mean, src)
            if (
                not torch.isfinite(batch_num)
                or not torch.isfinite(batch_den)
                or batch_den <= 0.0
            ):
                continue
            loss = batch_num / batch_den

            # Auxiliary quantile supervision (q05 / q95)
            if aux_head is not None and y_aux.shape[1] > 0:
                aux_pred = aux_head(y_pred)  # (B, 2*T, H, W)
                n_t = len(traits)
                n_aux_bands = y_aux.shape[1] // n_t  # number of aux bands (e.g. 2)
                for b_i in range(n_aux_bands):
                    y_aux_b = y_aux[:, b_i * n_t : (b_i + 1) * n_t]
                    aux_pred_b = aux_pred[:, b_i * n_t : (b_i + 1) * n_t]
                    aux_mask = (
                        (src > 0) & torch.isfinite(y_aux_b) & torch.isfinite(aux_pred_b)
                    )
                    if aux_mask.any():
                        aux_loss_b = torch.nn.functional.mse_loss(
                            aux_pred_b[aux_mask], y_aux_b[aux_mask]
                        )
                        if torch.isfinite(aux_loss_b):
                            loss = loss + aux_weight * aux_loss_b

            # MMoE auxiliary losses (no-ops for STL/MTL which lack these methods)
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

            loss.backward()
            all_clip_params = list(model.parameters()) + list(loss_fn.parameters())
            if aux_head is not None:
                all_clip_params += list(aux_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_clip_params, max_norm=grad_clip_norm)
            optimizer.step()

            train_num_total += batch_num.detach().item()
            train_den_total += batch_den.detach().item()

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss_avg = (
            train_num_total / train_den_total if train_den_total > 0.0 else float("nan")
        )
        # train_valid_pct = 100.0 * train_valid_pixels / train_total_pixels if train_total_pixels > 0 else float("nan")
        console.print(f"  train_loss={train_loss_avg:.6f}")

        # Validation loop
        model.eval()
        val_num_total = 0.0
        val_den_total = 0.0

        # Running stats for per-trait Pearson-r on SPLOT-only pixels (src == 2)
        # Using SPLOT-only matches the test evaluation metric and makes best_r checkpoint meaningful.
        n_traits_out = len(traits)
        r_sum_p = torch.zeros(n_traits_out)  # sum of predictions
        r_sum_g = torch.zeros(n_traits_out)  # sum of targets
        r_sum_pg = torch.zeros(n_traits_out)  # sum of pred*target
        r_sum_p2 = torch.zeros(n_traits_out)  # sum of pred^2
        r_sum_g2 = torch.zeros(n_traits_out)  # sum of target^2
        r_cnt = torch.zeros(n_traits_out)  # count of valid labeled pixels

        with torch.no_grad():
            for X, y, src in track(val_loader, description="Validation"):
                X = X.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                src = src.to(device, dtype=torch.float32)

                # Use only primary (mean) targets for validation
                y_mean = y[:, :primary_out_channels]

                valid = torch.isfinite(X).all(dim=1, keepdim=True)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                X = torch.clamp(X, min=-1e4, max=1e4)
                y_mean = torch.where(valid.expand_as(y_mean), y_mean, torch.nan)
                y_mean = torch.where(src > 0, y_mean, torch.nan)

                if not torch.isfinite(y_mean).any():
                    continue

                y_pred = model(X)
                if not torch.isfinite(y_pred).all():
                    continue
                batch_num, batch_den = loss_fn.loss_components(y_pred, y_mean, src)
                if (
                    not torch.isfinite(batch_num)
                    or not torch.isfinite(batch_den)
                    or batch_den <= 0.0
                ):
                    continue
                val_num_total += batch_num.item()
                val_den_total += batch_den.item()

                # Accumulate running stats for Pearson-r (SPLOT-only pixels, src == 2)
                # val_mask: (B, T, H, W) — True where src == 2 and both y/pred are finite
                val_r_mask = (
                    (src == 2) & torch.isfinite(y_mean) & torch.isfinite(y_pred)
                )
                # Flatten to (T, B*H*W)
                T = n_traits_out
                vs = val_r_mask.permute(1, 0, 2, 3).reshape(T, -1)  # bool
                p_flat = y_pred.permute(1, 0, 2, 3).reshape(T, -1)
                g_flat = y_mean.permute(1, 0, 2, 3).reshape(T, -1)
                p_safe = torch.where(vs, p_flat, torch.zeros_like(p_flat))
                g_safe = torch.where(vs, g_flat, torch.zeros_like(g_flat))
                vs_f = vs.float()
                r_sum_p += p_safe.sum(dim=1).cpu()
                r_sum_g += g_safe.sum(dim=1).cpu()
                r_sum_pg += (p_safe * g_safe).sum(dim=1).cpu()
                r_sum_p2 += p_safe.pow(2).sum(dim=1).cpu()
                r_sum_g2 += g_safe.pow(2).sum(dim=1).cpu()
                r_cnt += vs_f.sum(dim=1).cpu()

        val_loss_avg = (
            val_num_total / val_den_total if val_den_total > 0.0 else float("nan")
        )

        # Compute per-trait Pearson-r from running stats
        per_trait_r: list[float] = []
        for t in range(n_traits_out):
            n = r_cnt[t].item()
            if n > 1:
                sp = r_sum_p[t].item()
                sg = r_sum_g[t].item()
                spg = r_sum_pg[t].item()
                sp2 = r_sum_p2[t].item()
                sg2 = r_sum_g2[t].item()
                num_r = n * spg - sp * sg
                den_r_sq = (n * sp2 - sp * sp) * (n * sg2 - sg * sg)
                den_r = math.sqrt(max(den_r_sq, 0.0))
                r = (num_r / (den_r + 1e-8)) if den_r > 1e-8 else 0.0
                per_trait_r.append(max(-1.0, min(1.0, r)))
            else:
                per_trait_r.append(float("nan"))
        valid_r = [r for r in per_trait_r if math.isfinite(r)]
        val_macro_r = sum(valid_r) / len(valid_r) if valid_r else float("nan")

        console.print(f"  val_loss={val_loss_avg:.6f}  val_mean_r={val_macro_r:.4f}")

        # Checkpoint: save best-by-loss and best-by-mean_r separately
        val_loss_valid = math.isfinite(val_loss_avg)
        val_r_valid = math.isfinite(val_macro_r)

        if not val_loss_valid:
            console.print(
                "[yellow]val_loss is NaN — skipping checkpoint and early-stopping update.[/yellow]"
            )

        ckpt_payload = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "in_channels": total_pred_bands,
            "out_channels": primary_out_channels,  # primary (mean) channels only
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        # Primary checkpoint: best by val_mean_r (SPLOT-only) — used for early stopping
        # This is the metric that matches test evaluation, so we optimize for it directly.
        is_best_r = val_r_valid and val_macro_r > best_mean_r + early_stopping_min_delta
        if is_best_r:
            best_mean_r = val_macro_r
            best_r_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {**ckpt_payload, "val_loss": val_loss_avg, "val_mean_r": best_mean_r},
                best_r_path,
            )
            console.print(
                f"[green]Best checkpoint saved[/green] (val_mean_r={best_mean_r:.4f})"
            )
        elif val_r_valid:
            epochs_without_improvement += 1

        # Secondary checkpoint: best by val_loss (no early stopping on this)
        is_best_loss = (
            val_loss_valid and val_loss_avg < best_val_loss - early_stopping_min_delta
        )
        if is_best_loss:
            best_val_loss = val_loss_avg
            best_loss_epoch = epoch + 1
            torch.save({**ckpt_payload, "val_loss": best_val_loss}, best_loss_path)
            console.print(
                f"[dim]Best-loss checkpoint saved[/dim] (val_loss={best_val_loss:.6f})"
            )

        # W&B logging
        if cfg.wandb.enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss_avg,
                "val/loss": val_loss_avg,
                "val/mean_r": val_macro_r,
                "train/lr": current_lr,
            }
            if is_best_loss:
                log_dict["val/loss_best"] = best_val_loss
            if is_best_r:
                log_dict["val/mean_r_best"] = best_mean_r
            # Per-trait Pearson-r (only log when we have valid values to avoid noise)
            if val_r_valid:
                log_dict["val/per_trait_r"] = {
                    traits[t]: per_trait_r[t]
                    for t in range(n_traits_out)
                    if math.isfinite(per_trait_r[t])
                }
            if wandb_module is not None:
                wandb_module.log(log_dict)

        # Early stopping (by val_mean_r — primary metric)
        if (
            early_stopping_enabled
            and epochs_without_improvement >= early_stopping_patience
        ):
            console.print(
                f"[yellow]Early stopping triggered[/yellow] (patience={early_stopping_patience}, best_r_epoch={best_r_epoch})"
            )
            break

    console.rule("[bold cyan]DONE[/bold cyan]")
    if best_r_epoch > 0:
        console.print(f"Best val_mean_r={best_mean_r:.4f} at epoch {best_r_epoch}")
        console.print(f"  Primary checkpoint:     [cyan]{best_r_path}[/cyan]")
    if best_loss_epoch > 0:
        console.print(f"Best val_loss={best_val_loss:.6f} at epoch {best_loss_epoch}")
        console.print(f"  Secondary checkpoint:   [cyan]{best_loss_path}[/cyan]")

    if cfg.wandb.enabled:
        final_status = (
            "ok" if math.isfinite(best_val_loss) and best_loss_epoch > 0 else "failed"
        )
        run.summary["best_val_loss"] = (
            float(best_val_loss) if math.isfinite(best_val_loss) else None
        )
        run.summary["best_val_mean_r"] = (
            float(best_mean_r) if math.isfinite(best_mean_r) else None
        )
        run.summary["best_loss_epoch"] = int(best_loss_epoch)
        run.summary["best_r_epoch"] = int(best_r_epoch)
        run.summary["checkpoint_path"] = str(best_loss_path)
        run.summary["checkpoint_best_r_path"] = str(best_r_path)
        run.summary["status"] = final_status

    # Close wandb run
    if cfg.wandb.enabled:
        run.finish()


if __name__ == "__main__":
    main()
