import math
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import track

from ptev2.data.dataloader import get_dataloader
from ptev2.metrics.evaluation import summarize_single_trait_metrics
from ptev2.utils import (
    build_target_layout,
    move_bundle_to_device,
    predict_batch,
    resolve_device,
    resolve_model_cfg,
    resolve_train_zarr_dir,
    seed_all,
)

console = Console()


def _resolve_selection_value(
    metric_name: str,
    val_splot_loss: float,
    val_summary: dict[str, Any] | None,
) -> float:
    if metric_name == "val_splot_loss":
        return float(val_splot_loss)
    if val_summary is None:
        return float("nan")
    lookup = {
        "val_splot_macro_pearson": float(val_summary["macro_pearson_r"]),
        "val_splot_macro_rmse": float(val_summary["macro_rmse"]),
        "val_splot_rmse": float(val_summary["rmse"]),
    }
    if metric_name not in lookup:
        raise ValueError(
            "Unsupported train.selection.metric='{}'".format(metric_name)
            + ". Use val_splot_macro_pearson|val_splot_macro_rmse|val_splot_rmse|val_splot_loss."
        )
    return lookup[metric_name]


def _is_better(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not math.isfinite(current):
        return False
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError("train.selection.mode must be 'max' or 'min'.")


def _center_pixel(t: torch.Tensor) -> torch.Tensor:
    h = t.shape[-2]
    w = t.shape[-1]
    cy = h // 2
    cx = w // 2
    return t[..., cy : cy + 1, cx : cx + 1]


@hydra.main(
    config_path="../../config/22km", config_name="training/default", version_base=None
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]TRAINING 22KM[/bold cyan]")

    seed_all(int(cfg.train.seed))

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    console.print(f"Device: [cyan]{device}[/cyan]")

    zarr_dir = resolve_train_zarr_dir(cfg)
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Zarr directory does not exist: {zarr_dir}")
    console.print(f"Zarr directory: [cyan]{zarr_dir}[/cyan]")

    train_store = zarr.open_group(str(zarr_dir / "train.zarr"), mode="r")

    predictors = [
        name
        for name, predictor_cfg in cfg.data.predictors.items()
        if bool(predictor_cfg.use)
    ]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.data.predictors.")

    total_pred_bands = 0
    for predictor in predictors:
        n_bands = train_store[f"predictors/{predictor}"].shape[1]
        total_pred_bands += n_bands
        console.print(f"  - {predictor}: {n_bands} bands")

    target_layout = build_target_layout(cfg, train_store)
    traits = list(target_layout["traits"])
    bands = list(target_layout["bands"])
    eval_dataset = str(target_layout["eval_dataset"])

    console.print(f"Targets mode: [cyan]{target_layout['mode']}[/cyan]")
    console.print(f"Train datasets: [cyan]{target_layout['active_datasets']}[/cyan]")
    console.print(f"Eval dataset: [cyan]{eval_dataset}[/cyan]")
    console.print(f"Traits ({len(traits)}):")
    for trait in traits:
        console.print(f"  - {trait}")
    console.print(f"Bands: [cyan]{bands}[/cyan]")

    batch_size = int(cfg.data_loaders.batch_size)
    num_workers = int(cfg.data_loaders.num_workers)
    train_fraction = float(OmegaConf.select(cfg, "data_loaders.train_fraction") or 1.0)
    val_fraction = float(OmegaConf.select(cfg, "data_loaders.val_fraction") or 1.0)
    subset_seed = int(
        OmegaConf.select(cfg, "data_loaders.subset_seed") or int(cfg.train.seed)
    )

    dataloader_cfg = {
        "zarr_dir": zarr_dir,
        "predictors": predictors,
        "target_layouts": target_layout["layouts"],
        "return_target_bundle": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "subset_seed": subset_seed,
    }

    train_loader = get_dataloader(
        split="train", split_fraction=train_fraction, **dataloader_cfg
    )
    val_loader = get_dataloader(
        split="val", split_fraction=val_fraction, **dataloader_cfg
    )

    x0, _ = next(iter(train_loader))
    console.print(f"Predictor shape (C,H,W): [cyan]{tuple(x0.shape[1:])}[/cyan]")
    console.print(f"Train samples: [cyan]{len(train_loader.dataset):,}[/cyan]")
    console.print(f"Val samples: [cyan]{len(val_loader.dataset):,}[/cyan]")

    model_cfg = resolve_model_cfg(cfg)
    out_channels = len(traits) * len(bands)
    model = instantiate(
        model_cfg, in_channels=total_pred_bands, out_channels=out_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model: [cyan]{model_cfg._target_}[/cyan]")
    console.print(f"Params: [cyan]{n_params:,}[/cyan]")

    loss_fn = instantiate(cfg.train.loss).to(device)
    optimizer = instantiate(cfg.train.optimizer, params=list(model.parameters()))
    scheduler = instantiate(cfg.train.scheduler, optimizer=optimizer)

    grad_clip_norm = float(cfg.train.gradient_clip_norm)
    selection_metric_name = str(
        OmegaConf.select(cfg, "train.selection.metric") or "val_splot_macro_pearson"
    )
    selection_mode = str(OmegaConf.select(cfg, "train.selection.mode") or "max")

    primary_ds = str(OmegaConf.select(cfg, "data.targets.primary_dataset") or "splot")
    aux_ds = str(OmegaConf.select(cfg, "data.targets.auxiliary_dataset") or "gbif")
    w_splot = float(OmegaConf.select(cfg, "train.loss.w_splot") or 1.0)
    w_gbif = float(OmegaConf.select(cfg, "train.loss.w_gbif") or 0.1)

    lb_weight = float(OmegaConf.select(cfg, "mmoe.load_balance_weight") or 0.0)
    gc_weight = float(OmegaConf.select(cfg, "mmoe.group_consistency_weight") or 0.0)

    trait_group_indices: list[list[int]] | None = None
    trait_groups_ids = OmegaConf.select(cfg, "mmoe.trait_groups", default=None)
    if trait_groups_ids is not None:
        trait_id_to_idx = {str(t): i for i, t in enumerate(traits)}
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

    requested_run_name = OmegaConf.select(cfg, "train.run_name")
    requested_run_group = OmegaConf.select(cfg, "train.group")
    requested_run_name = None if requested_run_name is None else str(requested_run_name)
    requested_run_group = (
        None if requested_run_group is None else str(requested_run_group)
    )

    wandb_module = None
    if cfg.wandb.enabled:
        import wandb as wandb_module

        run = wandb_module.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=requested_run_name,
            group=requested_run_group,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        run_name = requested_run_name or run.name
    else:
        run_name = requested_run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{run_name}.pth"
    OmegaConf.save(cfg, checkpoint_dir / f"{run_name}.yaml")

    best_selection = float("inf") if selection_mode == "min" else float("-inf")
    best_epoch = -1
    epochs_wo_improve = 0
    best_val_mean_r = float("-inf")
    best_val_loss = float("inf")

    es_enabled = bool(cfg.train.early_stopping.enabled)
    es_patience = int(cfg.train.early_stopping.patience)
    es_min_delta = float(cfg.train.early_stopping.min_delta)

    for epoch in range(int(cfg.train.epochs)):
        console.rule(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        model.train()
        train_splot_num = train_splot_den = 0.0
        train_gbif_num = train_gbif_den = 0.0

        for X, bundle in track(train_loader, description="Training"):
            X = X.to(device, dtype=torch.float32)
            bundle = move_bundle_to_device(bundle, device)
            y_pred, valid_x = predict_batch(
                model, X, validity_mode="min_fraction", min_finite_ratio=0.2
            )

            p_payload = bundle.get(primary_ds)
            a_payload = bundle.get(aux_ds)
            if p_payload is None or a_payload is None:
                raise ValueError("Missing primary/aux dataset payload.")

            # mask targets where predictors are invalid
            p_y = torch.where(
                valid_x.expand_as(p_payload["y"]), p_payload["y"], torch.nan
            )
            a_y = torch.where(
                valid_x.expand_as(a_payload["y"]), a_payload["y"], torch.nan
            )

            p_num, p_den = loss_fn.loss_components(
                y_pred, p_y, p_payload["source_mask"]
            )
            a_num, a_den = loss_fn.loss_components(
                y_pred, a_y, a_payload["source_mask"]
            )

            zero = y_pred.sum() * 0.0
            p_loss = p_num / p_den if bool((p_den > 0).item()) else zero
            a_loss = a_num / a_den if bool((a_den > 0).item()) else zero
            loss = w_splot * p_loss + w_gbif * a_loss

            if lb_weight > 0 and hasattr(model, "get_load_balancing_loss"):
                lb = model.get_load_balancing_loss()
                if torch.isfinite(lb):
                    loss = loss + lb_weight * lb

            if (
                gc_weight > 0
                and trait_group_indices
                and hasattr(model, "get_group_consistency_loss")
            ):
                gc = model.get_group_consistency_loss(trait_group_indices)
                if torch.isfinite(gc):
                    loss = loss + gc_weight * gc

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            if bool((p_den > 0).item()):
                train_splot_num += p_num.detach().item()
                train_splot_den += p_den.detach().item()
            if bool((a_den > 0).item()):
                train_gbif_num += a_num.detach().item()
                train_gbif_den += a_den.detach().item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass
        else:
            scheduler.step()

        train_splot_loss = (
            train_splot_num / train_splot_den if train_splot_den > 0 else float("nan")
        )
        train_gbif_loss = (
            train_gbif_num / train_gbif_den if train_gbif_den > 0 else float("nan")
        )

        model.eval()
        val_splot_num = val_splot_den = 0.0
        val_gbif_num = val_gbif_den = 0.0
        val_y_true_parts = []
        val_y_pred_parts = []
        val_valid_parts = []

        with torch.no_grad():
            for X, bundle in track(val_loader, description="Validation"):
                X = X.to(device, dtype=torch.float32)
                bundle = move_bundle_to_device(bundle, device)
                y_pred, valid_x = predict_batch(
                    model, X, validity_mode="min_fraction", min_finite_ratio=0.2
                )

                p_payload = bundle.get(primary_ds)
                a_payload = bundle.get(aux_ds)
                if p_payload is None or a_payload is None:
                    raise ValueError("Missing primary/aux dataset payload.")

                p_y = torch.where(
                    valid_x.expand_as(p_payload["y"]), p_payload["y"], torch.nan
                )
                a_y = torch.where(
                    valid_x.expand_as(a_payload["y"]), a_payload["y"], torch.nan
                )

                p_num, p_den = loss_fn.loss_components(
                    y_pred, p_y, p_payload["source_mask"]
                )
                a_num, a_den = loss_fn.loss_components(
                    y_pred, a_y, a_payload["source_mask"]
                )

                if bool((p_den > 0).item()):
                    val_splot_num += p_num.item()
                    val_splot_den += p_den.item()
                if bool((a_den > 0).item()):
                    val_gbif_num += a_num.item()
                    val_gbif_den += a_den.item()

                eval_payload = bundle.get(eval_dataset)
                if eval_payload is None:
                    continue

                # train_chips-like validation logic: center pixel Pearson-R on SPLOT
                y_eval_c = _center_pixel(eval_payload["y"])
                src_eval_c = _center_pixel(eval_payload["source_mask"])
                y_pred_c = _center_pixel(y_pred)
                valid_eval = (
                    torch.isfinite(y_eval_c)
                    & torch.isfinite(y_pred_c)
                    & (src_eval_c > 0)
                    & _center_pixel(valid_x).expand_as(y_eval_c)
                )
                if bool(valid_eval.any()):
                    val_y_true_parts.append(y_eval_c.detach().cpu())
                    val_y_pred_parts.append(y_pred_c.detach().cpu())
                    val_valid_parts.append(valid_eval.detach().cpu())

        val_splot_loss = (
            val_splot_num / val_splot_den if val_splot_den > 0 else float("nan")
        )
        val_gbif_loss = (
            val_gbif_num / val_gbif_den if val_gbif_den > 0 else float("nan")
        )

        val_summary = None
        if val_y_true_parts:
            val_summary = summarize_single_trait_metrics(
                y_true=torch.cat(val_y_true_parts, dim=0),
                y_pred=torch.cat(val_y_pred_parts, dim=0),
                source_mask=None,
                valid_mask=torch.cat(val_valid_parts, dim=0),
                trait_names=traits,
                n_bands=len(bands),
            )

        selection_value = _resolve_selection_value(
            selection_metric_name, val_splot_loss, val_summary
        )

        improved = _is_better(
            selection_value, best_selection, selection_mode, es_min_delta
        )
        is_best_loss = math.isfinite(val_splot_loss) and (
            val_splot_loss < best_val_loss - es_min_delta
        )
        if improved:
            best_selection = selection_value
            best_epoch = epoch + 1
            epochs_wo_improve = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "state_dict": model.state_dict(),
                    "selection_metric": selection_metric_name,
                    "selection_value": best_selection,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                best_path,
            )
        else:
            epochs_wo_improve += 1

        if val_summary is not None and math.isfinite(
            float(val_summary["macro_pearson_r"])
        ):
            best_val_mean_r = max(
                best_val_mean_r, float(val_summary["macro_pearson_r"])
            )
        if is_best_loss:
            best_val_loss = float(val_splot_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        console.print(
            f"train_splot_loss={train_splot_loss:.5f} train_gbif_loss={train_gbif_loss:.5f} "
            f"val_splot_loss={val_splot_loss:.5f} val_gbif_loss={val_gbif_loss:.5f} "
            f"sel={selection_value:.5f} lr={current_lr:.2e}"
        )

        if cfg.wandb.enabled and wandb_module is not None:
            log_dict: dict[str, Any] = {
                "epoch": epoch + 1,
                "train/splot_loss": train_splot_loss,
                "train/gbif_loss": train_gbif_loss,
                "train/lr": current_lr,
                # Backward-compatible keys (origin/dev train_chips style)
                "val/loss": val_splot_loss,
                "val/mean_r": float("nan")
                if val_summary is None
                else float(val_summary["macro_pearson_r"]),
                # Keep current pipeline keys too
                "val/splot_loss": val_splot_loss,
                "val/gbif_loss": val_gbif_loss,
                "val/selection": selection_value,
            }
            if val_summary is not None:
                log_dict.update(
                    {
                        "val/splot/rmse": val_summary["rmse"],
                        "val/splot/r2": val_summary["r2"],
                        "val/splot/pearson_r": val_summary["pearson_r"],
                        "val/splot/macro_rmse": val_summary["macro_rmse"],
                        "val/splot/macro_pearson_r": val_summary["macro_pearson_r"],
                    }
                )
                # Exact structure used in origin/dev dashboards
                log_dict["val/per_trait_r"] = {
                    trait_name: float(m["pearson_r"])
                    for trait_name, m in val_summary["trait_metrics"].items()
                    if math.isfinite(float(m["pearson_r"]))
                }
                for trait_name, m in val_summary["trait_metrics"].items():
                    if math.isfinite(float(m["pearson_r"])):
                        log_dict[f"val/traits/{trait_name}/pearson_r"] = float(
                            m["pearson_r"]
                        )
                if improved:
                    log_dict["val/mean_r_best"] = float(val_summary["macro_pearson_r"])
            if is_best_loss:
                log_dict["val/loss_best"] = best_val_loss
            wandb_module.log(log_dict)

        if es_enabled and epochs_wo_improve >= es_patience:
            console.print(
                f"[yellow]Early stopping[/yellow] (patience={es_patience}, best_epoch={best_epoch})"
            )
            break

    console.rule("[bold cyan]DONE[/bold cyan]")
    console.print(
        f"Best {selection_metric_name}={best_selection:.6f} at epoch {best_epoch}"
    )
    console.print(f"Checkpoint: [cyan]{best_path}[/cyan]")

    if cfg.wandb.enabled and wandb_module is not None:
        run.summary["best_selection_metric_name"] = selection_metric_name
        run.summary["best_selection_metric_value"] = float(best_selection)
        run.summary["best_epoch"] = int(best_epoch)
        run.summary["checkpoint_path"] = str(best_path)
        run.summary["best_val_mean_r"] = (
            float(best_val_mean_r) if math.isfinite(best_val_mean_r) else None
        )
        run.summary["best_val_loss"] = (
            float(best_val_loss) if math.isfinite(best_val_loss) else None
        )
        run.finish()


if __name__ == "__main__":
    main()
