from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader
from ptev2.utils import (
    run_name_from_cfg,
    seed_all,
)


def _band_names_for_source(source: str, bands_per_trait: int) -> list[str]:
    source_l = source.lower()
    if bands_per_trait == 6:
        if source_l == "gbif":
            return ["mean", "std", "median", "q05", "q95", "count"]
        if source_l == "splot":
            return ["mean", "count", "std", "median", "q05", "q95"]
    return [f"band{i}" for i in range(1, bands_per_trait + 1)]


def train_model(cfg: DictConfig) -> float:
    seed_all(cfg.training.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    run_name = run_name_from_cfg(cfg)
    print(f"Run name: {run_name}")

    run = None
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=False),
            reinit="finish_previous",
        )

    predictors = [k for k, v in cfg.training.data.predictors.items() if v.use]
    if not predictors:
        raise ValueError("No predictors enabled in cfg.training.data.predictors.")

    target = cfg.training.data.target.source
    zarr_dir = Path(cfg.training.data.zarr_dir)
    batch_size = int(cfg.training.data_loaders.batch_size)
    num_workers = int(cfg.training.data_loaders.num_workers)

    trait_ids_cfg = OmegaConf.select(cfg, "training.data.target.trait_ids")
    trait_positions_cfg = OmegaConf.select(cfg, "training.data.target.trait_positions")
    bands_per_trait = int(
        OmegaConf.select(cfg, "training.data.target.bands_per_trait", default=6)
    )

    selected_trait_ids = (
        [int(trait_ids_cfg)]
        if isinstance(trait_ids_cfg, (str, int))
        else [int(t) for t in trait_ids_cfg]
    )
    trait_positions = (
        [int(trait_positions_cfg)]
        if isinstance(trait_positions_cfg, int)
        else [int(p) for p in trait_positions_cfg]
    )
    if len(selected_trait_ids) != len(trait_positions):
        raise ValueError(
            "training.data.target.trait_ids und trait_positions muessen gleich lang sein."
        )

    target_channel_indices: list[int] = []
    for pos in trait_positions:
        start = pos * bands_per_trait
        target_channel_indices.extend(range(start, start + bands_per_trait))

    selected_trait_count = len(selected_trait_ids)
    effective_target_channels = len(target_channel_indices)
    band_names = _band_names_for_source(target, bands_per_trait)

    print(
        "Target layout: "
        f"traits={selected_trait_count}, "
        f"bands_per_trait={bands_per_trait}, "
        f"output_channels={effective_target_channels}"
    )
    print(f"Band order ({target}): {band_names}")

    train_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=cfg.training.data.train_split,
        predictors=predictors,
        target=target,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        zarr_dir=zarr_dir,
        split=cfg.training.data.val_split,
        predictors=predictors,
        target=target,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = instantiate(cfg.models.active, out_channels=effective_target_channels).to(
        device
    )
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    loss_fn = instantiate(cfg.training.loss)
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer)

    final_val_loss = float("nan")
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_skipped_batches = 0
        printed_shapes = False
        for X, y in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [train]"
        ):
            X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
            y = y.to(device=device, dtype=torch.float32)

            # Filter trait channels if configured
            if target_channel_indices is not None:
                y = y[:, target_channel_indices]

            valid_samples = torch.isfinite(y).any(dim=(1, 2, 3))
            if not bool(valid_samples.any()):
                train_skipped_batches += 1
                continue
            X = X[valid_samples]
            y = y[valid_samples]

            if not printed_shapes:
                print(f"Batch shapes: X={tuple(X.shape)}, y={tuple(y.shape)}")
                printed_shapes = True

            optimizer.zero_grad()
            y_pred = model(X)
            if y_pred.shape != y.shape:
                raise ValueError(
                    f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                )

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu().item())
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_skipped_batches = 0
        with torch.no_grad():
            for X, y in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [val]"
            ):
                X = torch.nan_to_num(X.to(device=device, dtype=torch.float32))
                y = y.to(device=device, dtype=torch.float32)

                # Filter trait channels if configured
                if target_channel_indices is not None:
                    y = y[:, target_channel_indices]

                valid_samples = torch.isfinite(y).any(dim=(1, 2, 3))
                if not bool(valid_samples.any()):
                    val_skipped_batches += 1
                    continue
                X = X[valid_samples]
                y = y[valid_samples]

                y_pred = model(X)
                if y_pred.shape != y.shape:
                    raise ValueError(
                        f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs y={tuple(y.shape)}"
                    )

                loss = loss_fn(y_pred, y)
                val_loss_sum += float(loss.detach().cpu().item())
                val_batches += 1

        train_loss = (
            train_loss_sum / train_batches if train_batches > 0 else float("nan")
        )
        val_loss = val_loss_sum / val_batches if val_batches > 0 else float("nan")
        final_val_loss = val_loss
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} - "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"skipped(train/val)={train_skipped_batches}/{val_skipped_batches}"
        )
        if run is not None:
            wandb.log(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )

    save_model = OmegaConf.select(cfg, "training.checkpoint.save_model")
    if save_model is None:
        save_model = True
    checkpoint_dir_cfg = OmegaConf.select(cfg, "training.checkpoint.dir")

    if save_model:
        checkpoint_dir = (
            Path(checkpoint_dir_cfg)
            if checkpoint_dir_cfg
            else Path.cwd() / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.chmod(0o777)
        checkpoint_path = checkpoint_dir / f"{run_name}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_path.chmod(0o777)
        print(f"Saved model to: {checkpoint_path}")

    if run is not None:
        run.finish()

    return final_val_loss


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    final_val_loss = train_model(cfg)
    print(f"Training completed. Final validation loss: {final_val_loss:.4f}")


if __name__ == "__main__":
    main()
