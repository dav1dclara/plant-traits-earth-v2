import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name_from_cfg(cfg: DictConfig) -> str:
    train_cfg = cfg.training.train
    explicit = train_cfg.run_name
    if explicit:
        return str(explicit)
    seed = train_cfg.seed
    loss_name = str(train_cfg.loss._target_).split(".")[-1]
    patch_h = patch_w = cfg.training.data_loaders.patch_size
    return (
        f"{cfg.models.name}_"
        f"patch({patch_h}x{patch_w})_"
        f"bs{cfg.training.data_loaders.batch_size}_"
        f"{loss_name}_"
        f"seed{seed}"
    )


def checkpoint_paths_from_cfg(
    cfg: DictConfig,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    if run_name is None:
        run_name = run_name_from_cfg(cfg)

    checkpoint_dir_cfg = OmegaConf.select(cfg, "training.checkpoint.dir")
    checkpoint_dir = (
        Path(str(checkpoint_dir_cfg))
        if checkpoint_dir_cfg
        else Path.cwd() / "checkpoints"
    )

    best_filename = OmegaConf.select(cfg, "training.checkpoint.best_filename")
    last_filename = OmegaConf.select(cfg, "training.checkpoint.last_filename")
    best_name = str(best_filename) if best_filename else f"{run_name}_best.pth"
    last_name = str(last_filename) if last_filename else f"{run_name}.pth"

    return checkpoint_dir / best_name, checkpoint_dir / last_name
