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
    explicit = OmegaConf.select(cfg, "training.run_name")
    if explicit:
        return str(explicit)
    loss_name = str(cfg.training.loss._target_).split(".")[-1]
    return (
        f"{cfg.models.name}_"
        f"patch({cfg.data.patch_h}x{cfg.data.patch_w})_"
        f"bs{cfg.training.data_loaders.batch_size}_"
        f"{loss_name}_"
        f"seed{cfg.training.train.seed}"
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
