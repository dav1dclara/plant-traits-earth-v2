import random

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
        f"seed{cfg.training.seed}"
    )
