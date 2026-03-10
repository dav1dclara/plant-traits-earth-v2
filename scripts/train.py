from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch

from ptev2.data.dataset import (
    CanopyHeight,
    Modis,
    SoilGrids,
    Vodca,
    WorldClim,
    make_dataloader,
)
from ptev2.utils import seed_all


def train(cfg: DictConfig) -> None:
    # Set random seed
    seed_all()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # wandb
    # if cfg.wandb.enabled:
    #     run = wandb.init(
    #         entity=cfg.wandb.entity,
    #         project=cfg.wandb.project
    #     )
    #     print("W&B logging enabled.")
    # else:
    #     print("W&B logging disabled.")

    # Data paths  # TODO define in config later
    data_dir = Path(cfg.paths.data_root_dir)
    eo_data_dir = data_dir / "22km/eo_data/"

    # Get data loaders
    canopy_height = CanopyHeight(paths=eo_data_dir / "canopy_height")
    modis = Modis(paths=eo_data_dir / "modis")
    soilgrids = SoilGrids(paths=eo_data_dir / "soilgrids")
    vodca = Vodca(paths=eo_data_dir / "vodca")
    worldclim = WorldClim(paths=eo_data_dir / "worldclim")

    # TODO: write a function to combine the dataset to one dataset in @src/ptev2/data/dataset.py

    # Create dataloader # TODO: to revise
    dataloader = make_dataloader(canopy_height, 256, 4, 16, 0)

    # Get model
    # TODO import code from Luca

    # # print("Starting training...")

    # # finish wandb run
    # if cfg.wandb.enabled:
    #     run.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
