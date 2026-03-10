from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch

from ptev2.data.dataset import CanopyHeight, Modis, SoilGrids, Vodca, WorldClim
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
    datasets = {
        "CanopyHeight": CanopyHeight(paths=eo_data_dir / "canopy_height"),
        "Modis": Modis(paths=eo_data_dir / "modis"),
        "SoilGrids": SoilGrids(paths=eo_data_dir / "soilgrids"),
        "Vodca": Vodca(paths=eo_data_dir / "vodca"),
        "WorldClim": WorldClim(paths=eo_data_dir / "worldclim"),
    }

    for name, ds in datasets.items():
        print(f"{name}: {len(ds.all_bands)} bands, {len(ds)} files, res={ds.res}")

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
