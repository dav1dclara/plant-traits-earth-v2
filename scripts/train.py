import hydra
import torch
from omegaconf import DictConfig

from pathlib import Path

from ptev2.utils import seed_all
from ptev2.data.dataloader import get_train_dataloader, get_val_dataloader


def train(cfg: DictConfig) -> None:
    # Set random seed
    seed_all()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # # wandb
    # if cfg.wandb.enabled:
    #     run = wandb.init(
    #         entity=cfg.wandb.entity,
    #         project=cfg.wandb.project
    #     )
    #     print("W&B logging enabled.")
    # else:
    #     print("W&B logging disabled.")

    # Get data loaders
    # Later, specify the data properties in the training config and
    res = cfg.training.data.res
    targets = cfg.training.data.targets
    predictors = cfg.training.data.predictors
    targets = cfg.training.data.targets

    print("--- DATA PROPERTIES ---")
    print(f"Resolution:\n  - {res} km")
    print("Predictors:")
    for name, predictor in predictors.items():
        print(f"  - {name}")
    print(f"Targets:\n  - {targets.source}")
    # TODO: print traits

    print()

    print("--- DATA LOADERS ---")
    data_loader_cfg = cfg.training.data_loaders
    batch_size = data_loader_cfg.batch_size
    zarr_path = Path(f"/scratch3/plant-traits-v2/data/chips/{res}km/")

    # TODO: to implement
    get_train_dataloader(zarr_path, batch_size)
    get_val_dataloader(zarr_path, batch_size)

    # # Get model
    # # TODO import code from Luca

    # # # print("Starting training...")

    # # # finish wandb run
    # # if cfg.wandb.enabled:
    # #     run.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
