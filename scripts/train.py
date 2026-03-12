import hydra
import torch
from omegaconf import DictConfig

from pathlib import Path

from ptev2.utils import seed_all
from ptev2.data.dataloader import get_dataloader


def train(cfg: DictConfig) -> None:
    # Set random seed
    seed_all()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # # wandb - Just uncomment to enable
    # if cfg.wandb.enabled:
    #     run = wandb.init(
    #         entity=cfg.wandb.entity,
    #         project=cfg.wandb.project
    #     )
    #     print("W&B logging enabled.")
    # else:
    #     print("W&B logging disabled.")

    # Get data configuration for training
    print("--- DATA PROPERTIES ---")
    training_data_cfg = cfg.training.data
    res = training_data_cfg.res
    targets = training_data_cfg.targets.source
    predictors = training_data_cfg.predictors
    used_predictors = [name for name, cfg in predictors.items() if cfg.use]

    print(f"Resolution:          {res} km")
    print(f"Targets used:        {targets}")
    print("Predictors used:")
    for name in used_predictors:
        print(f"  - {name}")

    print()

    # Get data loaders
    print("--- DATA LOADERS ---")
    zarr_dir = Path(
        f"/scratch3/plant-traits-v2/data/chips/{res}km/"
    )  # TODO: to specify in config later
    data_loader_cfg = cfg.training.data_loaders
    batch_size = data_loader_cfg.batch_size
    num_workers = data_loader_cfg.num_workers

    print(f"Batch size:          {batch_size}")
    print(f"Number of workers:   {num_workers}")

    train_loader = get_dataloader(
        zarr_dir,
        split="train",
        predictors=used_predictors,
        target=targets,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = get_dataloader(
        zarr_dir,
        split="val",
        predictors=used_predictors,
        target=targets,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    a = 0

    # # Get model
    # # TODO import code from Luca

    # # # print("Starting training...")

    # # finish wandb run
    # if cfg.wandb.enabled:
    #     run.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
