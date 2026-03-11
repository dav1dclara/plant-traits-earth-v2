import hydra
import torch
from omegaconf import DictConfig

from ptev2.data.dataset import create_predictors_dataset, create_targets_dataset
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

    # Instantiate EO datasets
    targets_datasets = create_targets_dataset(cfg)
    predictors_datasets = create_predictors_dataset(
        cfg
    )  # TODO: bug in intersecting datasets

    # # TODO: write a function to combine the dataset to one dataset in @src/ptev2/data/dataset.py

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
