"""
Training script for Plant Traits Earth V2 with Hydra configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim


@hydra.main(version_base=None, config_path="../config", config_name="models")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Instantiate model using Hydra
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    print("\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    _ = optim.Adam(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )

    # Setup loss function
    _ = nn.MSELoss()

    print("\n" + "=" * 80)
    print("Model and optimizer initialized successfully!")
    print("=" * 80)

    # TODO: Add data loading
    # TODO: Add training loop
    # TODO: Add validation
    # TODO: Add checkpointing
    # TODO: Add logging (wandb/tensorboard)

    print("\n⚠️  Training loop not yet implemented.")
    print("Next steps:")
    print("  1. Implement data loading")
    print("  2. Implement training loop")
    print("  3. Add validation and metrics")
    print("  4. Add checkpointing")


if __name__ == "__main__":
    main()
