from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader
from ptev2.models.traitPatchCNN import ResPatchCenterCNN

ZARR_DIR = Path("/scratch3/plant-traits-v2/data/chips/22km/patch15_stride10")
PREDICTORS = ["canopy_height", "modis", "soil_grids", "vodca", "worldclim"]
TARGET = "gbif"
BATCH_SIZE = 32
N_EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dl = get_dataloader(
    ZARR_DIR,
    split="train",
    predictors=PREDICTORS,
    target=TARGET,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

# Infer in_channels and n_traits from a sample batch
X_sample, y_sample = next(iter(dl))
in_channels = X_sample.shape[1]
n_traits = y_sample.shape[1]
patch_size = X_sample.shape[-1]
print(f"X: {X_sample.shape}, y: {y_sample.shape}, device: {DEVICE}")

for epoch in range(N_EPOCHS):
    total_loss = 0.0

    for X, y in tqdm(dl, desc=f"Epoch {epoch + 1}/{N_EPOCHS}"):
        X = X.to(DEVICE)

    print(f"Epoch {epoch + 1}/{N_EPOCHS}  loss={total_loss / len(dl):.4f}")
