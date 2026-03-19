from pathlib import Path

from tqdm import tqdm

from ptev2.data.dataloader import get_dataloader

ZARR_DIR = Path("/scratch3/plant-traits-v2/data/chips/22km/patch15_stride10")
PREDICTORS = ["canopy_height", "modis", "soil_grids", "vodca", "worldclim"]
TARGET = "gbif"
BATCH_SIZE = 32

dl = get_dataloader(
    ZARR_DIR,
    split="train",
    predictors=PREDICTORS,
    target=TARGET,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

X, y = next(iter(dl))
print(f"X: {X.shape}, dtype={X.dtype}")
print(f"y: {y.shape}, dtype={y.dtype}")
print(f"Dataset size: {len(dl.dataset)} chips")

print("Iterating through all train batches...")
for i, (X, y) in enumerate(tqdm(dl)):
    pass
print(f"Done. {i + 1} batches.")
