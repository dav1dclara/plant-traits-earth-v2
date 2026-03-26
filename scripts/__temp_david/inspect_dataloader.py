from pathlib import Path

from rich.console import Console

from ptev2.data.dataloader import get_dataloader

console = Console()

ZARR_DIR = Path("/scratch3/plant-traits-v2/data/22km/chips/patch15_stride10")
PREDICTORS = ["canopy_height", "modis", "soil_grids", "vodca", "worldclim"]
TARGET = "comb"
SPLIT = "val"
CHIP_IDX = 1975

dl = get_dataloader(
    ZARR_DIR,
    split=SPLIT,
    predictors=PREDICTORS,
    target=TARGET,
    batch_size=1,
    num_workers=0,
)

console.print(f"Dataset size: {len(dl.dataset)} chips")

X, y = dl.dataset[CHIP_IDX]
console.print(f"[bold]X[/bold]: shape={X.shape}, dtype={X.dtype}")
console.print(X[0])
console.print(f"[bold]y[/bold]: shape={y.shape}, dtype={y.dtype}")
console.print(y[0])
