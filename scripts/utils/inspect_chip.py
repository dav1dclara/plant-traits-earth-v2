"""Visualize a single chip from a zarr split store.

Configure the constants below and run the script directly. Each band of each
selected predictor and target is rendered as a separate panel in a grid and
saved to viz/chips/.

Configuration:
    ZARR_PATH            Path to the chips directory (contains {split}.zarr stores).
    SPLIT                Which split store to read from: "train", "val", or "test".
    CHIP_IDX             Index of the chip to visualize.
    PREDICTORS_TO_PLOT   Predictor names to include; empty list plots all.
    TARGETS_TO_PLOT      Target dataset names to include; empty list plots all.
    TARGET_TRAITS_TO_PLOT  Trait band names to include; empty list plots all.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

ZARR_PATH = "/scratch3/plant-traits-v2/data/1km/chips/patch256_stride256/"
SPLIT = "train"
CHIP_IDX = 1221
PREDICTORS_TO_PLOT = []  # empty = plot all
TARGETS_TO_PLOT = ["splot", "gbif"]  # empty = plot all
TARGET_TRAITS_TO_PLOT = []  # empty = plot all

plt.rcParams["font.family"] = "monospace"

z = zarr.open_group(str(Path(ZARR_PATH) / f"{SPLIT}.zarr"), mode="r")

panels = []

pred_group = z["predictors"]
for arr_name, arr in sorted(pred_group.arrays()):
    if PREDICTORS_TO_PLOT and arr_name not in PREDICTORS_TO_PLOT:
        continue
    files = arr.attrs.get("files", [])
    data = arr[CHIP_IDX]  # (n_bands, H, W)
    for band_idx in range(data.shape[0]):
        file_stem = (
            Path(files[band_idx]).stem if band_idx < len(files) else str(band_idx)
        )
        panels.append((f"predictors/{arr_name}\n{file_stem}", data[band_idx]))

tgt_group = z["targets"]
# band_names is stored on the group and repeats for every source file
band_names = tgt_group.attrs.get("band_names", [])
for arr_name, arr in sorted(tgt_group.arrays()):
    if TARGETS_TO_PLOT and arr_name not in TARGETS_TO_PLOT:
        continue
    files = arr.attrs.get("files", [])
    n_bands_per_file = len(band_names) if band_names else 1
    data = arr[CHIP_IDX]  # (n_bands, H, W)
    for band_idx in range(data.shape[0]):
        file_idx = band_idx // n_bands_per_file
        band_in_file = band_idx % n_bands_per_file
        trait = Path(files[file_idx]).stem if file_idx < len(files) else str(file_idx)
        band_name = (
            band_names[band_in_file]
            if band_in_file < len(band_names)
            else str(band_in_file)
        )
        label = f"{trait} - {band_name}"
        if TARGET_TRAITS_TO_PLOT and label not in TARGET_TRAITS_TO_PLOT:
            continue
        panels.append((f"targets/{arr_name}\n{label}", data[band_idx]))

n = len(panels)
ncols = min(4, n)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for ax, (title, band) in zip(axes, panels):
    im = ax.imshow(band, cmap="viridis")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[n:]:
    ax.set_visible(False)

# Derive resolution and patch/stride config from the directory structure:
# .../data/{resolution}/chips/{patch_stride}/
_parts = Path(ZARR_PATH).parts
_resolution = _parts[-3]
_patch_stride = _parts[-1]
filename = f"viz/chips/{_resolution}_{_patch_stride}_{SPLIT}_chip{CHIP_IDX}.png"

plt.tight_layout()
plt.savefig(filename, dpi=150, bbox_inches="tight")
print(f"Saved {filename}")
