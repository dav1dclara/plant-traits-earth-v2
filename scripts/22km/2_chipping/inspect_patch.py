from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

ZARR_PATH = "/scratch3/plant-traits-v2/data/22km/chips/patch25_stride25/train.zarr"
CHIP_IDX = 532

# Filter which arrays to plot. Set to None to plot all, or a list of array names to restrict.
# Target arrays:    ["splot", "gbif"]
# Predictor arrays: ["modis", "canopy_height", "soil_grids", "vodca", "worldclim"]
TARGETS_TO_PLOT = None  # e.g. ["splot", "gbif"]
PREDICTORS_TO_PLOT = ["canopy_height"]  # e.g. ["modis", "canopy_height"]

# Filter which target traits to plot by file stem. Set to None to plot all.
TARGET_TRAITS_TO_PLOT = ["X1080", "X13"]  # e.g. ["X1080", "X13"]

plt.rcParams["font.family"] = "monospace"

z = zarr.open_group(ZARR_PATH, mode="r")

# Collect all (title, data) pairs for this chip
panels = []
for group_name, group in sorted(z.groups(), key=lambda g: g[0] != "predictors"):
    if group_name == "targets" and TARGETS_TO_PLOT is not None:
        arrays = [(n, a) for n, a in group.arrays() if n in TARGETS_TO_PLOT]
    elif group_name == "predictors" and PREDICTORS_TO_PLOT is not None:
        arrays = [(n, a) for n, a in group.arrays() if n in PREDICTORS_TO_PLOT]
    else:
        arrays = list(group.arrays())

    for arr_name, arr in arrays:
        files = arr.attrs.get("files", [])
        data = arr[CHIP_IDX]  # (n_bands, H, W)

        if group_name == "targets":
            band_names = arr.attrs.get("band_names", [])
            n_bands_per_file = len(band_names) if band_names else 1
            for band_idx in range(data.shape[0]):
                file_idx = band_idx // n_bands_per_file
                band_pos = band_idx % n_bands_per_file
                file_stem = (
                    Path(files[file_idx]).stem
                    if file_idx < len(files)
                    else str(file_idx)
                )
                if (
                    TARGET_TRAITS_TO_PLOT is not None
                    and file_stem not in TARGET_TRAITS_TO_PLOT
                ):
                    continue
                band_name = (
                    band_names[band_pos]
                    if band_pos < len(band_names)
                    else str(band_pos)
                )
                panels.append((f"{arr_name}/{file_stem}\n{band_name}", data[band_idx]))
        else:
            for band_idx in range(data.shape[0]):
                file_stem = (
                    Path(files[band_idx]).stem
                    if band_idx < len(files)
                    else str(band_idx)
                )
                panels.append((f"{arr_name}\n{file_stem}", data[band_idx]))

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

plt.tight_layout()
split = ZARR_PATH.rstrip("/").split("/")[-1].replace(".zarr", "")
plt.savefig(f"viz/chips/chip_{split}_{CHIP_IDX}.png", dpi=150, bbox_inches="tight")
print(f"Saved chip_{split}_{CHIP_IDX}.png")
