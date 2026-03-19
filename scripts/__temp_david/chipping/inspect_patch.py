import matplotlib.pyplot as plt
import numpy as np
import zarr

ZARR_PATH = "/scratch3/plant-traits-v2/data/chips/22km/patch15_stride10/train.zarr"
CHIP_IDX = 2490

z = zarr.open_group(ZARR_PATH, mode="r")

# Collect all (group/name, data) pairs for this chip
panels = []
for group_name, group in z.groups():
    for arr_name, arr in group.arrays():
        data = arr[CHIP_IDX]  # (n_bands, H, W)
        for band_idx in range(data.shape[0]):
            panels.append((f"{group_name}/{arr_name}\nband {band_idx}", data[band_idx]))

n = len(panels)
ncols = min(4, n)
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for ax, (title, band) in zip(axes, panels):
    im = ax.imshow(band, cmap="viridis")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle(f"Chip {CHIP_IDX} — {ZARR_PATH.split('/')[-2]}/train.zarr", fontsize=11)
plt.tight_layout()
plt.savefig(f"chip_{CHIP_IDX}.png", dpi=150, bbox_inches="tight")
print(f"Saved chip_{CHIP_IDX}.png")
