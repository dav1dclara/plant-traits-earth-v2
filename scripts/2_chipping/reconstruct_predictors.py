"""Reconstruct the full predictor image from zarr chips for all splits and save to repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from affine import Affine

ZARR_DIR = Path("/scratch3/plant-traits-v2/data/22km/chips/patch25_stride20")
OUT_PATH = Path(__file__).parents[2] / "reconstructed_predictors.png"
SPLITS = ["train", "val", "test"]


def reconstruct(zarr_path: Path) -> tuple[np.ndarray, list[str]]:
    """Reconstruct full-canvas predictor image from chips. Returns (canvas, band_labels)."""
    z = zarr.open_group(str(zarr_path), mode="r")

    t = z.attrs["transform"]
    transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    patch_size = z.attrs["patch_size"]
    pixel_w = transform.a
    pixel_h = abs(transform.e)

    bounds = z["bounds"][:]  # (n_chips, 4): min_x, min_y, max_x, max_y
    global_min_x = bounds[:, 0].min()
    global_max_y = bounds[:, 3].max()
    global_max_x = bounds[:, 2].max()
    global_min_y = bounds[:, 1].min()

    canvas_cols = round((global_max_x - global_min_x) / pixel_w) + patch_size
    canvas_rows = round((global_max_y - global_min_y) / pixel_h) + patch_size

    n_bands_total = sum(arr.shape[1] for _, arr in z["predictors"].arrays())
    canvas = np.zeros((n_bands_total, canvas_rows, canvas_cols), dtype=np.float32)
    count = np.zeros((canvas_rows, canvas_cols), dtype=np.float32)

    band_labels = []
    band_offset = 0
    for arr_name, arr in z["predictors"].arrays():
        n_bands = arr.shape[1]
        for b in range(n_bands):
            band_labels.append(f"{arr_name} / band {b}")
        for i in range(len(bounds)):
            min_x, _, _, max_y = bounds[i]
            col = round((min_x - global_min_x) / pixel_w)
            row = round((global_max_y - max_y) / pixel_h)
            chip = arr[i]
            valid = np.isfinite(chip[0])
            canvas[
                band_offset : band_offset + n_bands,
                row : row + patch_size,
                col : col + patch_size,
            ][:, valid] += chip[:, valid]
            count[row : row + patch_size, col : col + patch_size][valid] += 1
        band_offset += n_bands

    nonzero = count > 0
    canvas[:, nonzero] /= count[nonzero]
    canvas[:, ~nonzero] = np.nan

    return canvas, band_labels


def main() -> None:
    # Reconstruct all splits first to get shared colour limits per band
    results = {}
    for split in SPLITS:
        print(f"Reconstructing {split}...")
        results[split] = reconstruct(ZARR_DIR / f"{split}.zarr")

    n_bands = results[SPLITS[0]][0].shape[0]
    band_labels = results[SPLITS[0]][1]

    # Shared vmin/vmax per band across all splits
    vmins = [min(np.nanmin(results[s][0][b]) for s in SPLITS) for b in range(n_bands)]
    vmaxs = [max(np.nanmax(results[s][0][b]) for s in SPLITS) for b in range(n_bands)]

    fig, axes = plt.subplots(
        len(SPLITS), n_bands, figsize=(5 * n_bands, 4 * len(SPLITS))
    )
    axes = np.array(axes).reshape(len(SPLITS), n_bands)

    for row_idx, split in enumerate(SPLITS):
        canvas, _ = results[split]
        for b in range(n_bands):
            ax = axes[row_idx, b]
            im = ax.imshow(
                canvas[b], cmap="viridis", origin="upper", vmin=vmins[b], vmax=vmaxs[b]
            )
            if row_idx == 0:
                ax.set_title(band_labels[b], fontsize=10)
            ax.set_ylabel(split if b == 0 else "", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Reconstructed predictors — train / val / test", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
