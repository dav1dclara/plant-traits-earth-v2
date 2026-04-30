# TODO: FIX THE COLORBARS (SEE DANIEL'S PAPER)

"""Plot a single trait's observation count (sPlot and GBIF) on a global map.

Reads the count band of the specified trait raster, reprojects to EPSG:4326
at display resolution, and saves a single side-by-side figure.

Usage:
    python plot_trait_map.py
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from matplotlib.colors import LogNorm
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

TRAIT = "X3106"
TARGETS_DIR = Path("/scratch3/plant-traits-v2/data/1km/targets")
OUT_PATH = (
    Path(__file__).parents[3] / "viz/data/targets" / "targets_distribution_map.png"
)

DISPLAY_SHAPE = (900, 1800)  # (rows, cols) per panel
DST_CRS = "EPSG:4326"
BAND = "count"
CMAP = "viridis"

plt.rcParams["font.family"] = "monospace"


def load_for_display(
    path: Path, band_name: str = BAND
) -> tuple[np.ma.MaskedArray, tuple]:
    """Read a raster band at display resolution, reprojected to EPSG:4326."""
    with rasterio.open(path) as src:
        band_idx = [d.lower() for d in src.descriptions].index(band_name) + 1
        read_shape = (
            max(1, int(src.height * DISPLAY_SHAPE[0] / src.height)),
            max(1, int(src.width * DISPLAY_SHAPE[1] / src.width)),
        )
        data = src.read(
            band_idx, out_shape=read_shape, resampling=Resampling.average
        ).astype(np.float32)
        src_transform = from_bounds(*src.bounds, read_shape[1], read_shape[0])
        src_crs = src.crs

    bounds_4326 = rasterio.warp.transform_bounds(
        src_crs, DST_CRS, *rasterio.transform.array_bounds(*read_shape, src_transform)
    )
    dst_transform = from_bounds(*bounds_4326, DISPLAY_SHAPE[1], DISPLAY_SHAPE[0])

    dst = np.full(DISPLAY_SHAPE, np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=DST_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return np.ma.masked_invalid(dst), bounds_4326


print(f"Loading GBIF {TRAIT}...")
gbif_data, gbif_bounds = load_for_display(TARGETS_DIR / "gbif" / f"{TRAIT}.tif")
print(f"Loading sPlot {TRAIT}...")
splot_data, splot_bounds = load_for_display(TARGETS_DIR / "splot" / f"{TRAIT}.tif")

proj = ccrs.EqualEarth()
fig, axes = plt.subplots(1, 2, figsize=(16, 4), subplot_kw={"projection": proj})

vmin = max(1, min(gbif_data.min(), splot_data.min()))
vmax = max(gbif_data.max(), splot_data.max())
norm = LogNorm(vmin=vmin, vmax=vmax)

for ax, data, bounds, title, cbar_label in [
    (axes[0], gbif_data, gbif_bounds, "GBIF", "Number of GBIF observations"),
    (axes[1], splot_data, splot_bounds, "sPlot", "Number of sPlot vegetation surveys"),
]:
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    west, south, east, north = bounds
    im = ax.imshow(
        data,
        origin="upper",
        extent=[west, east, south, north],
        transform=ccrs.PlateCarree(),
        cmap=CMAP,
        interpolation="none",
        norm=norm,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555")
    ax.set_global()
    ax.spines["geo"].set_linewidth(0.5)
    cbar = fig.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.4, aspect=50
    )
    cbar.set_label(cbar_label)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(wspace=-0.2)
fig.savefig(
    OUT_PATH,
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compress_level": 9, "optimize": True},
)
print(f"Saved {OUT_PATH}")
