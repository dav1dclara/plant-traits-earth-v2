"""Plot H3 split cells on a world map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SPLIT_FILE = "/scratch3/plant-traits-v2/data/1km/splits/h3_splits_res2_1km.gpkg"
OUT_PATH = Path(__file__).parents[3] / "viz/data/splitting/1km/split_map.png"

SPLIT_COLORS = {
    "train": "#2196F3",
    "val": "#FF9800",
    "test": "#4CAF50",
    "none": "#9E9E9E",
}

gdf = gpd.read_file(SPLIT_FILE).to_crs("EPSG:4326")

proj = ccrs.EqualEarth()
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555")

for split, color in SPLIT_COLORS.items():
    sub = gdf[gdf["split"] == split]
    if not len(sub):
        continue
    sub.plot(
        ax=ax,
        color=color,
        edgecolor=color,
        linewidth=0.5,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )

plt.rcParams["font.family"] = "monospace"

handles = [
    mpatches.Patch(color=color, label=split)
    for split, color in SPLIT_COLORS.items()
    if (gdf["split"] == split).sum() > 0
]
ax.set_global()
ax.spines["geo"].set_linewidth(0.5)
ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=len(handles),
    frameon=True,
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(
    OUT_PATH,
    dpi=300,
    bbox_inches="tight",
    pil_kwargs={"compress_level": 9, "optimize": True},
)
print(f"Saved {OUT_PATH}")
