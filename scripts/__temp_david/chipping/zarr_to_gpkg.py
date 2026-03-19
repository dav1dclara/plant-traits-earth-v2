from pathlib import Path

import geopandas as gpd
import zarr
from shapely.geometry import box

zarr_dir = Path("/scratch3/plant-traits-v2/data/chips/22km/patch15_stride10")
output_dir = Path("/scratch3/plant-traits-v2/data/temp/chips")
output_dir.mkdir(parents=True, exist_ok=True)

resolution = zarr_dir.parts[-2]  # e.g. "22km"
patch_stride = zarr_dir.parts[-1]  # e.g. "patch15_stride10"

for zarr_path in sorted(zarr_dir.glob("*.zarr")):
    split = zarr_path.stem  # "train", "val", "test"
    z = zarr.open_group(str(zarr_path), mode="r")

    bounds = z["bounds"][:]  # (n_chips, 4): [min_x, min_y, max_x, max_y] in EPSG:6933
    geometries = [
        box(min_x, min_y, max_x, max_y) for min_x, min_y, max_x, max_y in bounds
    ]
    gdf = gpd.GeoDataFrame(
        {"chip_id": range(len(bounds))}, geometry=geometries, crs="EPSG:6933"
    )

    output = output_dir / f"{resolution}_{patch_stride}_{split}.gpkg"
    gdf.to_file(output, driver="GPKG")
    print(f"Saved {output} ({len(gdf):,} chips)")
