from pathlib import Path

import geopandas as gpd
import h3
from antimeridian import fix_polygon
from shapely.geometry import Polygon

OUT_DIR = Path("/scratch3/plant-traits-v2/data/temp/h3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def h3_to_polygon(cell: str) -> Polygon:
    coords = h3.cell_to_boundary(cell)
    poly = Polygon([(lon, lat) for lat, lon in coords])
    return fix_polygon(poly)


for res in range(1, 4):
    print(f"Resolution {res}...", flush=True)
    cells = list(h3.uncompact_cells(h3.get_res0_cells(), res))
    gdf = gpd.GeoDataFrame(
        {"h3_index": cells},
        geometry=[h3_to_polygon(c) for c in cells],
        crs="EPSG:4326",
    )
    out_path = OUT_DIR / f"h3_res{res}.gpkg"
    gdf.to_file(out_path, driver="GPKG")
    print(f"  {len(gdf):,} cells -> {out_path}")
