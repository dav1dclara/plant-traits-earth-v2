"""
For each H3 resolution and each trait split, assign the fold index of the
overlapping points to each H3 cell. Cells with no points get fold=NaN.
Cells where points disagree on fold are flagged to the console.

Output: /scratch3/plant-traits-v2/data/temp/h3_splits/h3_res<N>_<trait>.gpkg
"""

from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from antimeridian import fix_polygon
from shapely.geometry import Point, Polygon

PARQUET_DIR = Path("/scratch3/plant-traits-v2/data/22km/skcv_splits")
OUT_DIR = Path("/scratch3/plant-traits-v2/data/temp/h3_splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = [1]


def h3_to_polygon(cell: str) -> Polygon:
    coords = h3.cell_to_boundary(cell)
    poly = Polygon([(lon, lat) for lat, lon in coords])
    return fix_polygon(poly)


def make_h3_gdf(res: int) -> gpd.GeoDataFrame:
    cells = list(h3.uncompact_cells(h3.get_res0_cells(), res))
    return gpd.GeoDataFrame(
        {"h3_index": cells},
        geometry=[h3_to_polygon(c) for c in cells],
        crs="EPSG:4326",
    ).to_crs("EPSG:6933")


parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

for res in RESOLUTIONS:
    print(f"\n=== Resolution {res} ===")
    h3_gdf = make_h3_gdf(res)

    for parquet_file in parquet_files:
        trait = parquet_file.stem
        df = pd.read_parquet(parquet_file)
        points = gpd.GeoDataFrame(
            df[["fold"]],
            geometry=[Point(x, y) for x, y in zip(df["x"], df["y"])],
            crs="EPSG:6933",
        )

        # Spatial join: find which H3 cell each point falls in
        joined = gpd.sjoin(
            points, h3_gdf[["h3_index", "geometry"]], how="left", predicate="within"
        )

        # Check for cells where points disagree on fold
        grouped = joined.groupby("h3_index")["fold"]
        inconsistent = grouped.filter(lambda g: g.nunique() > 1)
        if not inconsistent.empty:
            bad_cells = joined.loc[inconsistent.index, "h3_index"].unique()
            print(
                f"[{trait}] {len(bad_cells)} cell(s) ({len(bad_cells) / len(h3_gdf):.2%}) with mixed folds"
            )

        # Aggregate: take the first fold value per cell (after consistency check)
        fold_per_cell = grouped.first().rename("fold")

        # Merge back onto H3 grid (unmatched cells get NaN)
        result = h3_gdf.merge(fold_per_cell, on="h3_index", how="left")
        result["fold"] = result["fold"].astype("float32")
        result["split"] = result["fold"].map(
            {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}
        )

        out_path = OUT_DIR / f"h3_res{res}_{trait}.gpkg"
        result.to_file(out_path, driver="GPKG")
        n_assigned = result["fold"].notna().sum()
        print(
            f"[{trait}] {n_assigned:,}/{len(result):,} cells assigned -> {out_path.name}\n"
        )
