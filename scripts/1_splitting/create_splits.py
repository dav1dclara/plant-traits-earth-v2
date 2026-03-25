"""
Create a unified train/val/test split (single H3 GeoPackage) usable for all traits.

Algorithm:
1. Generate all H3 cells at resolution 1
2. For each H3 cell, intersect with each GBIF trait raster to get all pixel values
   within the cell polygon (zonal statistics)
3. Exclude cells with no valid pixel values across all traits
4. Pre-compute per-cell histograms per trait over shared bin ranges
5. Randomly assign cells to train/val/test (~70/20/10) using random restarts,
   minimising mean Jensen-Shannon divergence between the pooled per-split distributions
6. Save result as a GeoPackage with columns: h3_index, split, geometry

Output: /scratch3/plant-traits-v2/data/temp/h3_splits/h3_unified.gpkg
"""

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import rasterio
import rasterio.features
from antimeridian import fix_polygon
from pyproj import Transformer
from rich.console import Console
from rich.progress import track
from scipy.spatial.distance import jensenshannon
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

GBIF_DIR = Path("/scratch3/plant-traits-v2/data/22km/gbif")
OUT_DIR = Path("/scratch3/plant-traits-v2/data/22km/splits")

BANDS_FOR_JSD = [1, 2, 3, 4, 5]  # mean, std, median, q05, q95 (skip band 6 = count)

H3_RESOLUTION = 2
SOURCE = "gbif"
DATA_RES = "22km"
OUT_FILE = OUT_DIR / f"h3_splits_res{H3_RESOLUTION}_{SOURCE}_{DATA_RES}.gpkg"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
N_RESTARTS = 200
N_BINS = 50
RANDOM_SEED = 42

console = Console()


def h3_to_polygon_4326(cell: str) -> Polygon:
    coords = h3.cell_to_boundary(cell)
    return fix_polygon(Polygon([(lon, lat) for lat, lon in coords]))


def reproject_polygon(poly: Polygon, src_crs: str, dst_crs: str) -> Polygon:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, poly)


def extract_cell_values(
    raster_path: Path, cell_polygons_raster_crs: list[Polygon]
) -> list[list[np.ndarray]]:
    """
    For each H3 cell polygon (in the raster's CRS), extract valid pixel values
    for each band in BANDS_FOR_JSD.

    Returns: cell_values[cell_idx][band_idx] = 1-D array of pixel values.
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        shapes = [
            (geom.__geo_interface__, idx + 1)
            for idx, geom in enumerate(cell_polygons_raster_crs)
            if geom is not None
        ]
        cell_labels = rasterio.features.rasterize(
            shapes,
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype=np.int32,
        )

        band_arrays = []
        for band in BANDS_FOR_JSD:
            data = src.read(band).astype(float)
            if nodata is not None:
                data[data == nodata] = np.nan
            band_arrays.append(data)

    cell_values = []
    for idx in range(len(cell_polygons_raster_crs)):
        mask = cell_labels == (idx + 1)
        cell_values.append(
            [(lambda v: v[np.isfinite(v)])(band[mask]) for band in band_arrays]
        )

    return cell_values


def build_histograms(
    all_cell_values: list[list[np.ndarray]],  # [n_cells][n_traits]
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Pre-compute per-cell histograms for each trait over shared bin edges.

    Returns:
        histograms: float array of shape (n_cells, n_traits, n_bins)
        bin_edges:  list of length n_traits, each an array of shape (n_bins+1,)
    """
    n_cells = len(all_cell_values)
    n_traits = len(all_cell_values[0])
    histograms = np.zeros((n_cells, n_traits, n_bins), dtype=float)
    bin_edges = []

    for t in range(n_traits):
        # Global min/max for this trait across all cells
        all_vals = np.concatenate(
            [
                all_cell_values[c][t]
                for c in range(n_cells)
                if len(all_cell_values[c][t]) > 0
            ]
        )
        if len(all_vals) == 0 or all_vals.min() == all_vals.max():
            bin_edges.append(None)
            continue
        edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
        bin_edges.append(edges)
        for c in range(n_cells):
            vals = all_cell_values[c][t]
            if len(vals) > 0:
                h, _ = np.histogram(vals, bins=edges)
                histograms[c, t] = h.astype(float)

    return histograms, bin_edges


def compute_total_jsd(
    histograms: np.ndarray,  # (n_cells, n_traits, n_bins)
    labels: np.ndarray,  # (n_cells,) with values 0/1/2
    bin_edges: list,
) -> float:
    """
    Compute mean pairwise JSD between split distributions, across all valid traits.
    Histograms are summed across cells per split before comparing.
    """
    scores = []
    for t, edges in enumerate(bin_edges):
        if edges is None:
            continue
        split_hists = []
        for s in range(3):
            h = histograms[labels == s, t].sum(axis=0) + 1e-10
            split_hists.append(h / h.sum())
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            scores.append(float(jensenshannon(split_hists[i], split_hists[j])))
    return float(np.mean(scores)) if scores else 1.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Generate all H3 cells at resolution 1
    # ------------------------------------------------------------------
    console.rule("[bold]Step 1: Generating H3 cells[/bold]")
    all_cells = sorted(h3.uncompact_cells(h3.get_res0_cells(), H3_RESOLUTION))
    console.print(f"Total H3 cells at resolution {H3_RESOLUTION}: {len(all_cells):,}")

    # Build polygons in EPSG:4326 (for output) and raster CRS (for intersection)
    polys_4326 = [h3_to_polygon_4326(c) for c in all_cells]

    # Determine raster CRS from first file
    gbif_rasters = sorted(GBIF_DIR.glob("*.tif"))
    console.print(f"Found {len(gbif_rasters)} GBIF rasters")
    with rasterio.open(gbif_rasters[0]) as src:
        raster_crs = src.crs.to_string()
    console.print(f"Raster CRS: {raster_crs}")

    polys_raster_crs = [
        reproject_polygon(p, "EPSG:4326", raster_crs) for p in polys_4326
    ]

    # ------------------------------------------------------------------
    # Step 2: Intersect each raster with all H3 cell polygons
    # ------------------------------------------------------------------
    console.rule("[bold]Step 2: Extracting pixel values per cell per trait[/bold]")

    # all_cell_values[cell_idx] = flat list of arrays, indexed by (trait * n_bands + band)
    # where band indexes into BANDS_FOR_JSD
    n_bands = len(BANDS_FOR_JSD)
    n_cells_all = len(all_cells)
    n_features = len(gbif_rasters) * n_bands
    all_cell_values = [[None] * n_features for _ in range(n_cells_all)]

    for t, raster_path in enumerate(
        track(gbif_rasters, description="Extracting pixel values...")
    ):
        cell_vals = extract_cell_values(raster_path, polys_raster_crs)
        for c, band_vals in enumerate(cell_vals):
            for b, vals in enumerate(band_vals):
                all_cell_values[c][t * n_bands + b] = vals

    # ------------------------------------------------------------------
    # Step 3: Exclude cells with no valid data across all traits
    # ------------------------------------------------------------------
    console.rule("[bold]Step 3: Filtering cells[/bold]")
    keep = [
        any(len(all_cell_values[c][f]) > 0 for f in range(n_features))
        for c in range(n_cells_all)
    ]

    trait_names = [r.stem for r in gbif_rasters]

    h3_cells = [cell for cell, k in zip(all_cells, keep) if k]
    polys_4326_kept = [p for p, k in zip(polys_4326, keep) if k]
    all_cell_values = [v for v, k in zip(all_cell_values, keep) if k]
    n_cells = len(h3_cells)
    console.print(f"Cells with data: {n_cells:,} / {n_cells_all:,}")

    count_data = {
        name: [len(all_cell_values[c][t * n_bands]) for c in range(n_cells)]
        for t, name in enumerate(trait_names)
    }
    n_traits_with_obs = [
        sum(
            1
            for t in range(len(trait_names))
            if len(all_cell_values[c][t * n_bands]) > 0
        )
        for c in range(n_cells)
    ]

    # ------------------------------------------------------------------
    # Step 4: Pre-compute histograms per cell per (trait × band) feature
    # ------------------------------------------------------------------
    console.rule("[bold]Step 4: Pre-computing histograms[/bold]")
    histograms, bin_edges = build_histograms(all_cell_values, n_bins=N_BINS)
    valid_features = sum(1 for e in bin_edges if e is not None)
    console.print(
        f"Features with valid data range: {valid_features} / {n_features} ({len(gbif_rasters)} traits × {n_bands} bands)"
    )

    # ------------------------------------------------------------------
    # Step 5: Optimise split assignment via random restarts
    # ------------------------------------------------------------------
    console.rule("[bold]Step 5: Optimising split assignment[/bold]")
    n_train = round(TRAIN_FRAC * n_cells)
    n_val = round(VAL_FRAC * n_cells)
    n_test = n_cells - n_train - n_val
    console.print(f"Target: train={n_train}  val={n_val}  test={n_test}")
    console.print(f"Running {N_RESTARTS} random restarts...")

    base_labels = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    best_labels = rng.permutation(base_labels)
    best_score = compute_total_jsd(histograms, best_labels, bin_edges)

    for seed in range(1, N_RESTARTS):
        labels = rng.permutation(base_labels)
        score = compute_total_jsd(histograms, labels, bin_edges)
        if score < best_score:
            best_score = score
            best_labels = labels.copy()
            console.print(f"  restart {seed:>4d}  →  best JSD = {best_score:.6f}")

    console.print(f"\nFinal best mean JSD: {best_score:.6f}")

    split_names = {0: "train", 1: "val", 2: "test"}
    split_labels = [split_names[int(l)] for l in best_labels]

    # ------------------------------------------------------------------
    # Step 6: Save unified split GeoPackage
    # ------------------------------------------------------------------
    console.rule("[bold]Step 6: Saving unified split GeoPackage[/bold]")
    gdf = gpd.GeoDataFrame(
        {
            "h3_index": h3_cells,
            "split": split_labels,
            "n_traits": n_traits_with_obs,
            **count_data,
        },
        geometry=polys_4326_kept,
        crs="EPSG:4326",
    ).to_crs("EPSG:6933")

    gdf.to_file(OUT_FILE, driver="GPKG")
    console.print(f"Saved: [cyan]{OUT_FILE}[/cyan]")

    console.rule("[bold]Summary[/bold]")
    counts = gdf["split"].value_counts()
    for split in ["train", "val", "test"]:
        n = counts.get(split, 0)
        console.print(f"  {split:5s}: {n:>4d} cells  ({n / n_cells:.1%})")


if __name__ == "__main__":
    main()
