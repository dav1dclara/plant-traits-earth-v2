from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import rasterio
import rasterio.features
from antimeridian import fix_polygon
from pyproj import Transformer
from scipy.spatial.distance import jensenshannon
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform


def h3_to_polygon_4326(cell: str) -> Polygon:
    """Convert an H3 cell index to a Shapely polygon in EPSG:4326, antimeridian-safe."""
    coords = h3.cell_to_boundary(cell)
    return fix_polygon(Polygon([(lon, lat) for lat, lon in coords]))


def reproject_polygon(poly: Polygon, src_crs: str, dst_crs: str) -> Polygon:
    """Reproject a Shapely polygon from src_crs to dst_crs."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, poly)


def generate_h3_grids(
    h3_resolution: int, raster_crs: str
) -> tuple[list[str], list[Polygon], list[Polygon]]:
    """Generate all global H3 cells at the given resolution with polygons in EPSG:4326 and raster_crs."""
    all_cells = sorted(h3.uncompact_cells(h3.get_res0_cells(), h3_resolution))
    polys_4326 = [h3_to_polygon_4326(c) for c in all_cells]
    polys_raster_crs = [
        reproject_polygon(p, "EPSG:4326", raster_crs) for p in polys_4326
    ]
    return all_cells, polys_4326, polys_raster_crs


def build_cell_labels(
    cell_polygons_raster_crs: list[Polygon],
    out_shape: tuple[int, int],
    transform,
) -> np.ndarray:
    """Rasterize H3 cell polygons once. Returns int32 array where each pixel holds its cell index+1 (0=no cell)."""
    shapes = [
        (geom.__geo_interface__, idx + 1)
        for idx, geom in enumerate(cell_polygons_raster_crs)
        if geom is not None
    ]
    return rasterio.features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.int32,
    )


def build_cell_index(
    cell_labels: np.ndarray, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute a sort-order and boundary array for O(N log N) groupby extraction.

    Returns (order, boundaries) where:
    - order: argsort of flattened cell_labels (stable, int32)
    - boundaries: length n_cells+2 array; pixels for 1-indexed cell k are at
      sorted_band[boundaries[k] : boundaries[k+1]]

    Pass both to extract_cell_values to replace the O(N × n_cells) mask loop.
    """
    flat = cell_labels.ravel()
    order = np.argsort(flat, kind="stable").astype(np.int32)
    sorted_labels = flat[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(n_cells + 2)).astype(np.int64)
    return order, boundaries


def extract_cell_values(
    raster_path: Path,
    cell_polygons_raster_crs: list[Polygon],
    bands_for_jsd: list[int],
    cell_labels: np.ndarray | None = None,
    cell_index: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[list[np.ndarray]]:
    """Extract valid pixel values per cell per band. Returns cell_values[cell_idx][band_idx].

    For large grids (e.g. 1km), pass cell_index=(order, boundaries) from build_cell_index
    to use a single O(N log N) sort-based groupby instead of O(N × n_cells) mask comparisons.
    Falls back to cell_labels or re-rasterizes if neither is provided.
    """
    n_cells = len(cell_polygons_raster_crs)

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        if cell_index is None and cell_labels is None:
            cell_labels = build_cell_labels(
                cell_polygons_raster_crs, src.shape, src.transform
            )

        band_arrays = []
        for band in bands_for_jsd:
            data = src.read(band).astype(float)
            if nodata is not None:
                data[data == nodata] = np.nan
            band_arrays.append(data)

    if cell_index is not None:
        order, boundaries = cell_index
        cell_values = [[] for _ in range(n_cells)]
        for band_arr in band_arrays:
            sorted_band = band_arr.ravel()[order]
            for idx in range(n_cells):
                v = sorted_band[boundaries[idx + 1] : boundaries[idx + 2]]
                cell_values[idx].append(v[np.isfinite(v)])
        return cell_values

    # Fallback: per-cell mask loop (slow on large grids)
    cell_values = []
    for idx in range(n_cells):
        mask = cell_labels == (idx + 1)
        cell_values.append(
            [(lambda v: v[np.isfinite(v)])(band[mask]) for band in band_arrays]
        )
    return cell_values


def build_histograms(
    all_cell_values: list[list[np.ndarray]],  # [n_cells][n_traits]
    n_bins: int = 10,
    categorical_features: set[int] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Pre-compute per-cell histograms over shared bin edges. Returns histograms (n_cells, n_features, n_bins) and bin_edges.

    For features in categorical_features, fixed edges [0.5, 1.5, 2.5] are used (values 1 and 2),
    with counts stored in the first two bins and the rest left as zero.
    """
    n_cells = len(all_cell_values)
    n_traits = len(all_cell_values[0])
    histograms = np.zeros((n_cells, n_traits, n_bins), dtype=float)
    bin_edges = []

    for t in range(n_traits):
        if categorical_features and t in categorical_features:
            edges = np.array([0.5, 1.5, 2.5])
            bin_edges.append(edges)
            for c in range(n_cells):
                vals = all_cell_values[c][t]
                if len(vals) > 0:
                    h, _ = np.histogram(vals, bins=edges)
                    histograms[c, t, :2] = h.astype(float)
            continue

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
    """Mean pairwise JSD across all features and split pairs (train/val, train/test, val/test)."""
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


def optimize_splits(
    histograms: np.ndarray,
    bin_edges: list,
    n_train: int,
    n_val: int,
    n_test: int,
    n_restarts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Find the cell assignment minimising mean JSD via random restarts. Returns (best_labels, best_score)."""
    base_labels = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    best_labels = rng.permutation(base_labels)
    best_score = compute_total_jsd(histograms, best_labels, bin_edges)

    for _ in range(1, n_restarts):
        labels = rng.permutation(base_labels)
        score = compute_total_jsd(histograms, labels, bin_edges)
        if score < best_score:
            best_score = score
            best_labels = labels.copy()

    return best_labels, best_score


def build_split_gdf(
    h3_cells: list[str],
    polys_4326: list[Polygon],
    split_labels: list[str],
    n_traits_with_obs: list[int],
    count_data: dict,
    crs: str = "EPSG:6933",
) -> gpd.GeoDataFrame:
    """Build the output GeoDataFrame with split and count columns, reprojected to crs."""
    return gpd.GeoDataFrame(
        {
            "h3_index": h3_cells,
            "split": split_labels,
            "n_traits": n_traits_with_obs,
            **count_data,
        },
        geometry=polys_4326,
        crs="EPSG:4326",
    ).to_crs(crs)
