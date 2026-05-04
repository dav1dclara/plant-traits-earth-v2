from collections import defaultdict
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
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    polys_raster_crs = [shapely_transform(transformer.transform, p) for p in polys_4326]
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


def spatially_stratified_split(
    histograms: np.ndarray,
    bin_edges: list,
    cell_ids: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    n_restarts: int,
    rng: np.random.Generator,
    stratum_resolution: int = 1,
) -> tuple[np.ndarray, float]:
    """Assign sPlot cells to train/val/test via geographically stratified random restarts.

    Each candidate is generated by:
      1. Grouping cells by their resolution-1 H3 parent (geographic strata).
      2. Shuffling cells within each stratum and shuffling stratum order.
      3. Round-robin interleaving strata so all splits draw from every region.
      4. Cutting the resulting sequence into train / val / test blocks.
    The candidate with the lowest 3-way JSD is returned.
    """
    assert n_train + n_val + n_test == len(cell_ids), "counts must sum to total cells"

    strata = defaultdict(list)
    for i, cell in enumerate(cell_ids):
        strata[h3.cell_to_parent(cell, stratum_resolution)].append(i)
    stratum_arrays = [np.array(v) for v in strata.values()]

    best_labels: np.ndarray | None = None
    best_score = float("inf")

    for _ in range(n_restarts):
        labels = np.empty(len(cell_ids), dtype=int)
        remaining = np.array([n_train, n_val, n_test], dtype=int)
        cells_left = len(cell_ids)

        for k, arr in enumerate(stratum_arrays):
            n = len(arr)
            perm = arr[rng.permutation(n)]

            if k == len(stratum_arrays) - 1:
                counts = remaining.copy()
            else:
                counts = np.round(remaining / cells_left * n).astype(int)
                counts = np.maximum(counts, 0)
                counts[-1] = n - counts[:-1].sum()
                counts[-1] = max(counts[-1], 0)
                diff = n - counts.sum()
                counts[np.argmax(counts)] += (
                    diff  # absorb any residual into largest bucket
                )
                if n >= 3 and counts[2] == 0 and remaining[2] > 0:
                    counts[2] = 1
                    counts[0] -= 1

            remaining -= counts
            cells_left -= n

            cuts = np.concatenate([[0], np.cumsum(counts)])
            for label, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
                labels[perm[lo:hi]] = label

        score = compute_total_jsd(histograms, labels, bin_edges)
        if score < best_score:
            best_score = score
            best_labels = labels.copy()

    return best_labels, best_score  # type: ignore[return-value]


def fix_clustering(
    labels: np.ndarray,
    cell_ids: list[str],
    histograms: np.ndarray,
    bin_edges: list,
    max_same_neighbors: int,
    jsd_tolerance: float,
    rng: np.random.Generator,
    n_iters: int = 10_000,
    splits: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    """Post-process labels to break up split clusters via greedy swaps.

    A cell is a violation if it belongs to a split in `splits` and has more than
    max_same_neighbors neighbors in the same split. Each iteration picks a random
    violating cell and tries swapping it with a cell from a different split. The swap
    is accepted only if it reduces local violations and keeps JSD within jsd_tolerance
    of the starting JSD. Defaults to checking only test cells (label=2).
    """
    if splits is None:
        splits = [2]
    splits_set = set(splits)

    cell_to_idx = {cell: i for i, cell in enumerate(cell_ids)}
    jsd_budget = compute_total_jsd(histograms, labels, bin_edges) * (1 + jsd_tolerance)

    def n_same_neighbors(idx: int) -> int:
        cell = cell_ids[idx]
        label = labels[idx]
        return sum(
            1
            for nb in h3.grid_disk(cell, 1)
            if nb != cell and nb in cell_to_idx and labels[cell_to_idx[nb]] == label
        )

    violations = {
        i
        for i in range(len(cell_ids))
        if labels[i] in splits_set and n_same_neighbors(i) > max_same_neighbors
    }

    n_accepted = 0
    for _ in range(n_iters):
        if not violations:
            break

        i = int(rng.choice(sorted(violations)))
        diff_idx = np.where(labels != labels[i])[0]
        j = int(rng.choice(diff_idx))

        # Cells whose violation status may change after swapping i and j
        affected: set[int] = {i, j}
        for cell_x in (cell_ids[i], cell_ids[j]):
            for nb in h3.grid_disk(cell_x, 1):
                if nb != cell_x and nb in cell_to_idx:
                    affected.add(cell_to_idx[nb])

        old_local = sum(1 for k in affected if n_same_neighbors(k) > max_same_neighbors)

        labels[i], labels[j] = labels[j], labels[i]
        new_local = sum(1 for k in affected if n_same_neighbors(k) > max_same_neighbors)

        if (
            new_local < old_local
            and compute_total_jsd(histograms, labels, bin_edges) <= jsd_budget
        ):
            for k in affected:
                if labels[k] in splits_set and n_same_neighbors(k) > max_same_neighbors:
                    violations.add(k)
                else:
                    violations.discard(k)
            n_accepted += 1
        else:
            labels[i], labels[j] = labels[j], labels[i]  # revert

    return labels, n_accepted


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


def build_histograms_from_rasters(
    raster_paths: list[Path],
    bands: list[int],
    cell_index: tuple[np.ndarray, np.ndarray],
    n_cells: int,
    n_bins: int,
    categorical_features: set[int] | None = None,
    progress=None,
) -> tuple[np.ndarray, list, np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass histogram computation directly from rasters.

    Replaces the store-everything approach of looping extract_cell_values then
    build_histograms. Uses vectorized np.digitize + np.bincount — no per-cell
    Python loops, no large intermediate pixel arrays.

    Returns:
        histograms:   float64 array (n_cells, n_features, n_bins)
        bin_edges:    list[np.ndarray | None], length n_features
        has_data:     bool array (n_cells,) — True if cell has any valid pixel
        obs_counts:   int64 array (n_cells,) — valid-pixel count for feature 0
        n_traits_obs: int32 array (n_cells,) — number of rasters with ≥1 valid pixel
    """
    order, boundaries = cell_index
    n_bands_per_raster = len(bands)
    n_features = len(raster_paths) * n_bands_per_raster

    # Pre-compute cell assignment for every in-cell pixel (computed once, reused per band).
    # boundaries[k+1]:boundaries[k+2] are the sorted-array positions for 0-indexed cell k.
    pixel_counts = np.diff(boundaries)[1:]  # (n_cells,) pixels per cell
    cell_id = np.repeat(np.arange(n_cells, dtype=np.int32), pixel_counts)
    cell_start = int(boundaries[1])  # first in-cell position in the sorted array

    histograms = np.zeros((n_cells, n_features, n_bins), dtype=np.float64)
    bin_edges: list = [None] * n_features
    has_data = np.zeros(n_cells, dtype=bool)
    obs_counts = np.zeros(n_cells, dtype=np.int64)
    n_traits_obs = np.zeros(n_cells, dtype=np.int32)
    raster_obs_counts = np.zeros((n_cells, len(raster_paths)), dtype=np.int64)

    for t, raster_path in enumerate(raster_paths):
        if progress is not None:
            _task = progress.add_task(f"{raster_path.stem}...")
        trait_has_data = np.zeros(n_cells, dtype=bool)
        with rasterio.open(raster_path) as src:
            nodata = src.nodata
            for b_idx, band in enumerate(bands):
                feature_idx = t * n_bands_per_raster + b_idx

                data = src.read(band).astype(np.float64)
                if nodata is not None:
                    data[data == nodata] = np.nan

                # Restrict to pixels that fall inside any H3 cell
                in_cell = data.ravel()[order][cell_start:]
                valid = np.isfinite(in_cell)
                vals = in_cell[valid]
                cids = cell_id[valid].astype(np.int64)

                if len(cids) > 0:
                    trait_has_data[cids] = True
                    has_data[cids] = True

                if b_idx == 0:
                    bc = np.bincount(cids, minlength=n_cells).astype(np.int64)
                    raster_obs_counts[:, t] = bc
                    if t == 0:
                        obs_counts = bc

                if len(vals) == 0:
                    continue

                if categorical_features and feature_idx in categorical_features:
                    edges = np.array([0.5, 1.5, 2.5])
                    bin_edges[feature_idx] = edges
                else:
                    vmin, vmax = float(vals.min()), float(vals.max())
                    if vmin == vmax:
                        continue
                    edges = np.linspace(vmin, vmax, n_bins + 1)
                    bin_edges[feature_idx] = edges

                bin_idx = np.clip(
                    np.digitize(vals, edges).astype(np.int64) - 1, 0, n_bins - 1
                )
                combined = cids * n_bins + bin_idx
                counts = np.bincount(combined, minlength=n_cells * n_bins)
                histograms[:, feature_idx, :] += counts.reshape(n_cells, n_bins)

        n_traits_obs += trait_has_data.astype(np.int32)
        if progress is not None:
            progress.update(
                _task,
                description=f"[green]✓[/green] {raster_path.stem}",
                completed=1,
                total=1,
            )

    return histograms, bin_edges, has_data, obs_counts, n_traits_obs, raster_obs_counts


def count_valid_pixels_per_raster(
    raster_paths: list[Path],
    band: int,
    cell_index: tuple[np.ndarray, np.ndarray],
    n_cells: int,
    progress=None,
) -> np.ndarray:
    """Count valid (non-NaN) pixels per cell for each raster. Returns int64 array (n_cells, n_rasters)."""
    order, boundaries = cell_index
    pixel_counts = np.diff(boundaries)[1:]
    cell_id = np.repeat(np.arange(n_cells, dtype=np.int32), pixel_counts)
    cell_start = int(boundaries[1])

    counts = np.zeros((n_cells, len(raster_paths)), dtype=np.int64)

    for t, raster_path in enumerate(raster_paths):
        if progress is not None:
            _task = progress.add_task(f"{raster_path.stem}...")
        with rasterio.open(raster_path) as src:
            nodata = src.nodata
            data = src.read(band).astype(np.float32)
            if nodata is not None:
                data[data == nodata] = np.nan
            in_cell = data.ravel()[order][cell_start:]
            cids = cell_id[np.isfinite(in_cell)].astype(np.int64)
            if len(cids) > 0:
                counts[:, t] = np.bincount(cids, minlength=n_cells)
        if progress is not None:
            progress.update(
                _task,
                description=f"[green]✓[/green] {raster_path.stem}",
                completed=1,
                total=1,
            )

    return counts
