"""Create spatial H3 train/val/test splits anchored on sPlot coverage."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track

from ptev2.data.splitting import (
    build_histograms,
    build_split_gdf,
    compute_total_jsd,
    extract_cell_values,
    generate_h3_grids,
    optimize_splits,
)

console = Console()
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def _resolve_data_root(candidates: list[str], resolution_km: int) -> Path:
    for root in candidates:
        path = Path(root) / f"{resolution_km}km"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve data root from candidates: {candidates} (for {resolution_km}km)."
    )


def _resolve_band_indices(path: Path, band_names: list[str]) -> list[int]:
    with rasterio.open(path) as src:
        descriptions = [
            str(d).strip().lower() if d is not None else "" for d in src.descriptions
        ]
    name_to_idx = {name: i + 1 for i, name in enumerate(descriptions) if name}
    missing = [name for name in band_names if name.lower() not in name_to_idx]
    if missing:
        raise ValueError(
            f"Missing requested bands {missing} in {path.name}. "
            f"Available descriptions: {descriptions}"
        )
    return [int(name_to_idx[name.lower()]) for name in band_names]


def _trait_tif_path(root: Path, trait: str) -> Path:
    direct = root / f"{trait}.tif"
    if direct.exists():
        return direct
    prefixed = root / f"X{trait}.tif"
    if prefixed.exists():
        return prefixed
    return direct


def _split_counts(n: int, fracs: tuple[float, float, float]) -> tuple[int, int, int]:
    """Round split counts while preserving total n."""
    raw = np.asarray(fracs, dtype=float) * float(n)
    base = np.floor(raw).astype(int)
    remainder = int(n - base.sum())
    order = np.argsort(-(raw - base))
    for i in range(remainder):
        base[int(order[i % 3])] += 1
    return int(base[0]), int(base[1]), int(base[2])


def _hemisphere_stats(
    labels: np.ndarray, latitudes: np.ndarray
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for code, name in SPLIT_NAMES.items():
        in_split = labels == code
        total = int(in_split.sum())
        south = int(((latitudes < 0) & in_split).sum())
        north = int(((latitudes >= 0) & in_split).sum())
        stats[name] = {
            "total_cells": total,
            "north_cells": north,
            "south_cells": south,
            "south_fraction": float(south / total) if total else 0.0,
        }
    return stats


def _passes_south_constraints(
    stats: dict[str, dict[str, float]],
    min_south_cells: int,
    min_south_fraction: float,
) -> bool:
    for split in ["train", "val", "test"]:
        if stats[split]["south_cells"] < min_south_cells:
            return False
        if stats[split]["south_fraction"] < min_south_fraction:
            return False
    return True


def _sample_hemisphere_stratified_labels(
    latitudes: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.empty(latitudes.shape[0], dtype=np.int8)
    for idx in (np.where(latitudes >= 0.0)[0], np.where(latitudes < 0.0)[0]):
        n = int(idx.size)
        n_train, n_val, n_test = _split_counts(n, (train_frac, val_frac, test_frac))
        hemi_labels = np.array(
            [0] * n_train + [1] * n_val + [2] * n_test, dtype=np.int8
        )
        labels[idx] = rng.permutation(hemi_labels)
    return labels


def _optimize_with_optional_hemisphere_stratification(
    *,
    histograms: np.ndarray,
    bin_edges: list,
    n_train: int,
    n_val: int,
    n_test: int,
    latitudes: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    n_restarts: int,
    rng: np.random.Generator,
    stratify_by_hemisphere: bool,
    min_south_cells_per_split: int,
    min_south_fraction_per_split: float,
) -> tuple[np.ndarray, float, dict[str, dict[str, float]], bool]:
    if not stratify_by_hemisphere:
        labels, best_jsd = optimize_splits(
            histograms,
            bin_edges,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_restarts=n_restarts,
            rng=rng,
        )
        return labels, float(best_jsd), _hemisphere_stats(labels, latitudes), True

    best_labels: np.ndarray | None = None
    best_jsd = float("inf")
    best_stats: dict[str, dict[str, float]] | None = None

    fallback_labels: np.ndarray | None = None
    fallback_jsd = float("inf")
    fallback_stats: dict[str, dict[str, float]] | None = None

    for _ in range(n_restarts):
        labels = _sample_hemisphere_stratified_labels(
            latitudes=latitudes,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            rng=rng,
        )
        jsd = float(compute_total_jsd(histograms, labels, bin_edges))
        stats = _hemisphere_stats(labels, latitudes)

        if jsd < fallback_jsd:
            fallback_labels, fallback_jsd, fallback_stats = labels.copy(), jsd, stats

        if (
            _passes_south_constraints(
                stats=stats,
                min_south_cells=min_south_cells_per_split,
                min_south_fraction=min_south_fraction_per_split,
            )
            and jsd < best_jsd
        ):
            best_labels, best_jsd, best_stats = labels.copy(), jsd, stats

    if best_labels is not None and best_stats is not None:
        return best_labels, best_jsd, best_stats, True
    assert fallback_labels is not None and fallback_stats is not None
    return fallback_labels, fallback_jsd, fallback_stats, False


def _compute_split_raster(gdf: pd.DataFrame, ref_raster_path: Path) -> np.ndarray:
    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        out_shape = src.shape
        ref_crs = src.crs

    gdf_ref = gdf.to_crs(ref_crs)
    shapes = []
    for split_name, split_code in [("train", 0), ("val", 1), ("test", 2)]:
        for geom in gdf_ref.loc[gdf_ref["split"] == split_name, "geometry"]:
            if geom is not None:
                shapes.append((geom, split_code))
    return rasterio.features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=-1,
        dtype=np.int16,
    )


def _validate_trait_coverage_or_raise(
    gdf: pd.DataFrame,
    raster_paths: list[Path],
    out_path: Path,
) -> None:
    split_raster = _compute_split_raster(gdf=gdf, ref_raster_path=raster_paths[0])
    rows = []
    union_valid = np.zeros_like(split_raster, dtype=bool)

    for path in raster_paths:
        with rasterio.open(path) as src:
            mean = src.read(1).astype(np.float32)
            if src.nodata is not None:
                mean[mean == src.nodata] = np.nan
        valid = np.isfinite(mean)
        union_valid |= valid
        missing = valid & (split_raster < 0)
        rows.append(
            {
                "trait": path.stem,
                "valid_pixel_count": int(valid.sum()),
                "uncovered_pixel_count": int(missing.sum()),
            }
        )

    coverage_df = pd.DataFrame(rows).sort_values("trait")
    coverage_csv = out_path.with_name(f"{out_path.stem}_trait_coverage.csv")
    coverage_df.to_csv(coverage_csv, index=False)

    union_missing = int((union_valid & (split_raster < 0)).sum())
    trait_missing = int(coverage_df["uncovered_pixel_count"].sum())
    if union_missing > 0 or trait_missing > 0:
        top = coverage_df[coverage_df["uncovered_pixel_count"] > 0].head(10)
        raise RuntimeError(
            "Trait coverage validation failed. "
            f"union_missing_pixels={union_missing}, trait_missing_sum={trait_missing}. "
            f"Top uncovered traits:\n{top.to_string(index=False)}\n"
            f"Coverage report: {coverage_csv}"
        )


@hydra.main(
    version_base=None,
    config_path="../../config/preprocessing",
    config_name="splot_anchor_split",
)
def main(cfg: DictConfig) -> None:
    resolution_km = int(cfg.settings.resolution_km)
    if cfg.paths.get("data_root"):
        data_root = Path(str(cfg.paths.data_root)) / f"{resolution_km}km"
        if not data_root.exists():
            raise FileNotFoundError(f"Configured data_root does not exist: {data_root}")
    else:
        data_root = _resolve_data_root(
            [str(v) for v in cfg.paths.data_root_candidates],
            resolution_km=resolution_km,
        )
    targets_dir = data_root / str(cfg.paths.targets_dirname)
    splits_dir = data_root / str(cfg.paths.splits_dirname)
    splits_dir.mkdir(parents=True, exist_ok=True)

    trait_filter = [str(v) for v in cfg.data.traits] if cfg.data.traits else []
    raster_paths = (
        [_trait_tif_path(targets_dir, trait) for trait in trait_filter]
        if trait_filter
        else sorted(targets_dir.glob("X*.tif"))
    )
    if not raster_paths:
        raise FileNotFoundError(f"No sPlot rasters found in {targets_dir}")

    h3_resolution = int(cfg.settings.h3_resolution)
    train_frac = float(cfg.settings.train_frac)
    val_frac = float(cfg.settings.val_frac)
    test_frac = float(cfg.settings.test_frac)
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    n_restarts = int(cfg.settings.n_restarts)
    n_bins = int(cfg.settings.n_bins)
    jsd_band_names = [str(v).strip().lower() for v in cfg.settings.jsd_band_names]
    rng = np.random.default_rng(int(cfg.settings.random_seed))
    stratify_by_hemisphere = bool(cfg.settings.get("stratify_by_hemisphere", False))
    min_south_cells = int(cfg.settings.get("min_south_cells_per_split", 0))
    min_south_fraction = float(cfg.settings.get("min_south_fraction_per_split", 0.0))
    enforce_coverage = bool(cfg.settings.get("enforce_full_trait_coverage", True))

    out_path = splits_dir / f"h3_splits_res{h3_resolution}_splot_{resolution_km}km.gpkg"

    console.rule("[bold]CREATE SPLOT-ANCHOR SPLITS[/bold]")
    console.print(f"Data root: [cyan]{data_root}[/cyan]")
    console.print(f"Targets: [cyan]{targets_dir}[/cyan]")
    console.print(f"Traits: [cyan]{len(raster_paths)}[/cyan]")
    console.print(
        f"Fractions train/val/test: [cyan]{train_frac:.2f}/{val_frac:.2f}/{test_frac:.2f}[/cyan]"
    )
    console.print(f"H3 resolution: [cyan]{h3_resolution}[/cyan]")
    console.print(f"Hemisphere stratification: [cyan]{stratify_by_hemisphere}[/cyan]")

    with rasterio.open(raster_paths[0]) as src:
        raster_crs = src.crs.to_string()
    all_cells, polys_4326, polys_raster_crs = generate_h3_grids(
        h3_resolution, raster_crs
    )

    n_bands = len(jsd_band_names)
    n_cells_all = len(all_cells)
    n_features = len(raster_paths) * n_bands
    all_cell_values: list[list[np.ndarray | None]] = [
        [None] * n_features for _ in range(n_cells_all)
    ]

    for t, path in enumerate(track(raster_paths, description="Extract sPlot values")):
        jsd_bands = _resolve_band_indices(path, jsd_band_names)
        cell_vals = extract_cell_values(path, polys_raster_crs, jsd_bands)
        for c, band_vals in enumerate(cell_vals):
            for b, vals in enumerate(band_vals):
                all_cell_values[c][t * n_bands + b] = vals

    keep = [
        any(len(all_cell_values[c][f]) > 0 for f in range(n_features))
        for c in range(n_cells_all)
    ]
    h3_cells = [cell for cell, keep_cell in zip(all_cells, keep) if keep_cell]
    polys_4326_kept = [poly for poly, keep_cell in zip(polys_4326, keep) if keep_cell]
    all_cell_values = [
        values for values, keep_cell in zip(all_cell_values, keep) if keep_cell
    ]
    if not h3_cells:
        raise RuntimeError("No H3 cells with sPlot data were found.")

    trait_names = [path.stem for path in raster_paths]
    count_data = {
        name: [len(all_cell_values[c][t * n_bands]) for c in range(len(h3_cells))]
        for t, name in enumerate(trait_names)
    }
    n_traits_with_obs = [
        sum(
            1
            for t in range(len(trait_names))
            if len(all_cell_values[c][t * n_bands]) > 0
        )
        for c in range(len(h3_cells))
    ]

    histograms, bin_edges = build_histograms(
        all_cell_values,
        n_bins=n_bins,
        categorical_features=None,
    )

    n_cells = len(h3_cells)
    n_train = round(train_frac * n_cells)
    n_val = round(val_frac * n_cells)
    n_test = n_cells - n_train - n_val
    latitudes = np.asarray(
        [float(poly.centroid.y) for poly in polys_4326_kept], dtype=float
    )

    labels, best_jsd, hemi_stats, constrained_ok = (
        _optimize_with_optional_hemisphere_stratification(
            histograms=histograms,
            bin_edges=bin_edges,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            latitudes=latitudes,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            n_restarts=n_restarts,
            rng=rng,
            stratify_by_hemisphere=stratify_by_hemisphere,
            min_south_cells_per_split=min_south_cells,
            min_south_fraction_per_split=min_south_fraction,
        )
    )
    if stratify_by_hemisphere and not constrained_ok:
        console.print(
            "[yellow]No candidate met south constraints; using best unconstrained stratified split.[/yellow]"
        )

    split_labels = [SPLIT_NAMES[int(v)] for v in labels]
    gdf = build_split_gdf(
        h3_cells=h3_cells,
        polys_4326=polys_4326_kept,
        split_labels=split_labels,
        n_traits_with_obs=n_traits_with_obs,
        count_data=count_data,
    )

    if enforce_coverage:
        _validate_trait_coverage_or_raise(
            gdf=gdf, raster_paths=raster_paths, out_path=out_path
        )

    gdf.to_file(out_path, driver="GPKG")

    console.rule("[bold]DONE[/bold]")
    console.print(f"Best mean JSD: [cyan]{best_jsd:.6f}[/cyan]")
    console.print(f"Output split file: [cyan]{out_path}[/cyan]")
    split_counts = gdf["split"].value_counts()
    for split in ["train", "val", "test"]:
        n_split = int(split_counts.get(split, 0))
        south_cells = int(hemi_stats[split]["south_cells"])
        south_frac = float(hemi_stats[split]["south_fraction"])
        console.print(
            f"  - {split}: [cyan]{n_split}[/cyan] cells | south={south_cells} ({south_frac:.2%})"
        )

    hemi_df = pd.DataFrame(
        [{"split": split, **hemi_stats[split]} for split in ["train", "val", "test"]]
    )
    hemi_csv = out_path.with_name(f"{out_path.stem}_hemisphere_summary.csv")
    hemi_df.to_csv(hemi_csv, index=False)
    console.print(f"Hemisphere summary: [cyan]{hemi_csv}[/cyan]")


if __name__ == "__main__":
    main()
