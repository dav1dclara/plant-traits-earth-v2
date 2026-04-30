"""
Create a train/val/test split for all traits.
"""

from pathlib import Path

import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track

from ptev2.data.splitting import (
    build_histograms,
    build_split_gdf,
    extract_cell_values,
    generate_h3_grids,
    optimize_splits,
)

console = Console()


def _derive_source_tag(active_sources: list[str], legacy_source: str | None) -> str:
    norm = [str(s).lower() for s in active_sources]
    norm_set = set(norm)
    if norm_set == {"splot", "gbif"}:
        return "comb"
    if len(norm) == 1:
        return norm[0]
    if legacy_source:
        legacy = str(legacy_source).lower()
        if legacy == "comb":
            return "comb"
    return "multi_" + "-".join(sorted(norm_set))


@hydra.main(
    version_base=None,
    config_path="../../../config/22km/preprocessing",
    config_name="splitting",
)
def main(cfg: DictConfig) -> None:  # Config
    # Target settings
    data_dir = Path(cfg.data_dir)
    legacy_source = cfg.targets.get("source")
    configured_sources = cfg.targets.get("sources")
    if configured_sources:
        active_sources = [str(s) for s in configured_sources]
    elif legacy_source:
        source_name = str(legacy_source).lower()
        if source_name == "comb":
            active_sources = ["splot", "gbif"]
        else:
            active_sources = [source_name]
    else:
        active_sources = []
    if not active_sources:
        raise ValueError("cfg.targets.sources must contain at least one source name.")
    resolution_km = cfg.targets.resolution_km
    targets_root_dir = data_dir / f"{resolution_km}km" / "targets"

    # H3 settings
    h3_resolution = cfg.h3.resolution

    # Split settings
    train_frac = cfg.splitting.train_frac
    val_frac = cfg.splitting.val_frac
    test_frac = cfg.splitting.test_frac
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got train={train_frac}, val={val_frac}, test={test_frac}."
        )

    # JSD optimisation settings
    bands_for_jsd = cfg.jsd.bands
    n_restarts = cfg.jsd.n_restarts
    n_bins = cfg.jsd.n_bins
    random_seed = cfg.jsd.random_seed
    jsd_sources_cfg = cfg.jsd.get("sources")
    if jsd_sources_cfg:
        jsd_sources = [str(s).lower() for s in jsd_sources_cfg]
    else:
        jsd_sources = [str(s).lower() for s in active_sources]
    active_source_set = {str(s).lower() for s in active_sources}
    jsd_sources = [s for s in jsd_sources if s in active_source_set]
    if not jsd_sources:
        raise ValueError(
            "cfg.jsd.sources must overlap with cfg.targets.sources, got "
            f"jsd.sources={jsd_sources_cfg}, targets.sources={active_sources}"
        )

    rng = np.random.default_rng(random_seed)

    # Output settings
    out_dir = data_dir / f"{resolution_km}km" / "splits"
    source_tag = _derive_source_tag(active_sources, legacy_source)
    out_file = (
        out_dir / f"h3_splits_res{h3_resolution}_{source_tag}_{resolution_km}km.gpkg"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary of configuration
    console.rule("[bold]CREATING SPLITS[/bold]")
    console.print(f"[bold]Targets settings:[/bold]")
    console.print(f"Targets root dir:   [cyan]{targets_root_dir}[/cyan]")
    console.print(f"Resolution:      [cyan]{resolution_km}km[/cyan]")
    console.print(f"Active sources:  [cyan]{active_sources}[/cyan]")

    console.print(f"[bold]H3 settings:[/bold]")
    console.print(f"H3 resolution:   [cyan]{h3_resolution}[/cyan]")

    console.print(f"[bold]Split settings:[/bold]")
    console.print(f"Train fraction:  [cyan]{train_frac:.2%}[/cyan]")
    console.print(f"Val fraction:    [cyan]{val_frac:.2%}[/cyan]")
    console.print(f"Test fraction:   [cyan]{test_frac:.2%}[/cyan]")

    console.print(f"[bold]JSD optimisation settings:[/bold]")
    console.print(f"Bands for JSD:   [cyan]{bands_for_jsd}[/cyan]")
    console.print(f"JSD sources:     [cyan]{jsd_sources}[/cyan]")
    console.print(f"Random restarts: [cyan]{n_restarts}[/cyan]")
    console.print(f"Histogram bins:  [cyan]{n_bins}[/cyan]")
    console.print(f"Random seed:     [cyan]{random_seed}[/cyan]")

    console.print(f"[bold]Output settings:[/bold]")
    console.print(f"Output dir:      [cyan]{out_dir}[/cyan]")
    console.print(f"Output file:     [cyan]{out_file}[/cyan]")

    # Determine raster CRS from first file
    console.print("[bold]\nDetermining raster CRS[/bold]")

    source_rasters: dict[str, list[Path]] = {}
    for src_name in active_sources:
        src_dir = targets_root_dir / src_name
        rasters = sorted(src_dir.glob("*.tif"))
        if not rasters:
            raise ValueError(f"No target rasters found in {src_dir}")
        source_rasters[src_name] = rasters
        console.print(f"  - {src_name}: {len(rasters)} rasters")

    with rasterio.open(source_rasters[active_sources[0]][0]) as src:
        raster_crs = src.crs.to_string()
        raster_band_count = int(src.count)
    console.print(f"Raster CRS: {raster_crs}")
    console.print(f"Raster band count: {raster_band_count}")

    valid_bands_for_jsd = [
        int(b) for b in bands_for_jsd if 1 <= int(b) <= raster_band_count
    ]
    dropped_bands = [int(b) for b in bands_for_jsd if int(b) not in valid_bands_for_jsd]
    if dropped_bands:
        console.print(
            f"[yellow]Dropping invalid JSD bands {dropped_bands} because raster has only {raster_band_count} band(s).[/yellow]"
        )
    if not valid_bands_for_jsd:
        raise ValueError(
            f"No valid JSD bands remain after filtering cfg.jsd.bands={bands_for_jsd} against raster count={raster_band_count}."
        )
    console.print(f"Valid JSD bands:  [cyan]{valid_bands_for_jsd}[/cyan]")

    source_trait_to_raster: dict[str, dict[str, Path]] = {
        src_name: {r.stem: r for r in rasters}
        for src_name, rasters in source_rasters.items()
    }
    trait_sets = [set(mapping.keys()) for mapping in source_trait_to_raster.values()]
    common_traits = sorted(set.intersection(*trait_sets))
    if not common_traits:
        raise ValueError("No common trait rasters across configured sources.")
    for src_name, mapping in source_trait_to_raster.items():
        missing = [t for t in common_traits if t not in mapping]
        if missing:
            raise ValueError(f"Source '{src_name}' is missing common traits: {missing}")
    trait_names = common_traits

    console.print("[bold]\nGenerating H3 cells[/bold]")
    all_cells, polys_4326, polys_raster_crs = generate_h3_grids(
        h3_resolution, raster_crs
    )
    console.print(f"Total H3 cells at resolution {h3_resolution}: {len(all_cells):,}")

    # Extract pixel values per cell per trait
    console.print("[bold]\nExtracting pixel values per cell per trait[/bold]")

    n_bands = len(valid_bands_for_jsd)
    n_cells_all = len(all_cells)
    n_features = len(trait_names) * n_bands
    all_cell_values = [[None] * n_features for _ in range(n_cells_all)]
    # JSD features can be computed from a subset of sources (e.g. splot only),
    # while split coverage can still consider all active sources.
    jsd_cell_values = [[None] * n_features for _ in range(n_cells_all)]
    for c in range(n_cells_all):
        for f in range(n_features):
            all_cell_values[c][f] = np.array([], dtype=float)
            jsd_cell_values[c][f] = np.array([], dtype=float)

    extraction_jobs = [
        (src_name, trait_name, source_trait_to_raster[src_name][trait_name])
        for src_name in active_sources
        for trait_name in trait_names
    ]
    for src_name, trait_name, raster_path in track(
        extraction_jobs, description="Extracting pixel values..."
    ):
        t = trait_names.index(trait_name)
        cell_vals = extract_cell_values(
            raster_path, polys_raster_crs, valid_bands_for_jsd
        )
        for c, band_vals in enumerate(cell_vals):
            for b, vals in enumerate(band_vals):
                f_idx = t * n_bands + b
                if vals.size == 0:
                    continue
                all_cell_values[c][f_idx] = np.concatenate(
                    [all_cell_values[c][f_idx], vals]
                )
                if str(src_name).lower() in jsd_sources:
                    jsd_cell_values[c][f_idx] = np.concatenate(
                        [jsd_cell_values[c][f_idx], vals]
                    )

    keep = [
        any(len(all_cell_values[c][f]) > 0 for f in range(n_features))
        for c in range(n_cells_all)
    ]

    h3_cells = [cell for cell, k in zip(all_cells, keep) if k]
    polys_4326_kept = [p for p, k in zip(polys_4326, keep) if k]
    all_cell_values = [v for v, k in zip(all_cell_values, keep) if k]
    jsd_cell_values = [v for v, k in zip(jsd_cell_values, keep) if k]
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

    # Compute histograms for each cell and trait
    console.print("[bold]\nComputing histograms[/bold]")
    source_band_pos = valid_bands_for_jsd.index(7) if 7 in valid_bands_for_jsd else None
    source_feature_indices = (
        {t * n_bands + source_band_pos for t in range(len(trait_names))}
        if source_band_pos is not None
        else set()
    )
    histograms, bin_edges = build_histograms(
        jsd_cell_values, n_bins=n_bins, categorical_features=source_feature_indices
    )
    valid_features = sum(1 for e in bin_edges if e is not None)
    console.print(
        f"Features with valid data range: {valid_features} / {n_features} ({len(trait_names)} traits × {n_bands} bands)"
    )

    console.print("[bold]\nOptimising split assignment[/bold]")
    n_train = round(train_frac * n_cells)
    n_val = round(val_frac * n_cells)
    n_test = n_cells - n_train - n_val
    console.print(f"Target: train={n_train}  val={n_val}  test={n_test}")
    console.print(f"Running {n_restarts} random restarts...")

    best_labels, best_score = optimize_splits(
        histograms,
        bin_edges,
        n_train,
        n_val,
        n_test,
        n_restarts,
        rng,
    )
    console.print(f"Final best mean JSD: {best_score:.6f}")

    split_names = {0: "train", 1: "val", 2: "test"}
    split_labels = [split_names[int(l)] for l in best_labels]

    console.print("[bold]\nSaving split GeoPackage[/bold]")
    gdf = build_split_gdf(
        h3_cells, polys_4326_kept, split_labels, n_traits_with_obs, count_data
    )

    gdf.to_file(out_file, driver="GPKG")
    console.print(f"Saved: [cyan]{out_file}[/cyan]")

    console.rule("[bold]Summary[/bold]")
    counts = gdf["split"].value_counts()
    for split in ["train", "val", "test"]:
        n = counts.get(split, 0)
        console.print(f"  {split:5s}: {n:>4d} cells  ({n / n_cells:.1%})")


if __name__ == "__main__":
    main()
