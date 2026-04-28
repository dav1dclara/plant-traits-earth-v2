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
    build_cell_index,
    build_cell_labels,
    build_histograms,
    build_split_gdf,
    extract_cell_values,
    generate_h3_grids,
    optimize_splits,
)

console = Console()


@hydra.main(
    version_base=None,
    config_path="../../config/preprocessing",
    config_name="splitting_1km",
)
def main(cfg: DictConfig) -> None:  # Config
    # Target settings
    data_dir = Path(cfg.data_dir)
    source = cfg.targets.source
    resolution_km = cfg.targets.resolution_km
    targets_dir = data_dir / f"{resolution_km}km" / "targets" / source

    # H3 settings
    h3_resolution = cfg.h3.resolution

    # Split settings
    train_frac = cfg.splitting.train_frac
    val_frac = cfg.splitting.val_frac
    test_frac = cfg.splitting.test_frac
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        "Fractions must sum to 1"
    )

    # JSD optimisation settings
    bands_for_jsd = cfg.jsd.bands
    n_restarts = cfg.jsd.n_restarts
    n_bins = cfg.jsd.n_bins
    random_seed = cfg.jsd.random_seed

    rng = np.random.default_rng(random_seed)

    # Output settings
    out_dir = data_dir / f"{resolution_km}km" / "splits"
    out_file = out_dir / f"h3_splits_res{h3_resolution}_{source}_{resolution_km}km.gpkg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary of configuration
    console.rule("[bold]CREATING SPLITS[/bold]")
    console.print(f"[bold]Targets settings:[/bold]")
    console.print(f"Targets dir:     [cyan]{targets_dir}[/cyan]")
    console.print(f"Resolution:      [cyan]{resolution_km}km[/cyan]")
    console.print(f"Source:          [cyan]{source}[/cyan]")

    console.print(f"[bold]H3 settings:[/bold]")
    console.print(f"H3 resolution:   [cyan]{h3_resolution}[/cyan]")

    console.print(f"[bold]Split settings:[/bold]")
    console.print(f"Train fraction:  [cyan]{train_frac:.2%}[/cyan]")
    console.print(f"Val fraction:    [cyan]{val_frac:.2%}[/cyan]")
    console.print(f"Test fraction:   [cyan]{test_frac:.2%}[/cyan]")

    console.print(f"[bold]JSD optimisation settings:[/bold]")
    console.print(f"Bands for JSD:   [cyan]{bands_for_jsd}[/cyan]")
    console.print(f"Random restarts: [cyan]{n_restarts}[/cyan]")
    console.print(f"Histogram bins:  [cyan]{n_bins}[/cyan]")
    console.print(f"Random seed:     [cyan]{random_seed}[/cyan]")

    console.print(f"[bold]Output settings:[/bold]")
    console.print(f"Output dir:      [cyan]{out_dir}[/cyan]")
    console.print(f"Output file:     [cyan]{out_file}[/cyan]")

    # Determine raster CRS from first file
    console.print("[bold]\nDetermining raster CRS[/bold]")
    target_rasters = sorted(targets_dir.glob("*.tif"))
    console.print(f"Found {len(target_rasters)} target rasters")
    with rasterio.open(target_rasters[0]) as src:
        raster_crs = src.crs.to_string()
    console.print(f"Raster CRS: {raster_crs}")

    # console.print("[bold]\nGenerating H3 cells[/bold]")
    # all_cells, polys_4326, polys_raster_crs = generate_h3_grids(
    #     h3_resolution, raster_crs
    # )
    # console.print(f"Total H3 cells at resolution {h3_resolution}: {len(all_cells):,}")

    # # Extract pixel values per cell per trait
    # console.print("[bold]\nExtracting pixel values per cell per trait[/bold]")

    # n_bands = len(bands_for_jsd)
    # n_cells_all = len(all_cells)
    # n_features = len(gbif_rasters) * n_bands
    # all_cell_values = [[None] * n_features for _ in range(n_cells_all)]

    # # Pre-compute cell label raster and sort index once (amortised over all traits)
    # console.print("Pre-computing H3 cell label raster...")
    # with rasterio.open(gbif_rasters[0]) as ref:
    #     ref_shape = ref.shape
    #     ref_transform = ref.transform
    # cell_labels = build_cell_labels(polys_raster_crs, ref_shape, ref_transform)
    # console.print(f"Cell label raster ready: {ref_shape[0]}×{ref_shape[1]}")

    # console.print("Building cell sort index...")
    # cell_index = build_cell_index(cell_labels, n_cells_all)
    # del cell_labels  # no longer needed; cell_index encodes the same information
    # console.print("Cell index ready")

    # for t, raster_path in enumerate(
    #     track(gbif_rasters, description="Extracting pixel values...")
    # ):
    #     cell_vals = extract_cell_values(
    #         raster_path, polys_raster_crs, bands_for_jsd, cell_index=cell_index
    #     )
    #     for c, band_vals in enumerate(cell_vals):
    #         for b, vals in enumerate(band_vals):
    #             all_cell_values[c][t * n_bands + b] = vals

    # keep = [
    #     any(len(all_cell_values[c][f]) > 0 for f in range(n_features))
    #     for c in range(n_cells_all)
    # ]

    # trait_names = [r.stem for r in gbif_rasters]

    # h3_cells = [cell for cell, k in zip(all_cells, keep) if k]
    # polys_4326_kept = [p for p, k in zip(polys_4326, keep) if k]
    # all_cell_values = [v for v, k in zip(all_cell_values, keep) if k]
    # n_cells = len(h3_cells)
    # console.print(f"Cells with data: {n_cells:,} / {n_cells_all:,}")

    # count_data = {
    #     name: [len(all_cell_values[c][t * n_bands]) for c in range(n_cells)]
    #     for t, name in enumerate(trait_names)
    # }
    # n_traits_with_obs = [
    #     sum(
    #         1
    #         for t in range(len(trait_names))
    #         if len(all_cell_values[c][t * n_bands]) > 0
    #     )
    #     for c in range(n_cells)
    # ]

    # # Compute histograms for each cell and trait
    # console.print("[bold]\nComputing histograms[/bold]")
    # source_band = cfg.jsd.source_band
    # source_band_pos = (
    #     bands_for_jsd.index(source_band) if source_band in bands_for_jsd else None
    # )
    # source_feature_indices = (
    #     {t * n_bands + source_band_pos for t in range(len(gbif_rasters))}
    #     if source_band_pos is not None
    #     else set()
    # )
    # histograms, bin_edges = build_histograms(
    #     all_cell_values, n_bins=n_bins, categorical_features=source_feature_indices
    # )
    # valid_features = sum(1 for e in bin_edges if e is not None)
    # console.print(
    #     f"Features with valid data range: {valid_features} / {n_features} ({len(gbif_rasters)} traits × {n_bands} bands)"
    # )

    # console.print("[bold]\nOptimising split assignment[/bold]")
    # n_train = round(train_frac * n_cells)
    # n_val = round(val_frac * n_cells)
    # n_test = n_cells - n_train - n_val
    # console.print(f"Target: train={n_train}  val={n_val}  test={n_test}")
    # console.print(f"Running {n_restarts} random restarts...")
    # best_labels, best_score = optimize_splits(
    #     histograms, bin_edges, n_train, n_val, n_test, n_restarts, rng
    # )
    # console.print(f"Final best mean JSD: {best_score:.6f}")

    # split_names = {0: "train", 1: "val", 2: "test"}
    # split_labels = [split_names[int(l)] for l in best_labels]

    # console.print("[bold]\nSaving split GeoPackage[/bold]")
    # gdf = build_split_gdf(
    #     h3_cells, polys_4326_kept, split_labels, n_traits_with_obs, count_data
    # )

    # gdf.to_file(out_file, driver="GPKG")
    # console.print(f"Saved: [cyan]{out_file}[/cyan]")

    # console.rule("[bold]Summary[/bold]")
    # counts = gdf["split"].value_counts()
    # for split in ["train", "val", "test"]:
    #     n = counts.get(split, 0)
    #     console.print(f"  {split:5s}: {n:>4d} cells  ({n / n_cells:.1%})")


if __name__ == "__main__":
    main()
