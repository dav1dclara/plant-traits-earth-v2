"""
Create a train/val/test/none split for all traits at 1km resolution.

- H3 cells with splot data: JSD-optimised assignment to train/val/test
- H3 cells with gbif data only: randomly assigned to train/val (never test)
- H3 cells with valid predictor data but no labels: assigned to "none"
- H3 cells with no data at all: discarded
"""

from pathlib import Path

import h3
import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ptev2.data.splitting import (
    build_cell_index,
    build_cell_labels,
    build_histograms_from_rasters,
    build_split_gdf,
    compute_total_jsd,
    count_valid_pixels_per_raster,
    fix_clustering,
    generate_h3_grids,
    spatially_stratified_split,
)

console = Console()


@hydra.main(
    version_base=None,
    config_path="../../../config/1km/preprocessing",
    config_name="splitting",
)
def main(cfg: DictConfig) -> None:
    # Data config
    data_dir = Path(cfg.data_dir)
    resolution_km = cfg.targets.resolution_km
    splot_dir = data_dir / f"{resolution_km}km" / "targets" / "splot"
    gbif_dir = data_dir / f"{resolution_km}km" / "targets" / "gbif"

    console.rule("[bold]CREATING SPLITS[/bold]")
    console.print("[bold]Data config[/bold]")
    console.print(f"Resolution:     [cyan]{resolution_km} km[/cyan]")
    console.print(f"sPlot dir:      [cyan]{splot_dir}[/cyan]")
    console.print(f"GBIF dir:       [cyan]{gbif_dir}[/cyan]")
    console.print()

    # Split config
    h3_resolution = cfg.h3.resolution
    train_frac = cfg.splitting.train_frac
    val_frac = cfg.splitting.val_frac
    test_frac = cfg.splitting.test_frac
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        "Fractions must sum to 1"
    )

    console.print("[bold]Split config[/bold]")
    console.print(f"H3 resolution:  [cyan]{h3_resolution}[/cyan]")
    console.print(
        f"Train/val/test: [cyan]{train_frac:.0%} / {val_frac:.0%} / {test_frac:.0%}[/cyan]"
    )
    console.print()

    # JSD config
    bands_for_jsd = cfg.jsd.bands
    n_restarts = cfg.jsd.n_restarts
    n_bins = cfg.jsd.n_bins
    random_seed = cfg.jsd.random_seed
    rng = np.random.default_rng(random_seed)

    console.print("[bold]JSD config[/bold]")
    console.print(f"JSD bands:      [cyan]{bands_for_jsd}[/cyan]")
    console.print(f"JSD restarts:   [cyan]{n_restarts}[/cyan]")
    console.print(f"JSD bins:       [cyan]{n_bins}[/cyan]")
    console.print(f"Random seed:    [cyan]{random_seed}[/cyan]")
    console.print()

    out_dir = data_dir / f"{resolution_km}km" / "splits"
    out_file = out_dir / f"h3_splits_res{h3_resolution}_{resolution_km}km.gpkg"
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Output file:    [cyan]{out_file}[/cyan]")

    # Load rasters and check counts
    console.print("\n[bold]Checking rasters[/bold]")
    splot_rasters = sorted(splot_dir.glob("*.tif"))
    gbif_rasters = sorted(gbif_dir.glob("*.tif"))
    if cfg.debug.n_traits is not None:
        splot_rasters = splot_rasters[: cfg.debug.n_traits]
        gbif_rasters = gbif_rasters[: cfg.debug.n_traits]
        console.print(
            f"[yellow]DEBUG: using first {cfg.debug.n_traits} trait(s)[/yellow]"
        )
    console.print(
        f"Found [cyan]{len(splot_rasters)}[/cyan] splot rasters, "
        f"[cyan]{len(gbif_rasters)}[/cyan] gbif rasters"
    )

    # Raster CRS and shape come from splot (reference source)
    with rasterio.open(splot_rasters[0]) as src:
        raster_crs = src.crs.to_string()
        ref_shape = src.shape
        ref_transform = src.transform
    console.print(f"Raster CRS: {raster_crs}, shape: {ref_shape[0]}×{ref_shape[1]}")
    for ras in splot_rasters + gbif_rasters:
        with rasterio.open(ras) as src:
            assert src.transform == ref_transform, f"Transform mismatch: {ras}"
    console.print()

    console.print("[bold]Building H3 grid and cell index[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        _t = progress.add_task(f"Generating H3 cells (resolution {h3_resolution})...")
        all_cells, polys_4326, polys_raster_crs = generate_h3_grids(
            h3_resolution, raster_crs
        )
        n_cells_all = len(all_cells)
        progress.update(
            _t,
            description=f"[green]✓[/green] H3 cells generated: {n_cells_all:,}",
            completed=1,
            total=1,
        )

        _t = progress.add_task("Rasterizing H3 cells...")
        cell_labels = build_cell_labels(polys_raster_crs, ref_shape, ref_transform)
        progress.update(
            _t,
            description=f"[green]✓[/green] Cell label raster ready ({ref_shape[0]}×{ref_shape[1]})",
            completed=1,
            total=1,
        )

        _t = progress.add_task("Building cell sort index...")
        cell_index = build_cell_index(cell_labels, n_cells_all)
        progress.update(
            _t,
            description="[green]✓[/green] Cell sort index ready",
            completed=1,
            total=1,
        )
    del cell_labels

    # Splot: build histograms directly (single pass, no pixel-array storage)
    n_bands = len(bands_for_jsd)
    n_splot_features = len(splot_rasters) * n_bands

    console.print("\n[bold]Computing sPlot histograms (single-pass)[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        (
            histograms_all,
            bin_edges,
            has_splot_arr,
            _,
            n_traits_all,
            splot_pixel_counts,
        ) = build_histograms_from_rasters(
            splot_rasters,
            bands_for_jsd,
            cell_index,
            n_cells_all,
            n_bins,
            progress=progress,
        )

    splot_indices = list(np.where(has_splot_arr)[0])
    console.print(f"Cells with splot data: {len(splot_indices):,} / {n_cells_all:,}")
    valid_features = sum(1 for e in bin_edges if e is not None)
    console.print(
        f"Features with valid data range: {valid_features} / {n_splot_features}"
    )

    histograms = histograms_all[splot_indices]
    del histograms_all

    n_splot = len(splot_indices)
    splot_n_traits_obs = n_traits_all[splot_indices].tolist()
    del n_traits_all

    # GBIF: count gbif-only cells before JSD so splot fractions can be adjusted
    console.print("\n[bold]Counting GBIF-only cells[/bold]")
    non_splot_indices = set(np.where(~has_splot_arr)[0].tolist())
    console.print(f"Non-splot cells to check: {len(non_splot_indices):,}")

    console.print("\n[bold]Counting GBIF pixels per cell[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        gbif_pixel_counts = count_valid_pixels_per_raster(
            gbif_rasters, bands_for_jsd[0], cell_index, n_cells_all, progress=progress
        )

    gbif_obs_counts_all = gbif_pixel_counts[:, 0].tolist()
    gbif_only_indices = [c for c in non_splot_indices if gbif_obs_counts_all[c] > 0]
    n_gbif = len(gbif_only_indices)
    console.print(f"GBIF-only cells: {n_gbif:,}")

    gbif_train_frac = train_frac / (train_frac + val_frac)
    n_train_gbif = round(gbif_train_frac * n_gbif)
    n_val_gbif = n_gbif - n_train_gbif
    gbif_base = np.array(["train"] * n_train_gbif + ["val"] * n_val_gbif)
    gbif_splits = list(rng.permutation(gbif_base))
    console.print(f"GBIF-only cells → train={n_train_gbif}  val={n_val_gbif}")

    # Predictor-only cells: no splot or GBIF labels, but valid predictor data
    console.print("\n[bold]Identifying predictor-only cells[/bold]")
    validity_mask_path = Path(cfg.validity_mask)
    labeled_set = set(splot_indices) | set(gbif_only_indices)
    no_label_indices = [c for c in range(n_cells_all) if c not in labeled_set]
    console.print(f"H3 cells with no labels: {len(no_label_indices):,}")

    predict_indices: list[int] = []
    if validity_mask_path.exists():
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            valid_counts = count_valid_pixels_per_raster(
                [validity_mask_path], 1, cell_index, n_cells_all, progress=progress
            )
        predict_indices = [c for c in no_label_indices if valid_counts[c, 0] > 0]
        console.print(
            f"Cells with valid predictor data: [cyan]{len(predict_indices):,}[/cyan] → split='none'"
        )
    else:
        console.print(
            f"[yellow]Validity mask not found at {validity_mask_path} — skipping predict cells.[/yellow]\n"
            "[yellow]Run scripts/utils/build_predictor_validity_mask.py first.[/yellow]"
        )

    # JSD optimisation: splot counts back-solved to hit global train/val/test targets
    console.print("\n[bold]Optimising split assignment[/bold]")
    n_total = n_splot + n_gbif
    n_train_splot = round(train_frac * n_total) - n_train_gbif
    n_val_splot = round(val_frac * n_total) - n_val_gbif
    n_test_splot = n_splot - n_train_splot - n_val_splot
    assert n_train_splot > 0 and n_val_splot > 0 and n_test_splot > 0, (
        f"Invalid splot allocation: train={n_train_splot} val={n_val_splot} test={n_test_splot}. "
        "GBIF contribution too large to hit global targets."
    )
    console.print(
        f"Splot cells → train={n_train_splot}  val={n_val_splot}  test={n_test_splot}"
    )
    stratum_resolution = cfg.splitting.stratum_resolution
    n_strata = len(h3.uncompact_cells(h3.get_res0_cells(), stratum_resolution))
    console.print(f"Running {n_restarts} spatially stratified restarts...")
    console.print(
        f"Stratum H3 resolution: [cyan]{stratum_resolution}[/cyan] ({n_strata:,} cells globally)"
    )

    splot_cell_ids = [all_cells[c] for c in splot_indices]
    best_labels, best_score = spatially_stratified_split(
        histograms,
        bin_edges,
        splot_cell_ids,
        n_train_splot,
        n_val_splot,
        n_test_splot,
        n_restarts,
        rng,
        stratum_resolution=stratum_resolution,
    )
    console.print(f"Best mean JSD: {best_score:.6f}")

    # Post-processing: break up same-split clusters via greedy swaps
    console.print("\n[bold]Fixing split clustering[/bold]")
    cleanup_splits = list(cfg.cleanup.splits)
    max_same_neighbors = cfg.cleanup.max_same_neighbors
    jsd_tolerance = cfg.cleanup.jsd_tolerance
    n_cleanup_iters = cfg.cleanup.n_iters
    console.print(f"Splits to de-cluster:     [cyan]{cleanup_splits}[/cyan]")
    console.print(f"Max same-split neighbors: [cyan]{max_same_neighbors}[/cyan]")
    console.print(f"JSD tolerance:            [cyan]{jsd_tolerance:.0%}[/cyan]")
    console.print(f"Max swap iterations:      [cyan]{n_cleanup_iters:,}[/cyan]")

    best_labels, n_accepted = fix_clustering(
        best_labels,
        splot_cell_ids,
        histograms,
        bin_edges,
        max_same_neighbors=max_same_neighbors,
        jsd_tolerance=jsd_tolerance,
        rng=rng,
        n_iters=n_cleanup_iters,
        splits=cleanup_splits,
    )
    final_jsd = compute_total_jsd(histograms, best_labels, bin_edges)
    console.print(f"Swaps accepted: [cyan]{n_accepted:,}[/cyan]")
    console.print(f"Final JSD:      [cyan]{final_jsd:.6f}[/cyan]")

    split_map = {0: "train", 1: "val", 2: "test"}
    splot_splits = [split_map[int(l)] for l in best_labels]

    # Assemble output GeoDataFrame
    console.print("\n[bold]Saving split GeoPackage[/bold]")

    h3_cells_out = (
        [all_cells[c] for c in splot_indices]
        + [all_cells[c] for c in gbif_only_indices]
        + [all_cells[c] for c in predict_indices]
    )
    polys_out = (
        [polys_4326[c] for c in splot_indices]
        + [polys_4326[c] for c in gbif_only_indices]
        + [polys_4326[c] for c in predict_indices]
    )
    splits_out = splot_splits + gbif_splits + ["none"] * len(predict_indices)
    source_out = (
        ["splot"] * n_splot
        + ["gbif_only"] * n_gbif
        + ["predictor_only"] * len(predict_indices)
    )
    n_traits_out = splot_n_traits_obs + [0] * n_gbif + [0] * len(predict_indices)

    # Per-trait pixel counts for every output cell
    cells_out = splot_indices + list(gbif_only_indices) + predict_indices
    per_trait_counts = {}
    for i, raster in enumerate(splot_rasters):
        col = splot_pixel_counts[cells_out, i].tolist()
        per_trait_counts[f"{raster.stem}_splot"] = col
    for i, raster in enumerate(gbif_rasters):
        col = gbif_pixel_counts[cells_out, i].tolist()
        per_trait_counts[f"{raster.stem}_gbif"] = col

    gdf = build_split_gdf(
        h3_cells_out,
        polys_out,
        splits_out,
        n_traits_out,
        per_trait_counts,
    )
    gdf.insert(2, "data_source", source_out)

    gdf.to_file(out_file, driver="GPKG")
    console.print(f"Saved: [cyan]{out_file}[/cyan]")

    console.rule("[bold]Summary[/bold]")
    n_total = len(gdf)
    for source, splits in [
        ("splot", ["train", "val", "test"]),
        ("gbif_only", ["train", "val"]),
        ("predictor_only", ["none"]),
    ]:
        sub = gdf[gdf["data_source"] == source]
        if len(sub) == 0:
            continue
        console.print(f"[bold]{source}[/bold]: {len(sub):,} cells")
        for split in splits:
            n = (sub["split"] == split).sum()
            console.print(f"  {split:8s}: {n:>5,} cells  ({n / len(sub):.1%})")
    console.print(f"[bold]Total:[/bold] {n_total:,} cells")
    for split in ["train", "val", "test", "none"]:
        n = (gdf["split"] == split).sum()
        if n > 0:
            console.print(f"  {split:8s}: {n:>5,} cells  ({n / n_total:.1%})")


if __name__ == "__main__":
    main()
