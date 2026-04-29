"""
For each biome (BIOME_NAME) in the Ecoregions2017 shapefile, compute summary
statistics across a set of trait TIFFs and print a table.

Each trait TIFF has two bands:
  - Band 1 (mean): NaN where no observation exists for that trait
  - Band 2 (source): 1 = GBIF, 2 = sPlot, NaN where no observation

Columns per biome (aggregated across traits):
  Total px   — raster pixels inside the biome (trait-independent)
  Valid %    — min / avg / max fraction of biome pixels with a valid observation
  GBIF %     — min / avg / max fraction of valid pixels that are GBIF
  sPlot %    — min / avg / max fraction of valid pixels that are sPlot

Usage:
    python ecoregion_trait_coverage.py
"""

import os
from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.features import rasterize
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

console = Console(width=160)

ECOREGIONS_PATH = Path(
    "/scratch3/plant-traits-v2/data/other/Ecoregions2017/Ecoregions2017.shp"
)
GBIF_VAL = 1.0
SPLOT_VAL = 2.0
BLOCK_ROWS = 500
N_TRAITS = None  # None = all traits


@hydra.main(
    version_base=None,
    config_path="../../../config/preprocessing",
    config_name="1km",
)
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------ #
    # Load ecoregions and build biome index
    # ------------------------------------------------------------------ #
    console.print("[bold]Loading ecoregions ...[/bold]")
    gdf = gpd.read_file(ECOREGIONS_PATH)

    biome_names = sorted(gdf[gdf["BIOME_NAME"] != "N/A"]["BIOME_NAME"].unique())
    biome_to_idx = {name: i for i, name in enumerate(biome_names)}
    n_biomes = len(biome_names)
    NA_IDX = n_biomes  # index reserved for N/A polygons (land but no named biome)
    console.print(f"  {n_biomes} biomes, {len(gdf)} ecoregions")

    # ------------------------------------------------------------------ #
    # Trait TIFFs
    # ------------------------------------------------------------------ #
    targets_dir = Path(cfg.base_dir) / "targets" / cfg.targets.source
    tif_paths = sorted(targets_dir.glob("*.tif"))[:N_TRAITS]
    n_traits = len(tif_paths)
    trait_ids = [p.stem for p in tif_paths]
    console.print(f"  Traits ({n_traits}): {', '.join(trait_ids)}")

    # ------------------------------------------------------------------ #
    # Reference grid + biome raster (built once)
    # ------------------------------------------------------------------ #
    with rasterio.open(tif_paths[0]) as src:
        raster_crs = src.crs
        transform = src.transform
        n_rows, n_cols = src.shape

    console.print(f"  Grid: {n_rows} × {n_cols}  CRS: {raster_crs}")
    console.print("[bold]Rasterizing biomes ...[/bold]")

    gdf_proj = gdf.to_crs(raster_crs)
    gdf_proj = gdf_proj.assign(
        biome_idx=gdf_proj["BIOME_NAME"].map(biome_to_idx).fillna(NA_IDX).astype(int)
    )
    shapes = [
        (geom, int(idx))
        for geom, idx in zip(gdf_proj.geometry, gdf_proj["biome_idx"])
        if geom is not None and not geom.is_empty
    ]
    biome_raster = rasterize(
        shapes,
        out_shape=(n_rows, n_cols),
        transform=transform,
        fill=-1,
        dtype=np.int16,
    )

    # Total pixels per biome — trait-independent (index n_biomes = N/A land polygons)
    n_total_px = np.zeros(n_biomes + 1, dtype=np.int64)
    np.add.at(n_total_px, biome_raster[biome_raster >= 0], 1)

    # ------------------------------------------------------------------ #
    # Per-trait counts: shape (n_traits, n_biomes)
    # ------------------------------------------------------------------ #
    n_gbif_all = np.zeros((n_traits, n_biomes + 1), dtype=np.int32)
    n_splot_all = np.zeros((n_traits, n_biomes + 1), dtype=np.int32)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing traits ...", total=n_traits)

        for t_idx, tif_path in enumerate(tif_paths):
            progress.update(task, description=f"Processing {tif_path.stem} ...")
            with rasterio.open(tif_path) as src:
                for row_start in range(0, n_rows, BLOCK_ROWS):
                    row_end = min(row_start + BLOCK_ROWS, n_rows)
                    window = rasterio.windows.Window(
                        col_off=0,
                        row_off=row_start,
                        width=n_cols,
                        height=row_end - row_start,
                    )
                    mean_chunk = src.read(1, window=window)
                    source_chunk = src.read(2, window=window)
                    biome_chunk = biome_raster[row_start:row_end]

                    valid = ~np.isnan(mean_chunk) & (biome_chunk >= 0)
                    np.add.at(
                        n_gbif_all[t_idx],
                        biome_chunk[valid & (source_chunk == GBIF_VAL)],
                        1,
                    )
                    np.add.at(
                        n_splot_all[t_idx],
                        biome_chunk[valid & (source_chunk == SPLOT_VAL)],
                        1,
                    )

            progress.advance(task)

    # ------------------------------------------------------------------ #
    # Compute per-biome percentages across traits
    # ------------------------------------------------------------------ #
    n_valid_all = n_gbif_all + n_splot_all  # (n_traits, n_biomes)

    # pct_valid[t, b] = valid pixels / total biome pixels
    pct_valid = np.where(
        n_total_px > 0, n_valid_all / n_total_px[np.newaxis, :], np.nan
    )
    # pct_gbif/splot[t, b] = source pixels / valid pixels
    pct_gbif = np.where(n_valid_all > 0, n_gbif_all / n_valid_all, np.nan)
    pct_splot = np.where(n_valid_all > 0, n_splot_all / n_valid_all, np.nan)

    valid_avg = np.nanmean(pct_valid, axis=0)
    gbif_avg = np.nanmean(pct_gbif, axis=0)
    splot_avg = np.nanmean(pct_splot, axis=0)

    # Average absolute pixel counts across traits
    valid_px_avg = np.nanmean(n_valid_all.astype(float), axis=0)
    gbif_px_avg = np.nanmean(n_gbif_all.astype(float), axis=0)
    splot_px_avg = np.nanmean(n_splot_all.astype(float), axis=0)

    # ------------------------------------------------------------------ #
    # Print table
    # ------------------------------------------------------------------ #
    console.rule(f"[bold blue]Biome coverage — {n_traits} traits")

    table = Table(show_lines=False)
    table.add_column("Biome", style="cyan", no_wrap=True)
    table.add_column("Total px", justify="right", min_width=14)
    table.add_column("Valid px (avg)", justify="right", min_width=14)
    table.add_column("Valid avg%", justify="right")
    table.add_column("GBIF px (avg)", justify="right", min_width=14)
    table.add_column("GBIF avg%", justify="right")
    table.add_column("sPlot px (avg)", justify="right", min_width=14)
    table.add_column("sPlot avg%", justify="right")

    # Sort named biomes by size; N/A row appended separately
    order = np.argsort(n_total_px[:n_biomes])[::-1]
    for i in order:
        table.add_row(
            biome_names[i],
            f"{int(n_total_px[i]):,}",
            f"{valid_px_avg[i]:,.0f}",
            f"{valid_avg[i]:.1%}",
            f"{gbif_px_avg[i]:,.0f}",
            f"{gbif_avg[i]:.1%}",
            f"{splot_px_avg[i]:,.0f}",
            f"{splot_avg[i]:.1%}",
        )

    table.add_section()
    table.add_row(
        "[dim]N/A biome (land, no named biome)[/dim]",
        f"[dim]{int(n_total_px[NA_IDX]):,}[/dim]",
        f"[dim]{valid_px_avg[NA_IDX]:,.0f}[/dim]",
        f"[dim]{valid_avg[NA_IDX]:.1%}[/dim]",
        f"[dim]{gbif_px_avg[NA_IDX]:,.0f}[/dim]",
        f"[dim]{gbif_avg[NA_IDX]:.1%}[/dim]",
        f"[dim]{splot_px_avg[NA_IDX]:,.0f}[/dim]",
        f"[dim]{splot_avg[NA_IDX]:.1%}[/dim]",
    )

    table.add_section()
    table.add_row(
        "[bold]TOTAL / AVG[/bold]",
        f"[bold]{int(n_total_px.sum()):,}[/bold]",
        f"[bold]{valid_px_avg.sum():,.0f}[/bold]",
        f"[bold]{np.nanmean(valid_avg):.1%}[/bold]",
        f"[bold]{gbif_px_avg.sum():,.0f}[/bold]",
        f"[bold]{np.nanmean(gbif_avg):.1%}[/bold]",
        f"[bold]{splot_px_avg.sum():,.0f}[/bold]",
        f"[bold]{np.nanmean(splot_avg):.1%}[/bold]",
    )

    console.print(table)

    # ------------------------------------------------------------------ #
    # Write markdown table
    # ------------------------------------------------------------------ #
    headers = [
        "Biome",
        "Total px",
        "Valid px (avg)",
        "Valid avg%",
        "GBIF px (avg)",
        "GBIF avg%",
        "sPlot px (avg)",
        "sPlot avg%",
    ]

    def md_row(cells):
        return "| " + " | ".join(cells) + " |"

    rows = [
        md_row(headers),
        md_row(["---"] * len(headers)),
    ]
    for i in order:
        rows.append(
            md_row(
                [
                    biome_names[i],
                    f"{int(n_total_px[i]):,}",
                    f"{valid_px_avg[i]:,.0f}",
                    f"{valid_avg[i]:.1%}",
                    f"{gbif_px_avg[i]:,.0f}",
                    f"{gbif_avg[i]:.1%}",
                    f"{splot_px_avg[i]:,.0f}",
                    f"{splot_avg[i]:.1%}",
                ]
            )
        )
    rows.append(
        md_row(
            [
                "*N/A biome (land, no named biome)*",
                f"{int(n_total_px[NA_IDX]):,}",
                f"{valid_px_avg[NA_IDX]:,.0f}",
                f"{valid_avg[NA_IDX]:.1%}",
                f"{gbif_px_avg[NA_IDX]:,.0f}",
                f"{gbif_avg[NA_IDX]:.1%}",
                f"{splot_px_avg[NA_IDX]:,.0f}",
                f"{splot_avg[NA_IDX]:.1%}",
            ]
        )
    )
    rows.append(
        md_row(
            [
                "**TOTAL / AVG**",
                f"**{int(n_total_px.sum()):,}**",
                f"**{valid_px_avg.sum():,.0f}**",
                f"**{np.nanmean(valid_avg):.1%}**",
                f"**{gbif_px_avg.sum():,.0f}**",
                f"**{np.nanmean(gbif_avg):.1%}**",
                f"**{splot_px_avg.sum():,.0f}**",
                f"**{np.nanmean(splot_avg):.1%}**",
            ]
        )
    )

    md_path = Path(__file__).parent / "ecoregion_trait_coverage.md"
    md_path.write_text(
        f"# Biome coverage — {n_traits} traits\n\n" + "\n".join(rows) + "\n"
    )
    console.print(f"\n[bold green]Markdown saved:[/bold green] {md_path}")


if __name__ == "__main__":
    main()
