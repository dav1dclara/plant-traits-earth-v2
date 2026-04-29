"""
Print statistics and integrity checks for the 1km H3 split file.
"""

from pathlib import Path

import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track
from rich.table import Table
from scipy.stats import gaussian_kde

from ptev2.data.splitting import build_cell_index, build_cell_labels

console = Console()

SPLITS = ["train", "val", "test"]
SOURCES = ["splot", "gbif_only"]
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}

plt.rcParams["font.family"] = "monospace"


def section(title: str) -> None:
    console.print(f"\n[bold]{title}[/bold]")


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
    console.print(f"  {mark}  {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))


@hydra.main(
    version_base=None,
    config_path="../../../config/1km",
    config_name="splitting",
)
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    resolution_km = cfg.targets.resolution_km
    h3_resolution = cfg.h3.resolution
    split_file = (
        data_dir
        / f"{resolution_km}km"
        / "splits"
        / f"h3_splits_res{h3_resolution}_{resolution_km}km.gpkg"
    )

    console.rule("[bold]SPLIT INSPECTION[/bold]")
    console.print(f"File: [cyan]{split_file}[/cyan]")

    if not split_file.exists():
        console.print("[red]File not found.[/red]")
        return

    gdf = gpd.read_file(split_file)
    console.print(f"Loaded [cyan]{len(gdf):,}[/cyan] cells\n")

    # -------------------------------------------------------------------------
    # Integrity checks
    # -------------------------------------------------------------------------
    section("Integrity checks")
    check("No duplicate H3 cells", gdf["h3_index"].nunique() == len(gdf))
    check(
        "All cells assigned to train/val/test",
        gdf["split"].isin(SPLITS).all(),
        f"unexpected: {sorted(set(gdf['split']) - set(SPLITS))}",
    )
    check(
        "All cells have a known source",
        gdf["data_source"].isin(SOURCES).all(),
    )
    check(
        "gbif_only cells never in test",
        (gdf[gdf["data_source"] == "gbif_only"]["split"] != "test").all(),
    )
    check(
        "splot and gbif_only are mutually exclusive",
        gdf["data_source"].value_counts().sum() == len(gdf),
    )

    # -------------------------------------------------------------------------
    # Split counts
    # -------------------------------------------------------------------------
    section("Split counts")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Source", style="cyan")
    for s in SPLITS:
        table.add_column(s.capitalize(), justify="right")
    table.add_column("Total", justify="right", style="bold")

    n_total = len(gdf)
    for source in SOURCES:
        sub = gdf[gdf["data_source"] == source]
        row = [source]
        for s in SPLITS:
            n = (sub["split"] == s).sum()
            pct = n / len(sub) * 100 if len(sub) > 0 else 0
            row.append(f"{n:,}  ({pct:.0f}%)")
        row.append(f"{len(sub):,}")
        table.add_row(*row)

    # Total row
    row = ["[bold]Total[/bold]"]
    for s in SPLITS:
        n = (gdf["split"] == s).sum()
        pct = n / n_total * 100
        row.append(f"{n:,}  ({pct:.0f}%)")
    row.append(f"{n_total:,}")
    table.add_row(*row, style="bold")

    console.print(table)

    # -------------------------------------------------------------------------
    # Observation counts per split
    # -------------------------------------------------------------------------
    section("Observation counts per split")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Split", style="cyan")
    table.add_column("splot_obs  median", justify="right")
    table.add_column("splot_obs  mean", justify="right")
    table.add_column("gbif_obs  median", justify="right")
    table.add_column("gbif_obs  mean", justify="right")

    splot_cells = gdf[gdf["data_source"] == "splot"]
    for s in SPLITS:
        sp = splot_cells[splot_cells["split"] == s]["splot_obs_count"]
        gb = gdf[gdf["split"] == s]["gbif_obs_count"]
        table.add_row(
            s,
            f"{np.median(sp):.0f}" if len(sp) > 0 else "—",
            f"{np.mean(sp):.0f}" if len(sp) > 0 else "—",
            f"{np.median(gb):.0f}" if len(gb) > 0 else "—",
            f"{np.mean(gb):.0f}" if len(gb) > 0 else "—",
        )
    console.print(table)

    # -------------------------------------------------------------------------
    # Trait coverage per split (splot cells only)
    # -------------------------------------------------------------------------
    section("Trait coverage per split  [dim](splot cells only)[/dim]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Split", style="cyan")
    table.add_column("n_traits  min", justify="right")
    table.add_column("n_traits  median", justify="right")
    table.add_column("n_traits  max", justify="right")

    for s in SPLITS:
        nt = splot_cells[splot_cells["split"] == s]["n_traits"]
        if len(nt) == 0:
            table.add_row(s, "—", "—", "—")
        else:
            table.add_row(
                s, str(int(nt.min())), f"{np.median(nt):.0f}", str(int(nt.max()))
            )
    console.print(table)

    # -------------------------------------------------------------------------
    # Per-trait pixel counts
    # -------------------------------------------------------------------------
    splot_trait_cols = sorted(
        c for c in gdf.columns if c.endswith("_splot") and c != "splot_obs_count"
    )
    gbif_trait_cols = sorted(
        c for c in gdf.columns if c.endswith("_gbif") and c != "gbif_obs_count"
    )

    if splot_trait_cols or gbif_trait_cols:
        section("Per-trait pixel counts")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Trait", style="cyan")
        for s in SPLITS:
            table.add_column(f"splot {s}", justify="right")
        table.add_column("splot total", justify="right", style="bold")
        table.add_column("gbif total", justify="right")

        traits = sorted(
            {c.rsplit("_", 1)[0] for c in splot_trait_cols + gbif_trait_cols}
        )
        for trait in traits:
            splot_col = f"{trait}_splot"
            gbif_col = f"{trait}_gbif"
            row = [trait]
            splot_total = 0
            for s in SPLITS:
                split_mask = gdf["split"] == s
                if splot_col in gdf.columns:
                    n = int((gdf.loc[split_mask, splot_col] > 0).sum())
                    px = int(gdf.loc[split_mask, splot_col].sum())
                    row.append(f"{n:,} cells / {px:,} px")
                    splot_total += px
                else:
                    row.append("—")
            row.append(f"{splot_total:,}")
            if gbif_col in gdf.columns:
                gbif_total = int(gdf[gbif_col].sum())
                row.append(f"{gbif_total:,}")
            else:
                row.append("—")
            table.add_row(*row)
        console.print(table)

    plot_distributions(cfg, gdf)


def plot_distributions(cfg: DictConfig, gdf: gpd.GeoDataFrame) -> None:
    splot_dir = (
        Path(cfg.data_dir) / f"{cfg.targets.resolution_km}km" / "targets" / "splot"
    )
    splot_rasters = sorted(splot_dir.glob("*.tif"))
    if cfg.debug.n_traits is not None:
        splot_rasters = splot_rasters[: cfg.debug.n_traits]

    if not splot_rasters:
        console.print("[red]No splot rasters found — skipping plot.[/red]")
        return

    # Work only with splot cells (they carry the trait measurements)
    splot_cells = gdf[gdf["data_source"] == "splot"].copy()
    splits = list(splot_cells["split"])

    with rasterio.open(splot_rasters[0]) as src:
        raster_crs = src.crs.to_string()
        ref_shape = src.shape
        ref_transform = src.transform

    cell_polys = list(splot_cells.to_crs(raster_crs).geometry)

    console.print("\n[bold]Building cell index for plot...[/bold]")
    cell_labels = build_cell_labels(cell_polys, ref_shape, ref_transform)
    cell_index = build_cell_index(cell_labels, len(cell_polys))
    del cell_labels

    order, boundaries = cell_index
    n_cells = len(cell_polys)

    n_traits = len(splot_rasters)
    fig, axes = plt.subplots(n_traits, 1, figsize=(7, 3 * n_traits), squeeze=False)

    for i, raster_path in enumerate(
        track(splot_rasters, description="Plotting traits...")
    ):
        ax = axes[i, 0]

        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
        sorted_band = data.ravel()[order]

        split_values: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
        for idx in range(n_cells):
            v = sorted_band[boundaries[idx + 1] : boundaries[idx + 2]]
            v = v[np.isfinite(v)]
            if len(v) > 0 and splits[idx] in split_values:
                split_values[splits[idx]].append(v)

        for split in SPLITS:
            vals = (
                np.concatenate(split_values[split])
                if split_values[split]
                else np.array([])
            )
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals, bw_method="scott")
            x = np.linspace(vals.min(), vals.max(), 500)
            y = kde(x)
            color = SPLIT_COLORS[split]
            ax.plot(
                x, y, color=color, linewidth=1.5, label=f"{split} (n={len(vals):,})"
            )
            ax.fill_between(x, y, alpha=0.15, color=color)

        ax.set_title(raster_path.stem, fontsize=9, loc="left")
        ax.set_xlabel("mean value")
        ax.set_ylabel("density")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_path = (
        Path(__file__).parent
        / f"split_distributions_res{cfg.h3.resolution}_{cfg.targets.resolution_km}km.png"
    )
    fig.savefig(out_path, dpi=150)
    console.print(f"Saved: [cyan]{out_path}[/cyan]")


if __name__ == "__main__":
    main()
