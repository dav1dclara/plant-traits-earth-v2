"""
Estimate spatial autocorrelation range of plant trait rasters via semivariogram analysis.
Fits empirical semivariograms for all traits and recommends an H3 resolution for splitting.

Usage:
    python estimate_autocorrelation_range.py
"""

import json
import warnings
from pathlib import Path

import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

console = Console()

# H3 average edge lengths in km (resolutions 1-6)
H3_EDGE_KM = {1: 418.7, 2: 158.2, 3: 59.8, 4: 22.6, 5: 8.5, 6: 3.2}


# ---------------------------------------------------------------------------
# Semivariogram models
# ---------------------------------------------------------------------------


def _spherical(h, nugget, sill, a):
    gamma = np.where(
        h <= a,
        nugget + sill * (1.5 * h / a - 0.5 * (h / a) ** 3),
        nugget + sill,
    )
    return gamma


def _exponential(h, nugget, sill, a):
    return nugget + sill * (1 - np.exp(-h / a))


def _gaussian(h, nugget, sill, a):
    return nugget + sill * (1 - np.exp(-(h**2) / (a**2)))


MODELS = {
    "spherical": (_spherical, lambda a: a),
    "exponential": (_exponential, lambda a: 3 * a),
    "gaussian": (_gaussian, lambda a: np.sqrt(3) * a),
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    xy: np.ndarray,
    values: np.ndarray,
    n_sample: int,
    grid_n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide bounding box into grid_n x grid_n cells; sample proportionally."""
    x, y = xy[:, 0], xy[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    col_bin = np.floor((x - x_min) / (x_max - x_min + 1e-9) * grid_n).astype(int)
    row_bin = np.floor((y - y_min) / (y_max - y_min + 1e-9) * grid_n).astype(int)
    cell_id = row_bin * grid_n + col_bin

    unique_cells = np.unique(cell_id)
    n_per_cell = max(1, n_sample // len(unique_cells))

    selected = []
    for cell in unique_cells:
        idx = np.where(cell_id == cell)[0]
        k = min(len(idx), n_per_cell)
        selected.append(rng.choice(idx, size=k, replace=False))

    sel_idx = np.concatenate(selected)
    return xy[sel_idx], values[sel_idx]


# ---------------------------------------------------------------------------
# Empirical semivariogram
# ---------------------------------------------------------------------------


def empirical_semivariogram(
    xy_m: np.ndarray, values: np.ndarray, max_lag_km: float, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binned empirical semivariogram. Distances in km."""
    dist_m = cdist(xy_m, xy_m)
    dist_km = dist_m[np.triu_indices(len(xy_m), k=1)] / 1000.0
    z = values
    diff_sq = 0.5 * (z[:, None] - z[None, :]) ** 2
    gamma_flat = diff_sq[np.triu_indices(len(z), k=1)]

    bin_edges = np.linspace(0, max_lag_km, n_lags + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_gamma = np.full(n_lags, np.nan)

    for i in range(n_lags):
        mask = (dist_km >= bin_edges[i]) & (dist_km < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_gamma[i] = gamma_flat[mask].mean()

    valid = ~np.isnan(bin_gamma)
    return bin_centres[valid], bin_gamma[valid]


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_variogram(lags: np.ndarray, gamma: np.ndarray, model_names: list[str]) -> dict:
    """Fit each model, return best fit by RMSE."""
    sill_est = float(np.nanmax(gamma))
    range_est = float(
        lags[np.argmax(gamma > 0.5 * sill_est)]
        if any(gamma > 0.5 * sill_est)
        else lags[-1] / 2
    )
    p0 = [sill_est * 0.1, sill_est * 0.9, range_est]
    bounds = ([0, 0, 0], [sill_est * 2, sill_est * 2, lags[-1] * 2])

    best = {
        "model": None,
        "nugget": np.nan,
        "sill": np.nan,
        "range_km": np.nan,
        "rmse": np.inf,
    }

    for name in model_names:
        fn, practical_range_fn = MODELS[name]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(fn, lags, gamma, p0=p0, bounds=bounds, maxfev=5000)
            nugget, sill, a = popt
            residuals = gamma - fn(lags, *popt)
            rmse = float(np.sqrt(np.mean(residuals**2)))
            if rmse < best["rmse"]:
                best = {
                    "model": name,
                    "nugget": float(nugget),
                    "sill": float(sill),
                    "range_km": float(practical_range_fn(a)),
                    "rmse": rmse,
                }
        except Exception:
            continue

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../../../config/preprocessing",
    config_name="1km",
)
def main(cfg: DictConfig) -> None:
    targets_dir = Path(cfg.semivariogram.targets_dir)
    n_sample = cfg.semivariogram.n_sample
    grid_n = cfg.semivariogram.grid_n
    max_lag_km = cfg.semivariogram.max_lag_km
    n_lags = cfg.semivariogram.n_lags
    model_names = list(cfg.semivariogram.models)

    with open(cfg.trait_mapping) as f:
        trait_mapping = json.load(f)

    rng = np.random.default_rng(42)
    tif_paths = sorted(targets_dir.glob("*.tif"))
    if cfg.semivariogram.max_traits is not None:
        tif_paths = tif_paths[: cfg.semivariogram.max_traits]

    console.rule("[bold blue]Spatial Autocorrelation Range Estimation")
    console.print(
        f"Traits: {len(tif_paths)}  |  Sample: {n_sample}  |  Grid: {grid_n}×{grid_n}  |  Max lag: {max_lag_km} km\n"
    )

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Processing traits...", total=len(tif_paths))
        step_task = progress.add_task("", total=4)

        for tif_path in tif_paths:
            trait_id = tif_path.stem.lstrip("X")
            short_name = trait_mapping.get(trait_id, {}).get("short", "")
            label = f"{tif_path.stem}" + (f" ({short_name})" if short_name else "")

            # Step 1: load
            progress.update(
                step_task,
                description=f"[dim]{label}[/dim] loading raster...",
                completed=0,
            )
            with rasterio.open(tif_path) as src:
                band = src.read(1).astype(np.float32)
                transform = src.transform

            valid_mask = ~np.isnan(band)
            n_valid = int(valid_mask.sum())
            if n_valid < 100:
                progress.console.print(
                    f"  [yellow]⚠ {tif_path.stem} — skipped (only {n_valid} valid pixels)"
                )
                progress.advance(overall)
                continue

            rows, cols = np.where(valid_mask)
            x = transform.c + (cols + 0.5) * transform.a
            y = transform.f + (rows + 0.5) * transform.e
            xy = np.column_stack([x, y])
            values = band[valid_mask]

            # Step 2: sample
            progress.update(
                step_task,
                description=f"[dim]{label}[/dim] stratified sampling...",
                completed=1,
            )
            xy_s, values_s = stratified_sample(xy, values, n_sample, grid_n, rng)
            n_sampled = len(xy_s)

            # Step 3: variogram
            progress.update(
                step_task,
                description=f"[dim]{label}[/dim] computing variogram ({n_sampled} pts)...",
                completed=2,
            )
            lags, gamma = empirical_semivariogram(xy_s, values_s, max_lag_km, n_lags)
            if len(lags) < 5:
                progress.console.print(
                    f"  [yellow]⚠ {tif_path.stem} — skipped (insufficient variogram points)"
                )
                progress.advance(overall)
                continue

            # Step 4: fit
            progress.update(
                step_task,
                description=f"[dim]{label}[/dim] fitting models...",
                completed=3,
            )
            fit = fit_variogram(lags, gamma, model_names)
            fit["trait"] = tif_path.stem
            fit["short_name"] = short_name
            fit["n_valid"] = n_valid
            fit["n_sampled"] = n_sampled
            results.append(fit)

            range_str = (
                f"{fit['range_km']:.0f} km"
                if not np.isnan(fit["range_km"])
                else "failed"
            )
            color = "green" if not np.isnan(fit["range_km"]) else "red"
            progress.console.print(
                f"  [cyan]{tif_path.stem:<10}[/cyan]  {short_name:<20}  "
                f"model=[bold]{fit['model']}[/bold]  range=[{color}]{range_str}[/{color}]"
            )
            progress.update(step_task, completed=4)
            progress.advance(overall)

    # ── Per-trait results table ──────────────────────────────────────────────
    table = Table(title="Semivariogram Results", show_lines=False)
    table.add_column("Trait", style="cyan", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Valid", justify="right")
    table.add_column("Sampled", justify="right")
    table.add_column("Model")
    table.add_column("Nugget", justify="right")
    table.add_column("Sill", justify="right")
    table.add_column("Range (km)", justify="right")

    ranges = []
    for r in results:
        range_km = r["range_km"]
        range_str = f"{range_km:.0f}" if not np.isnan(range_km) else "[red]failed"
        if not np.isnan(range_km):
            ranges.append(range_km)
        table.add_row(
            r["trait"],
            r["short_name"],
            f"{r['n_valid']:,}",
            f"{r['n_sampled']:,}",
            r["model"] or "[red]failed",
            f"{r['nugget']:.3f}",
            f"{r['sill']:.3f}",
            range_str,
        )

    console.print(table)

    if not ranges:
        console.print("[red]No valid range estimates — cannot recommend H3 resolution.")
        return

    # ── Summary ──────────────────────────────────────────────────────────────
    ranges = np.array(ranges)
    median_range = float(np.median(ranges))
    console.print(f"\n[bold]Range summary across {len(ranges)} traits:[/bold]")
    console.print(f"  Median: [green]{median_range:.0f} km[/green]")
    console.print(f"  Mean:   {ranges.mean():.0f} km")
    console.print(f"  Min:    {ranges.min():.0f} km")
    console.print(f"  Max:    {ranges.max():.0f} km")

    # ── H3 resolution recommendation ─────────────────────────────────────────
    console.print(f"\n[bold]H3 resolution recommendation:[/bold]")
    console.print(f"  Target: cell edge length > median range ({median_range:.0f} km)")
    console.print()

    rec_res = None
    for res in sorted(H3_EDGE_KM.keys()):
        edge = H3_EDGE_KM[res]
        marker = ""
        if edge > median_range and rec_res is None:
            rec_res = res
            marker = "  ← [bold green]recommended[/bold green]"
        console.print(f"  H3 res {res}: edge {edge:.0f} km{marker}")

    if rec_res is None:
        console.print(
            f"\n  [yellow]Warning: median range ({median_range:.0f} km) exceeds all listed H3 resolutions."
        )


if __name__ == "__main__":
    main()
