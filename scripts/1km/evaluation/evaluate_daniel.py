"""
Evaluate Daniel's 1km predictions against sPlot original-space targets.

R² is computed over all pixels where the sPlot target has a valid value.
Rasters are read window-by-window — no full raster is loaded into memory.

Usage:
    python evaluate_daniel.py              # default: X3106
    python evaluate_daniel.py --trait X14
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()

PRED_DIR = Path("/scratch3/plant-traits-v2/data/1km/predictions_daniel")
SPLOT_DIR = Path("/scratch3/plant-traits-v2/data/1km/targets/splot_original")


def find_prediction(trait: str) -> Path:
    matches = sorted(PRED_DIR.glob(f"{trait}_*.tif"))
    if not matches:
        raise FileNotFoundError(
            f"No prediction file found for trait '{trait}' in {PRED_DIR}"
        )
    if len(matches) > 1:
        console.print(
            f"[yellow]Multiple prediction files found for {trait}, using first:[/yellow]"
        )
        for m in matches:
            console.print(f"  [dim]{m.name}[/dim]")
    return matches[0]


def main(trait: str) -> None:
    console.rule(f"[bold]Evaluating {trait}[/bold]")

    pred_path = find_prediction(trait)
    splot_path = SPLOT_DIR / f"{trait}.tif"

    if not splot_path.exists():
        console.print(f"[red]sPlot original not found: {splot_path}[/red]")
        console.print("[dim]Run unnormalize_targets.py --source splot first.[/dim]")
        return

    # Read scale/offset and reported model performance from prediction metadata
    with rasterio.open(pred_path) as src:
        pred_scale = src.scales[0]
        pred_offset = src.offsets[0]
        tags = src.tags()

    reported_r2 = tags.get("model_performance", None)
    trait_name = tags.get("trait_long_name", "")
    trait_unit = tags.get("trait_unit", "")

    console.print(f"Prediction:  [cyan]{pred_path.name}[/cyan]")
    console.print(f"sPlot obs:   [cyan]{splot_path.name}[/cyan]")
    if trait_name:
        console.print(f"Trait:       [dim]{trait_name} ({trait_unit})[/dim]")
    if reported_r2:
        console.print(f"Reported CV: [dim]{reported_r2}[/dim]")
    console.print()

    # Accumulators for streaming R², RMSE, bias
    # SS_tot = Σy² - n*ȳ²  (computed after all windows via König-Huygens)
    n = np.int64(0)
    sum_true = np.float64(0)
    sum_pred = np.float64(0)
    sum_sq_true = np.float64(0)
    ss_res = np.float64(0)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with rasterio.open(splot_path) as splot_src, rasterio.open(pred_path) as pred_src:
        splot_nodata = splot_src.nodata
        pred_nodata = pred_src.nodata
        windows = list(splot_src.block_windows(1))

        with Progress(
            TextColumn("[bold cyan]Reading windows"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=len(windows))

            for _, window in windows:
                obs = splot_src.read(1, window=window).astype(np.float64)
                pred = pred_src.read(1, window=window).astype(np.float64)

                if splot_nodata is not None:
                    obs[obs == splot_nodata] = np.nan
                # Mask nodata before applying scale/offset
                if pred_nodata is not None:
                    pred[pred == pred_nodata] = np.nan
                pred = pred * pred_scale + pred_offset

                valid = np.isfinite(obs) & np.isfinite(pred)
                if not valid.any():
                    progress.advance(task)
                    continue

                yt = obs[valid]
                yp = pred[valid]

                n += yt.size
                sum_true += yt.sum()
                sum_pred += yp.sum()
                sum_sq_true += (yt**2).sum()
                ss_res += ((yt - yp) ** 2).sum()
                all_true.append(yt.astype(np.float32))
                all_pred.append(yp.astype(np.float32))

                progress.advance(task)

    if n == 0:
        console.print("[red]No overlapping valid pixels found.[/red]")
        return

    mean_true = sum_true / n
    mean_pred = sum_pred / n
    ss_tot = sum_sq_true - n * mean_true**2
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / n))
    bias = float(mean_pred - mean_true)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Valid pixels (sPlot)", f"{int(n):,}")
    table.add_row("R²", f"{r2:.4f}")
    table.add_row("RMSE", f"{rmse:.4f}")
    table.add_row("Bias (pred − obs)", f"{bias:.4f}")
    table.add_row("Mean observed", f"{mean_true:.4f}")
    table.add_row("Mean predicted", f"{mean_pred:.4f}")

    console.print()
    console.print(table)

    # Histogram plot
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Shared bin edges over the combined range
    lo = float(np.percentile(np.concatenate([y_true, y_pred]), 1))
    hi = float(np.percentile(np.concatenate([y_true, y_pred]), 99))
    bins = np.linspace(lo, hi, 80)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        y_true,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        label=f"sPlot observed (n={int(n):,})",
    )
    ax.hist(
        y_pred,
        bins=bins,
        density=True,
        alpha=0.6,
        color="coral",
        label=f"Predicted (n={int(n):,})",
    )
    ax.axvline(float(mean_true), color="steelblue", lw=1.5, ls="--")
    ax.axvline(float(mean_pred), color="coral", lw=1.5, ls="--")

    title = f"{trait}"
    if trait_name:
        title += f" — {trait_name}"
        if trait_unit:
            title += f" ({trait_unit})"
    title += f"\nR²={r2:.3f}  RMSE={rmse:.3f}  Bias={bias:.3f}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"{trait}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"\nPlot saved to [cyan]{out_path}[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", default="X3106", help="Trait ID, e.g. X3106")
    args = parser.parse_args()
    main(args.trait)
