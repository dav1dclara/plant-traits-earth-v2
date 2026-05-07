"""
Re-save GBIF and/or sPlot target rasters with zstd compression.

GBIF:  recompresses in-place (no band reordering).
sPlot: reorders bands to match GBIF order (mean, std, median, q05, q95, count).

Both read and write one tile at a time — constant memory regardless of raster size.

Usage:
    python prepare_targets.py                  # both (default)
    python prepare_targets.py --source gbif
    python prepare_targets.py --source splot
    python prepare_targets.py --source both
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import rasterio
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

os.environ.setdefault("GDAL_NUM_THREADS", "1")

console = Console()

# Parse --source before hydra consumes sys.argv
_source = "both"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--source" and _i + 1 < len(sys.argv):
        _source = sys.argv.pop(_i + 1)
        sys.argv.pop(_i)
        break
    elif _arg.startswith("--source="):
        _source = _arg.split("=", 1)[1]
        sys.argv.pop(_i)
        break

if _source not in {"gbif", "splot", "both"}:
    console.print(
        f"[red]Invalid --source '{_source}'. Choose from: gbif, splot, both[/red]"
    )
    sys.exit(1)


def process_gbif_tif(tif: Path, out_dir: Path, output_cfg: dict) -> tuple[str, float]:
    out_path = out_dir / tif.name

    with rasterio.open(tif) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            zstd_level=output_cfg["zstd_level"],
            tiled=True,
            blockxsize=output_cfg["blockxsize"],
            blockysize=output_cfg["blockysize"],
        )
        bands = src.descriptions

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = bands
            for _, window in src.block_windows(1):
                for band_idx in range(1, src.count + 1):
                    dst.write(
                        src.read(band_idx, window=window).astype("float32"),
                        band_idx,
                        window=window,
                    )

    with rasterio.open(out_path, "r+") as dst:
        dst.build_overviews(output_cfg["overviews"], rasterio.enums.Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    return tif.stem, out_path.stat().st_size / 1e6


def process_splot_tif(
    tif: Path, out_dir: Path, target_bands: tuple, output_cfg: dict
) -> tuple[str, float]:
    out_path = out_dir / tif.name

    with rasterio.open(tif) as src:
        src_bands = src.descriptions
        band_order = [src_bands.index(b) + 1 for b in target_bands]
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            zstd_level=output_cfg["zstd_level"],
            tiled=True,
            blockxsize=output_cfg["blockxsize"],
            blockysize=output_cfg["blockysize"],
        )
        windows = list(src.block_windows(1))

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.descriptions = target_bands
            for _, window in windows:
                for out_idx, src_idx in enumerate(band_order, start=1):
                    dst.write(
                        src.read(src_idx, window=window).astype("float32"),
                        out_idx,
                        window=window,
                    )

    with rasterio.open(out_path, "r+") as dst:
        dst.build_overviews(output_cfg["overviews"], rasterio.enums.Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    return tif.stem, out_path.stat().st_size / 1e6


def run_gbif(cfg: DictConfig) -> None:
    in_dir = Path(cfg.targets.gbif.in_dir)
    out_dir = Path(cfg.targets.gbif.out_dir)
    n_workers = cfg.targets.gbif.n_workers
    output_cfg = {
        "zstd_level": cfg.targets.gbif.output.zstd_level,
        "blockxsize": cfg.targets.gbif.output.blockxsize,
        "blockysize": cfg.targets.gbif.output.blockysize,
        "overviews": list(cfg.targets.gbif.output.overviews),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = sorted(in_dir.glob("*.tif"))
    console.rule("[bold blue]GBIF")
    console.print(f"Found {len(tifs)} rasters — processing with {n_workers} workers\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids = {
            tif.stem: progress.add_task(f"{tif.stem} ...", total=None) for tif in tifs
        }

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(process_gbif_tif, tif, out_dir, output_cfg): tif
                for tif in tifs
            }
            for fut in as_completed(futures):
                stem, size_mb = fut.result()
                progress.update(
                    task_ids[stem],
                    description=f"[green]✓[/green] {stem:<25} {size_mb:.1f} MB",
                    completed=1,
                    total=1,
                )

    console.print(f"\n[bold green]Done![/bold green] Output: {out_dir}\n")


def run_splot(cfg: DictConfig) -> None:
    in_dir = Path(cfg.targets.splot.in_dir)
    out_dir = Path(cfg.targets.splot.out_dir)
    gbif_ref = Path(cfg.targets.splot.gbif_ref)
    n_workers = cfg.targets.splot.n_workers
    output_cfg = {
        "zstd_level": cfg.targets.splot.output.zstd_level,
        "blockxsize": cfg.targets.splot.output.blockxsize,
        "blockysize": cfg.targets.splot.output.blockysize,
        "overviews": list(cfg.targets.splot.output.overviews),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(gbif_ref) as ref:
        target_bands = ref.descriptions

    tifs = sorted(in_dir.glob("*.tif"))
    console.rule("[bold blue]sPlot")
    console.print(f"Target band order: {target_bands}")
    console.print(f"Found {len(tifs)} rasters — processing with {n_workers} workers\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_ids = {
            tif.stem: progress.add_task(f"{tif.stem} ...", total=None) for tif in tifs
        }

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    process_splot_tif, tif, out_dir, target_bands, output_cfg
                ): tif
                for tif in tifs
            }
            for fut in as_completed(futures):
                stem, size_mb = fut.result()
                progress.update(
                    task_ids[stem],
                    description=f"[green]✓[/green] {stem:<25} {size_mb:.1f} MB",
                    completed=1,
                    total=1,
                )

    console.print(f"\n[bold green]Done![/bold green] Output: {out_dir}\n")


@hydra.main(
    version_base=None,
    config_path="../../../config/1km/preprocessing",
    config_name="preprocessing",
)
def main(cfg: DictConfig) -> None:
    if _source in {"gbif", "both"}:
        run_gbif(cfg)
    if _source in {"splot", "both"}:
        run_splot(cfg)


if __name__ == "__main__":
    main()
