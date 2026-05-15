"""Chip 1km predictor and target rasters into per-split HDF5 stores.

Drop-in replacement for chip_rasters.py that writes .h5 files instead of .zarr
directories. The gpkg bounds export reads from the HDF5 files directly.
"""

from pathlib import Path

import geopandas as gpd
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from shapely.geometry import box

from ptev2.data.chipping import chip_rasters_to_zarr

console = Console()


@hydra.main(
    version_base=None,
    config_path="../../../config/1km/preprocessing",
    config_name="chipping",
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold]CHIPPING 1km DATA TO HDF5[/bold]")

    predictors_dir = Path(cfg.paths.predictors_dir)
    targets_dir = Path(cfg.paths.targets_dir)
    splits_file = Path(cfg.paths.splits_file)
    zarr_dir = Path(cfg.paths.zarr_dir)

    patch_size = cfg.settings.patch_size
    stride = cfg.settings.stride
    stride_per_split = {"train": stride, "val": stride, "test": stride}

    used_predictors = {
        name: pred_cfg for name, pred_cfg in cfg.data.predictors.items() if pred_cfg.use
    }
    used_targets = {
        name: target_cfg
        for name, target_cfg in cfg.data.targets.items()
        if target_cfg.use
    }

    predictor_paths = {
        name: sorted((predictors_dir / name).glob("*.tif")) for name in used_predictors
    }

    target_paths = {}
    target_bands = {}
    for name, target_cfg in used_targets.items():
        tif_dir = targets_dir / name
        traits = target_cfg.traits
        if traits:
            target_paths[name] = sorted(tif_dir / f"X{t}.tif" for t in traits)
        else:
            target_paths[name] = sorted(tif_dir.glob("*.tif"))
        if target_cfg.get("bands"):
            target_bands[name] = list(target_cfg.bands)

    output_dir = zarr_dir / f"patch{patch_size}_stride{stride}"

    console.print("[bold]Paths:[/bold]")
    console.print(f"Predictors dir:    [cyan]{predictors_dir}[/cyan]")
    console.print(f"Targets dir:       [cyan]{targets_dir}[/cyan]")
    console.print(f"Splits file:       [cyan]{splits_file}[/cyan]")
    console.print(f"Output dir:        [cyan]{output_dir}[/cyan]")
    console.rule()

    chip_rasters_to_zarr(
        predictors=predictor_paths,
        targets=target_paths,
        output_dir=output_dir,
        patch_size=patch_size,
        stride_per_split=stride_per_split,
        h3_file=splits_file,
        save_all=cfg.settings.get("save_all", False),
        overwrite=cfg.settings.get("overwrite", False),
        target_bands=target_bands or None,
        backend="hdf5",
    )

    gpkg_dir = output_dir / "gpkg"
    gpkg_dir.mkdir(parents=True, exist_ok=True)
    console.print("\n[bold]Exporting chip bounds to GeoPackage...[/bold]")
    for h5_path in sorted(output_dir.glob("*.h5")):
        split = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            bounds = f["bounds"][:]
        gdf = gpd.GeoDataFrame(
            {"chip_id": range(len(bounds))},
            geometry=[
                box(min_x, min_y, max_x, max_y) for min_x, min_y, max_x, max_y in bounds
            ],
            crs="EPSG:6933",
        )
        out_path = gpkg_dir / f"{split}.gpkg"
        gdf.to_file(out_path, driver="GPKG")
        console.print(f"  [cyan]{out_path}[/cyan] ({len(gdf):,} chips)")


if __name__ == "__main__":
    main()
