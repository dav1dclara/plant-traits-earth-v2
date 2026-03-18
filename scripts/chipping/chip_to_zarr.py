"""Script to chip predictor and target rasters into per-split zarr stores."""

from pathlib import Path

import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.pretty import Pretty

from ptev2.data.chipping import chip_to_zarr

console = Console()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    console.rule("[bold]CHIPPING DATA TO ZARR[/bold]")
    # chipping configuration
    chipping_cfg = cfg.chipping

    zarr_dir = Path(chipping_cfg.zarr_dir)
    patch_size = chipping_cfg.patch_size
    stride = chipping_cfg.stride

    # get predictors and targets to use
    used_predictors = {
        name: pred_cfg
        for name, pred_cfg in chipping_cfg.data.predictors.items()
        if pred_cfg.use
    }
    used_targets = {
        name: target_cfg
        for name, target_cfg in chipping_cfg.data.targets.items()
        if target_cfg.use
    }

    # data configuration
    data_cfg = cfg.data
    data_root_dir = Path(data_cfg.root_dir)

    predictor_paths = {
        name: sorted((data_root_dir / data_cfg[name].path).glob("*.tif"))
        for name in used_predictors
    }

    target_paths = {}
    for name, target_cfg in used_targets.items():
        tif_dir = data_root_dir / data_cfg[name].path
        traits = target_cfg.traits
        if traits:
            target_paths[name] = sorted(tif_dir / f"X{t}.tif" for t in traits)
        else:
            target_paths[name] = sorted(tif_dir.glob("*.tif"))

    console.print(f"Zarr dir:     [cyan]{zarr_dir}[/cyan]")
    console.print(f"Patch size:   [cyan]{patch_size}[/cyan] px")
    console.print(f"Stride:       [cyan]{stride}[/cyan] px")
    console.print("Predictors:")
    for name, pred_cfg in used_predictors.items():
        bands = list(pred_cfg.bands) if pred_cfg.bands else "all"
        console.print(
            f"  [green]+[/green] {name} (bands: {bands}, {len(predictor_paths[name])} file(s))"
        )
        for path in predictor_paths[name]:
            console.print(f"      [dim]{path.name}[/dim]")
    console.print("Targets:")
    for name, target_cfg in used_targets.items():
        traits = list(target_cfg.traits) if target_cfg.traits else "all"
        console.print(
            f"  [green]+[/green] {name} (traits: {traits}, {len(target_paths[name])} file(s))"
        )
        for path in target_paths[name]:
            console.print(f"      [dim]{path.name}[/dim]")

    print()

    # TODO: rewrite with our new splits
    h3_splits_dir = Path("/scratch3/plant-traits-v2/data/temp/h3_splits")
    h3_file = h3_splits_dir / "h3_res1_X1080_mean.gpkg"
    console.print(f"Loading H3 split cells from [cyan]{h3_file.name}[/cyan]...")
    h3_gdf = gpd.read_file(h3_file)

    output_dir = zarr_dir / f"patch{patch_size}_stride{stride}"

    chip_to_zarr(
        predictors=predictor_paths,
        targets=target_paths,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        h3_gdf=h3_gdf,
    )


if __name__ == "__main__":
    main()
