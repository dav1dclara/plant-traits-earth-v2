"""Script to chip predictor and target rasters into per-split zarr stores."""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.pretty import Pretty

from ptev2.data.chipping import chip_rasters_to_zarr

console = Console()


@hydra.main(
    version_base=None, config_path="../../config/chipping", config_name="default"
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold]CHIPPING DATA TO ZARR[/bold]")

    # paths
    predictors_dir = Path(cfg.paths.predictors_dir)
    targets_dir = Path(cfg.paths.targets_dir)
    splits_file = Path(cfg.paths.splits_file)
    zarr_dir = Path(cfg.paths.zarr_dir)

    # chipping settings
    patch_size = cfg.settings.patch_size
    stride = cfg.settings.stride

    # get predictors and targets to use
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
    for name, target_cfg in used_targets.items():
        tif_dir = targets_dir / name
        traits = target_cfg.traits
        if traits:
            target_paths[name] = sorted(tif_dir / f"X{t}.tif" for t in traits)
        else:
            target_paths[name] = sorted(tif_dir.glob("*.tif"))

    # define zarr output dir
    output_dir = zarr_dir / f"patch{patch_size}_stride{stride}"

    # print configuration summary
    console.print("[bold]Paths:[/bold]")
    console.print(f"Predictors dir:    [cyan]{predictors_dir}[/cyan]")
    console.print(f"Targets dir:       [cyan]{targets_dir}[/cyan]")
    console.print(f"Splits file:       [cyan]{splits_file}[/cyan]")
    console.print(f"Zarr dir:          [cyan]{zarr_dir}[/cyan]")
    console.print(f"Output dir:        [cyan]{output_dir}[/cyan]")

    console.rule()

    console.print("[bold]Chipping settings:[/bold]")
    console.print(f"Patch size:        [cyan]{patch_size}[/cyan] px")
    console.print(f"Stride:            [cyan]{stride}[/cyan] px")

    console.rule()

    console.print("[bold]Predictors:[/bold]")
    for name, pred_cfg in used_predictors.items():
        bands = list(pred_cfg.bands) if pred_cfg.bands else "all"
        console.print(
            f"[green]+[/green] {name} (bands: {bands}, {len(predictor_paths[name])} file(s))"
        )
        for path in predictor_paths[name]:
            console.print(f"    [dim]{path}[/dim]")
    console.print("[bold]Targets:[/bold]")
    for name, target_cfg in used_targets.items():
        traits = list(target_cfg.traits) if target_cfg.traits else "all"
        console.print(
            f"[green]+[/green] {name} (traits: {traits}, {len(target_paths[name])} file(s))"
        )
        for path in target_paths[name]:
            console.print(f"    [dim]{path}[/dim]")

    console.rule()

    chip_rasters_to_zarr(
        predictors=predictor_paths,
        targets=target_paths,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        h3_file=splits_file,
    )


if __name__ == "__main__":
    main()
