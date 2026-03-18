"""Script to chip predictor and target rasters into per-split zarr stores."""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.pretty import Pretty

from ptev2.data.chipping import chip_to_zarr

console = Console()


@hydra.main(
    version_base=None, config_path="../../config/chipping", config_name="default"
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold]CHIPPING DATA TO ZARR[/bold]")

    # configuration
    patch_size = cfg.patch_size
    stride = cfg.stride
    h3_splits_dir = Path(cfg.splits.h3_dir)
    h3_file = (
        h3_splits_dir / "h3_res1_X1080_mean.gpkg"
    )  # TODO: right now this depends on the trait, make this more flexible
    zarr_dir = Path(cfg.zarr_dir)
    output_dir = zarr_dir / f"patch{patch_size}_stride{stride}"
    data_root_dir = Path(cfg.data.root_dir)
    predictors_dir = Path(cfg.data.predictors_dir)

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
        tif_dir = data_root_dir / name
        traits = target_cfg.traits
        if traits:
            target_paths[name] = sorted(tif_dir / f"X{t}.tif" for t in traits)
        else:
            target_paths[name] = sorted(tif_dir.glob("*.tif"))

    # print configuration summary
    console.print(f"Patch size:       [cyan]{patch_size}[/cyan] px")
    console.print(f"Stride:           [cyan]{stride}[/cyan] px")
    console.print(f"H3 splits file:    [cyan]{h3_file}[/cyan]")
    console.print(f"Zarr output dir:  [cyan]{output_dir}[/cyan]")

    console.print("Predictors:")
    for name, pred_cfg in used_predictors.items():
        bands = list(pred_cfg.bands) if pred_cfg.bands else "all"
        console.print(
            f"  [green]+[/green] {name} (bands: {bands}, {len(predictor_paths[name])} file(s))"
        )
        for path in predictor_paths[name]:
            console.print(f"      [dim]{path}[/dim]")
    console.print("Targets:")
    for name, target_cfg in used_targets.items():
        traits = list(target_cfg.traits) if target_cfg.traits else "all"
        console.print(
            f"  [green]+[/green] {name} (traits: {traits}, {len(target_paths[name])} file(s))"
        )
        for path in target_paths[name]:
            console.print(f"      [dim]{path}[/dim]")

    console.print()

    chip_to_zarr(
        predictors=predictor_paths,
        targets=target_paths,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        h3_file=h3_file,
    )


if __name__ == "__main__":
    main()
