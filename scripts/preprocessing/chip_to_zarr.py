"""Script to chip predictor and target rasters into a single zarr store."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from ptev2.data.chipping import chip_to_zarr


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    zarr_dir = Path(
        "/scratch3/plant-traits-v2/data/chips/22km/"
    )  # TODO: define in config later

    print("--- CHIPPING DATA ---")
    chipping_cfg = cfg.chipping
    patch_size = chipping_cfg.patch_size
    stride = chipping_cfg.stride
    used_targets = [
        name for name, enabled in chipping_cfg.data.targets.items() if enabled
    ]
    used_predictors = [
        name for name, enabled in chipping_cfg.data.predictors.items() if enabled
    ]

    data_cfg = cfg.data
    data_root_dir = Path(data_cfg.root_dir)
    predictor_paths = {
        name: data_root_dir / data_cfg[name].path for name in used_predictors
    }
    target_paths = {name: data_root_dir / data_cfg[name].path for name in used_targets}

    print(f"Patch size: {patch_size} px")
    print(f"Stride:     {stride} px")
    print("Predictors:")
    for name, path in predictor_paths.items():
        print(f"  - {name}: '{path}'")
    print("Targets:")
    for name, path in target_paths.items():
        print(f"  - {name}: '{path}'")

    print()

    # TODO: add split support — currently chips all data into a single store
    output_path = zarr_dir / f"patch{patch_size}_stride{stride}" / "train.zarr"

    chip_to_zarr(
        predictors=predictor_paths,
        targets=target_paths,
        output_path=output_path,
        patch_size=patch_size,
        stride=stride,
    )


if __name__ == "__main__":
    main()
