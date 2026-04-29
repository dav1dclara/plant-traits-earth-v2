"""Hydra-configured EO preprocessing script for multi-resolution raster datasets.

Applies Yeo-Johnson transform + z-score standardization to each EO predictor raster
and saves the transform parameters to a CSV for later use (e.g. inverse transform).

Usage (run from PTEV2/):
    python scripts/0_preprocessing/1km/predictors/eo_transform.py

    # Dry run (no files written):
    python scripts/0_preprocessing/1km/predictors/eo_transform.py settings.dry_run=true

    # Override paths:
    python scripts/0_preprocessing/1km/predictors/eo_transform.py \
        paths.data_root=/scratch3/plant-traits-v2/data \
        paths.resolution=1km
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Make preprocess_common importable from the parent (1km/) directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from preprocess_common import (
    dist_stats,
    ensure_dir,
    fit_yj_transform,
    load_as_physical,
    save_tif,
)

warnings.filterwarnings("ignore")


def _process_one_file(
    filepath: Path,
    out_dir: Path,
    variable_name: str,
    settings: DictConfig,
    extra_nodata=None,
    manual_scale: float | None = None,
    manual_offset: float = 0.0,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> dict:
    physical, profile, scale_used, offset_used = load_as_physical(
        filepath,
        extra_nodata=extra_nodata,
        manual_scale=manual_scale,
        manual_offset=manual_offset,
        valid_min=valid_min,
        valid_max=valid_max,
    )
    physical_stats = dist_stats(physical)

    transformed, pt = fit_yj_transform(
        physical,
        max_fit_pixels=settings.max_fit_pixels,
        transform_chunk_size=settings.transform_chunk_size,
        random_seed=settings.random_seed,
    )
    transformed_stats = dist_stats(transformed)

    out_path = out_dir / filepath.name
    written = False
    if not settings.dry_run:
        written = save_tif(
            transformed,
            profile,
            out_path,
            overwrite=settings.overwrite,
        )

    return {
        "variable": variable_name,
        "filename": filepath.name,
        "source_folder": out_dir.name,
        "scale_applied": scale_used,
        "offset_applied": offset_used,
        "yeo_johnson_lambda": float(pt.lambdas_[0]),
        "standardize_mean": float(pt._scaler.mean_[0]),
        "standardize_scale": float(pt._scaler.scale_[0]),
        "phys_min": physical_stats["min"],
        "phys_median": physical_stats["median"],
        "phys_mean": physical_stats["mean"],
        "phys_max": physical_stats["max"],
        "phys_std": physical_stats["std"],
        "trans_min": transformed_stats["min"],
        "trans_median": transformed_stats["median"],
        "trans_mean": transformed_stats["mean"],
        "trans_max": transformed_stats["max"],
        "trans_std": transformed_stats["std"],
        "n_valid": physical_stats["n"],
        "output_path": str(out_path),
        "written": bool(written),
    }


def _iter_files(folder: Path, sample_limit: int | None) -> list[Path]:
    files = sorted(folder.glob("*.tif"))
    if sample_limit is not None:
        files = files[: int(sample_limit)]
    return files


def _process_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    cfg: DictConfig,
) -> list[dict]:
    files = _iter_files(input_dir, cfg.settings.sample_limit)
    if not files:
        print(f"Skipping {dataset_name}: no tif files found in {input_dir}")
        return []

    print(f"Processing {dataset_name}: {len(files)} file(s)")
    records: list[dict] = []

    for filepath in files:
        lower_name = filepath.name.lower()

        if dataset_name == "modis" and "ndvi" in lower_name:
            record = _process_one_file(
                filepath=filepath,
                out_dir=output_dir,
                variable_name=filepath.stem,
                settings=cfg.settings,
                extra_nodata=[-32768],
                manual_scale=cfg.dataset_rules.modis.ndvi_scale,
                manual_offset=cfg.dataset_rules.modis.ndvi_offset,
                valid_min=cfg.dataset_rules.modis.ndvi_valid_min,
                valid_max=cfg.dataset_rules.modis.ndvi_valid_max,
            )
        elif dataset_name == "vodca":
            record = _process_one_file(
                filepath=filepath,
                out_dir=output_dir,
                variable_name=filepath.stem,
                settings=cfg.settings,
                extra_nodata=[-32768],
                valid_min=cfg.dataset_rules.vodca.valid_min,
                valid_max=cfg.dataset_rules.vodca.valid_max,
            )
        elif dataset_name == "worldclim":
            record = _process_one_file(
                filepath=filepath,
                out_dir=output_dir,
                variable_name=filepath.stem,
                settings=cfg.settings,
                extra_nodata=[-32768],
            )
        else:
            record = _process_one_file(
                filepath=filepath,
                out_dir=output_dir,
                variable_name=filepath.stem,
                settings=cfg.settings,
            )

        print(
            f"  OK {record['filename']} | lambda={record['yeo_johnson_lambda']:.4f} "
            f"| n_valid={record['n_valid']:,}"
        )
        records.append(record)

    return records


@hydra.main(
    version_base=None,
    config_path="../../../../config/preprocessing",
    config_name="eo_transform",
)
def main(cfg: DictConfig) -> None:
    print("Starting EO preprocessing")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    input_root = Path(cfg.paths.eo_input_dir)
    output_root = Path(cfg.paths.eo_output_dir)
    params_out = Path(cfg.paths.params_out)
    ensure_dir(output_root)
    ensure_dir(params_out.parent)

    all_records: list[dict] = []

    for dataset_name, dataset_cfg in cfg.datasets.items():
        if not dataset_cfg.enabled:
            continue

        input_dir = input_root / dataset_cfg.input_subdir
        output_dir = output_root / dataset_cfg.output_subdir
        ensure_dir(output_dir)

        if not input_dir.exists():
            print(f"Skipping {dataset_name}: input directory not found -> {input_dir}")
            continue

        records = _process_dataset(dataset_name, input_dir, output_dir, cfg)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No raster files were processed. Check the configured paths."
        )

    df = pd.DataFrame(all_records)
    if not cfg.settings.dry_run:
        df.to_csv(params_out, index=False)
        summary_out = params_out.with_name(params_out.stem + "_summary.json")
        summary = {
            "n_files": int(len(df)),
            "datasets": {
                str(name): int(count)
                for name, count in df.groupby("source_folder").size().items()
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        with open(summary_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved parameter table to {params_out}")
        print(f"Saved summary to {summary_out}")
    else:
        print("Dry run enabled: no output rasters or CSV were written")

    print(df[["source_folder", "filename", "yeo_johnson_lambda", "n_valid"]].head())
    print(f"Completed EO preprocessing for {len(df)} file(s)")


if __name__ == "__main__":
    main()
