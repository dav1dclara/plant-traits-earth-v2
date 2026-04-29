"""Hydra-configured reverse transformation for GBIF and sPlot trait rasters.

Applies the inverse Yeo-Johnson + z-score transform to model-predicted or
raw trait rasters to convert them back to physical units.

Requires the parameter CSV produced by eo_transform.py (or a separately
fitted transform with matching columns: trait, yeo_johnson_lambda,
standardize_mean, standardize_scale).

Usage (run from PTEV2/):
    python scripts/0_preprocessing/1km/targets/reverse_transform_traits.py

    # Dry run (no files written):
    python scripts/0_preprocessing/1km/targets/reverse_transform_traits.py \
        settings.dry_run=true

    # Override transform CSV path:
    python scripts/0_preprocessing/1km/targets/reverse_transform_traits.py \
        paths.transform_csv=/path/to/power_transformer_params.csv
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Make preprocess_common importable from the parent (1km/) directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from preprocess_common import (
    back_transform_values,
    dist_stats,
    ensure_dir,
    extract_trait_id,
    load_params_table,
    mask_nodata,
    read_raster,
    save_tif,
)

warnings.filterwarnings("ignore")


def _process_one_trait(
    filepath: Path,
    trait_id: str,
    params_df: pd.DataFrame,
    out_dir: Path,
    output_suffix: str,
    overwrite: bool,
    dry_run: bool,
    source_name: str,
) -> dict:
    row = params_df.loc[trait_id]
    data, profile, nodata = read_raster(filepath)
    masked = mask_nodata(data, nodata)

    output = np.full(masked.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(masked)
    if valid_mask.any():
        output[valid_mask] = back_transform_values(
            masked[valid_mask],
            row["yeo_johnson_lambda"],
            row["standardize_mean"],
            row["standardize_scale"],
        ).astype(np.float32)

    profile.update(dtype="float32", nodata=np.nan, count=1)
    out_path = out_dir / f"{filepath.stem}{output_suffix}.tif"
    written = False
    if not dry_run:
        written = save_tif(output, profile, out_path, overwrite=overwrite)

    stats = dist_stats(output)
    return {
        "source": source_name,
        "trait": trait_id,
        "file_in": filepath.name,
        "file_out": out_path.name,
        "n_valid": stats["n"],
        "min": stats["min"],
        "mean": stats["mean"],
        "max": stats["max"],
        "std": stats["std"],
        "written": bool(written),
    }


def _run_source(
    source_name: str,
    input_dir: Path,
    output_dir: Path,
    cfg: DictConfig,
    params_df: pd.DataFrame,
) -> list[dict]:
    if not input_dir.exists():
        print(f"Skipping {source_name}: input directory not found -> {input_dir}")
        return []

    files = sorted(input_dir.glob(cfg.settings.glob_pattern))
    if cfg.settings.sample_limit is not None:
        files = files[: int(cfg.settings.sample_limit)]

    if not files:
        print(f"Skipping {source_name}: no matching tif files found")
        return []

    ensure_dir(output_dir)
    print(f"Processing {source_name}: {len(files)} file(s)")

    valid_traits = params_df.index.tolist()
    records: list[dict] = []

    for filepath in files:
        trait_id = extract_trait_id(filepath, valid_traits)
        if trait_id is None:
            print(f"  Skipping {filepath.name}: no matching trait id found")
            continue

        record = _process_one_trait(
            filepath=filepath,
            trait_id=trait_id,
            params_df=params_df,
            out_dir=output_dir,
            output_suffix=cfg.settings.output_suffix,
            overwrite=cfg.settings.overwrite,
            dry_run=cfg.settings.dry_run,
            source_name=source_name,
        )
        print(
            f"  OK {source_name}:{record['trait']} | n_valid={record['n_valid']:,} "
            f"| mean={record['mean']:.4f}"
        )
        records.append(record)

    return records


@hydra.main(
    version_base=None,
    config_path="../../../../config/preprocessing",
    config_name="reverse_transform",
)
def main(cfg: DictConfig) -> None:
    print("Starting reverse transform for trait rasters")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    params_df = load_params_table(Path(cfg.paths.transform_csv))
    print(f"Loaded parameters for {len(params_df)} trait(s)")

    all_records: list[dict] = []

    if cfg.sources.gbif.enabled:
        all_records.extend(
            _run_source(
                source_name="gbif",
                input_dir=Path(cfg.paths.gbif_input_dir),
                output_dir=Path(cfg.paths.gbif_output_dir),
                cfg=cfg,
                params_df=params_df,
            )
        )

    if cfg.sources.splot.enabled:
        all_records.extend(
            _run_source(
                source_name="splot",
                input_dir=Path(cfg.paths.splot_input_dir),
                output_dir=Path(cfg.paths.splot_output_dir),
                cfg=cfg,
                params_df=params_df,
            )
        )

    if not all_records:
        raise RuntimeError(
            "No trait rasters were processed. Check the configured paths."
        )

    results_df = pd.DataFrame(all_records)
    summary_csv = Path(cfg.paths.summary_dir) / "reverse_transform_summary.csv"
    ensure_dir(summary_csv.parent)

    if not cfg.settings.dry_run:
        results_df.to_csv(summary_csv, index=False)
        summary_json = summary_csv.with_suffix(".json")
        payload = {
            "n_outputs": int(len(results_df)),
            "sources": {
                str(name): int(count)
                for name, count in results_df.groupby("source").size().items()
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        with open(summary_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved summary CSV to {summary_csv}")
        print(f"Saved summary JSON to {summary_json}")
    else:
        print("Dry run enabled: no output rasters or summaries were written")

    print(results_df.head())
    print(f"Completed reverse transform for {len(results_df)} file(s)")


if __name__ == "__main__":
    main()
