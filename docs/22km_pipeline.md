# 22 km Pipeline Guide

This document describes the 22 km PlantTraits V2 pipeline end to end: how to create splits, chip the rasters, train the four model variants, and run inference.

## Overview

The 22 km benchmark uses patch-based inputs and dense model outputs.

- Input chips are shaped `(B, C, H, W)`.
- Models return dense predictions shaped `(B, out_channels, H, W)`.
- The default 22 km chip geometry is `patch_size=7` with `stride=3`.
- The default chip directory is `data/22km/chips/patch7_stride3/`.

The pipeline is source-aware:

- sPlot is the primary supervision source.
- GBIF is auxiliary supervision.
- Dense supervision is supported in the training code, but the 22 km setup uses the existing source-weighted configuration.

## Model Variants

The 22 km pipeline currently supports four model variants:

1. `PixelMLP`
   - Tiny per-pixel baseline.
   - Implemented with stacked 1x1 convolutions.
   - Config: `config/22km/models/mlp.yaml`

2. `STLModel`
   - Single-task ResPatch-style baseline.
   - This is the canonical baseline for the old `respatch` naming.
   - Config: `config/22km/models/stl.yaml`

3. `MTLModel`
   - Shared-encoder multi-task model.
   - Config: `config/22km/models/mtl.yaml`

4. `GatedMMoEModelV3`
   - Dense mixture-of-experts model.
   - Config: `config/22km/models/mmoe.yaml`

## Data Splits

The split generation step produces train/val/test assignments at 22 km resolution.

Split semantics:

- sPlot cells are assigned to train, val, or test.
- GBIF-only cells are assigned to train or val only, never test.
- Cells with valid predictors but no labels are assigned to `none`.
- Cells with no data at all are discarded.

The default split settings are defined in `config/22km/preprocessing/splitting.yaml`.

## Chips

After splits are created, raster chips are generated into per-split zarr stores.

Default chip settings:

- `patch_size: 7`
- `stride: 3`
- `save_all: true`
- `overwrite: true`

The chip configuration lives in `config/22km/preprocessing/chipping.yaml`.

The generated output is written under:

- `data/22km/chips/patch7_stride3/train.zarr`
- `data/22km/chips/patch7_stride3/val.zarr`
- `data/22km/chips/patch7_stride3/test.zarr`
- `data/22km/chips/patch7_stride3/all.zarr` when `save_all: true`

GeoPackage bounds are also exported under the `gpkg/` subfolder inside the chip directory.

## Step 1: Create Splits

Run the split builder first:

```bash
python scripts/22km/splitting/create_splits.py
```

This reads the 22 km target rasters and writes the H3 split file to:

- `data/22km/splits/h3_splits_res1_22km.gpkg`

## Step 2: Chip the Rasters

After the split file exists, chip predictors and targets into zarr stores:

```bash
python scripts/22km/chipping/chip_rasters.py
```

This uses the default 22 km preprocessing config and writes chips to:

- `data/22km/chips/patch7_stride3/`

## Step 3: Train a Model

The training entry point is:

```bash
python scripts/22km/train.py
```

By default this uses the STL baseline.

### Train the STL baseline

```bash
python scripts/22km/train.py
```

### Train the PixelMLP baseline

```bash
python scripts/22km/train.py models=mlp train.run_name=mlp
```

### Train the MTL model

```bash
python scripts/22km/train.py models=mtl train.run_name=mtl
```

### Train the MMoE model

```bash
python scripts/22km/train.py models=mmoe train.run_name=mmoe
```

Training uses the config under `config/22km/training/default.yaml`, which already points to the canonical STL baseline. The model selection is overridden with Hydra when running the other variants.

Typical checkpoint outputs are written to:

- `checkpoints/<run_name>.pth`

## Multi-Seed Runs

For a fair comparison, it is a good idea to run each model with multiple random
seeds and report the mean and standard deviation across runs.

All 22 km training runs log to the same Weights & Biases project configured in
`config/22km/training/default.yaml`:

- entity: `plant-traits-v2`
- project: `Project_PTV2`

If you want to group multiple seeds in W&B, use `train.group`.

### Example: three seeds per model

```bash
for seed in 0 1 2; do
   python scripts/22km/train.py \
      models=stl \
      train.seed=$seed \
      train.run_name=stl_s${seed} \
      train.group=stl-seeds
done
```

You can repeat the same pattern for the other variants:

```bash
for seed in 0 1 2; do
   python scripts/22km/train.py \
      models=mlp \
      train.seed=$seed \
      train.run_name=mlp_s${seed} \
      train.group=mlp-seeds
done

for seed in 0 1 2; do
   python scripts/22km/train.py \
      models=mtl \
      train.seed=$seed \
      train.run_name=mtl_s${seed} \
      train.group=mtl-seeds
done

for seed in 0 1 2; do
   python scripts/22km/train.py \
      models=mmoe \
      train.seed=$seed \
      train.run_name=mmoe_s${seed} \
      train.group=mmoe-seeds
done
```

### Inference for a seed run

After training, evaluate each checkpoint separately:

```bash
python scripts/22km/test.py \
   checkpoint_path=/scratch3/plant-traits-v2/checkpoints/stl_s0.pth \
   write_all_map=false
```

If you want the merged prediction map, enable `write_all_map`:

```bash
python scripts/22km/test.py \
   checkpoint_path=/scratch3/plant-traits-v2/checkpoints/stl_s0.pth \
   write_all_map=true \
   prediction_tif_name=stl_s0.tif
```

## Step 4: Run Inference and Evaluation

The test entry point evaluates a checkpoint on the requested split and can optionally export a full prediction GeoTIFF.

### Metrics only

```bash
python scripts/22km/test.py checkpoint_path=/scratch3/plant-traits-v2/checkpoints/stl.pth write_all_map=false
```

### Metrics plus full map export

```bash
python scripts/22km/test.py checkpoint_path=/scratch3/plant-traits-v2/checkpoints/stl.pth write_all_map=true prediction_tif_name=stl.tif
```

The test script reads the checkpoint's embedded training config to keep the preprocessing, supervision, and target layout consistent.

Default output locations:

- Metrics JSON: `test_outputs/`
- Per-trait CSV diagnostics: `outputs/diagnostics/`
- Full-map GeoTIFF: `predictions/`

If `write_all_map` is enabled, the script writes the merged prediction raster to:

- `predictions/<prediction_tif_name>`

## Recommended Run Order

1. Create splits.
2. Chip rasters.
3. Train one of the four model variants.
4. Run inference on a checkpoint.

## Notes

- The 22 km pipeline expects the chip geometry to match the model supervision mode.
- The STL baseline is the canonical replacement for the old `respatch` naming.
- Legacy imports are still available for old checkpoints, but new runs should use the dedicated model modules.
