# __temp_dev/MTL

This folder contains a fresh, self-contained MTL/MTMoE pipeline built from the STL backbone. The goal is to provide clean training, evaluation, and inspection code that is easy to read and modify.

## What is included

- `dataloader.py` — self-contained Zarr dataset loader for `predictors` and `targets`.
- `models.py` — backbone, STL/MTL heads, and MMoE models.
- `loss.py` — weighted dense regression loss for spatial outputs.
- `utils.py` — helper functions for target layout, JSON saving, normalization, and split logic.
- `train.py` — training script for STL / MTL / MMoE models.
- `evaluate.py` — test-time evaluation script that loads checkpoints.
- `heatmaps.py` — simple result visualization for per-trait scores.
- `inspect_zarr.py` — inspect Zarr structure and export sample arrays / summaries.

## Recommended workflow

1. Inspect the data:
   ```bash
   python scripts/__temp_dev/MTL/inspect_zarr.py \
       --zarr-path /path/to/train.zarr \
       --export-summary \
       --export-dir scripts/__temp_dev/MTL/results/zarr_inspect
   ```

2. Train a model:
   ```bash
   python scripts/__temp_dev/MTL/train.py \
       --model mtl \
       --zarr-dir /scratch3/plant-traits-v2/data/22km/chips/patch15_stride10 \
       --predictors canopy_height modis worldclim soil_grids vodca \
       --target comb \
       --epochs 20 \
       --batch-size 32
   ```

3. Evaluate the best checkpoint:
   ```bash
   python scripts/__temp_dev/MTL/evaluate.py \
       --checkpoint-dir scripts/__temp_dev/MTL/checkpoints \
       --zarr-dir /scratch3/plant-traits-v2/data/22km/chips/patch15_stride10 \
       --predictors canopy_height modis worldclim soil_grids vodca \
       --target comb
   ```

4. Plot results:
   ```bash
   python scripts/__temp_dev/MTL/heatmaps.py \
       --metrics-path scripts/__temp_dev/MTL/results/best_metrics.json
   ```

## Notes

- The new code is intentionally simple and avoids Hydra complexity.
- Training and evaluation both use the same backbone as the STL model.
- The MMoE model is implemented at the spatial patch level using the same backbone features.
