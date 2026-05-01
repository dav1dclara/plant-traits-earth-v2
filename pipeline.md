# 22km Pipeline (Splits -> Chips -> Train -> Test)

This is the current end-to-end pipeline for 22km using:
- separate `splot` + `gbif` targets,
- masking-aware chipping (`any_overlap`),
- source-aware training (`splot` primary, `gbif` auxiliary),
- dense masked supervision on `patch7_stride3` chips,
- models: `STL` baseline (configured under `training/respatch`), `mtl`, `mmoe`.

## 0) Environment

```bash
cd /scratch3/plant-traits-v3/ldominiak/plant-traits-earth-v3
conda activate PTEV3
```

## 1) Create Splits

Default config already uses:
- sources: `[splot, gbif]`
- JSD optimization on `splot` only
- H3 resolution 1

```bash
python scripts/22km/splitting/create_splits.py
```

Output split file (default):
`/scratch3/plant-traits-v3/data/22km/splits/h3_splits_res1_comb_22km.gpkg`
(`comb` in this filename is legacy split naming, not a merged training label tensor)

## 2) Create Chips

Default config writes to:
`/scratch3/plant-traits-v3/data/22km/chips/patch7_stride3`

```bash
python scripts/22km/chipping/chip_rasters.py
```

Important defaults:
- `split_assignment: any_overlap`
- pixel masking by split (prevents leakage)
- targets: `splot` + `gbif` (no merged `comb` target tensor)

## 3) Training

### 3.1 Method defaults (22 km benchmark)

Default `config/22km/training/default.yaml` now uses:
- `data.targets.mode: source_weighted`
- `train.supervision.mode: dense`
- `train.predictor_validity.mode: min_fraction`
- `train.predictor_validity.min_finite_ratio: 0.2`
- `train.epochs: 15`
- `train.early_stopping.patience: 5`
- `train.source_weights.splot: 1.0`
- `train.source_weights.gbif: 0.1`
- `train.sampling.splot_balanced: true`
- `train.sampling.positive_dataset: splot`
- `train.sampling.positive_fraction: 0.5`
- `train.eval_dataset: splot`
- scheduler: `StepLR(step_size=5, gamma=0.7)`
- model selection metric: `val_splot_macro_pearson` (`mode=max`)
- STL baseline width: `models.base_channels: 32` (`config/22km/models/respatch.yaml`)

Notes:
- In `dense`, loss/metrics use all valid labeled pixels in each 7x7 chip.
- Overlapping chips can duplicate target pixels in per-chip dense evaluation.
- `center_pixel` / `center_crop` remain available for ablations.
- Naming: `training/respatch` is the historical config name, but it runs `ptev2.models.multitask.STLModel`.

### 3.2 Single runs

```bash
# STL baseline (config alias: training/respatch)
python scripts/22km/train.py --config-name training/respatch train.run_name=stl_v3_seed0 train.seed=0

# MTL
python scripts/22km/train.py --config-name training/mtl train.run_name=mtl_v3_seed0 train.seed=0

# MMoE
python scripts/22km/train.py --config-name training/mmoe train.run_name=mmoe_v3_seed0 train.seed=0
```

### 3.3 Multiple seeds (recommended)

```bash
for seed in 0 1 2; do
  python scripts/22km/train.py --config-name training/respatch train.run_name=stl_v3_seed${seed} train.seed=${seed}
  python scripts/22km/train.py --config-name training/mtl      train.run_name=mtl_v3_seed${seed}      train.seed=${seed}
  python scripts/22km/train.py --config-name training/mmoe     train.run_name=mmoe_v3_seed${seed}     train.seed=${seed}
done
```

### 3.4 Optional loader override on busy machine

```bash
python scripts/22km/train.py --config-name training/respatch \
  train.run_name=stl_v3_seed0_nw2 train.seed=0 \
  data_loaders.num_workers=2
```

### 3.5 Center-pixel ablation (optional)

```bash
python scripts/22km/train.py --config-name training/respatch \
  train.run_name=stl_v3_centerpixel_seed0 train.seed=0 \
  train.supervision.mode=center_pixel
```

This computes loss/metrics only on the center pixel.

## 4) Testing

Testing can be called by `run_name` (checkpoint inferred from `checkpoint_dir`) or by explicit `checkpoint_path`.
`test.py` reads supervision and predictor-validity directly from the checkpoint config.

### 4.1 Test by run name

```bash
python scripts/22km/test.py run_name=stl_v3_seed0
python scripts/22km/test.py run_name=mtl_v3_seed0
python scripts/22km/test.py run_name=mmoe_v3_seed0
```

### 4.2 Test multiple seeds

```bash
for seed in 0 1 2; do
  python scripts/22km/test.py run_name=stl_v3_seed${seed}
  python scripts/22km/test.py run_name=mtl_v3_seed${seed}
  python scripts/22km/test.py run_name=mmoe_v3_seed${seed}
done
```

### 4.3 Explicit checkpoint path (alternative)

```bash
python scripts/22km/test.py checkpoint_path=/scratch3/plant-traits-v3/checkpoints/stl_v3_seed0.pth
```

Test outputs now include:
- `n_valid_splot`
- `n_valid_gbif`
- per-trait source counts
- supervision and predictor-validity settings used

## 5) W&B Notes

Default training config logs to:
- `wandb.entity=plant-traits-v3`
- `wandb.project=Project_PTV3`

Key logged metrics:
- `train/splot_loss`, `train/gbif_loss`
- `val/splot_loss`, `val/gbif_loss`
- `val_splot_macro_pearson`, `val_splot_macro_rmse`, `val_splot_macro_nrmse`
- `val_gbif_macro_pearson`
- `val/per_trait_r.X...` (per-trait Pearson for sPlot)
- checkpoint updates:
  - console message when best checkpoint is overwritten
  - W&B keys: `checkpoint/best_updated`, `checkpoint/best_epoch`, `checkpoint/best_selection_value`, `checkpoint/best_path`
