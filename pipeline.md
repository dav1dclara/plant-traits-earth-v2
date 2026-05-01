# 22km Pipeline (Splits -> Chips -> Train -> Test)

This is the current end-to-end pipeline for 22km using:
- separate `splot` + `gbif` targets,
- masking-aware chipping (`any_overlap`),
- source-aware training (`splot` primary, `gbif` auxiliary),
- supervision-consistent train/val/test (`center_pixel`, `center_crop`, or `dense`),
- models: `respatch` (STL baseline), `mtl`, `mmoe`.

## 0) Environment

```bash
cd /scratch3/plant-traits-v2/ldominiak/plant-traits-earth-v2
conda activate PTEV2
```

## 1) Create Splits

Default config already uses:
- sources: `[splot, gbif]`
- JSD optimization on `splot` only
- H3 resolution 1

```bash
python scripts/22km/1_splitting/create_splits.py
```

Output split file (default):
`/scratch3/plant-traits-v2/data/22km/splits/h3_splits_res1_comb_22km.gpkg`
(`comb` in this filename is legacy split naming, not a merged training label tensor)

## 2) Create Chips

Default config writes to:
`/scratch3/plant-traits-v2/data/22km/chips_luca_new/patch7_stride3`

```bash
python scripts/22km/2_chipping/chip_rasters.py
```

Important defaults:
- `split_assignment: any_overlap`
- pixel masking by split (prevents leakage)
- targets: `splot` + `gbif` (no merged `comb` target tensor)

## 3) Training

### 3.1 Method defaults (22 km benchmark)

Default `config/22km/training/default.yaml` now uses:
- `data.targets.mode: source_weighted`
- `train.supervision.mode: center_pixel`
- `train.supervision.center_crop_size: 1`
- `train.predictor_validity.mode: min_fraction`
- `train.predictor_validity.min_finite_ratio: 0.2`
- `train.source_weights.splot: 1.0`
- `train.source_weights.gbif: 0.1`
- `train.eval_dataset: splot`
- model selection metric must be `val_splot_*`

Notes:
- In `center_pixel`, tensors entering loss/metrics are `1x1` spatially.
- `dense` is ablation-only; overlapping chips can duplicate target pixels.
- `center_crop` is supported for future 1km-scale experiments.

### 3.2 Single runs

```bash
# ResPatch (STL baseline)
python scripts/22km/train.py --config-name training/respatch train.run_name=respatch_v2_seed0 train.seed=0

# MTL
python scripts/22km/train.py --config-name training/mtl train.run_name=mtl_v2_seed0 train.seed=0

# MMoE
python scripts/22km/train.py --config-name training/mmoe train.run_name=mmoe_v2_seed0 train.seed=0
```

### 3.3 Multiple seeds (recommended)

```bash
for seed in 0 1 2; do
  python scripts/22km/train.py --config-name training/respatch train.run_name=respatch_v2_seed${seed} train.seed=${seed}
  python scripts/22km/train.py --config-name training/mtl      train.run_name=mtl_v2_seed${seed}      train.seed=${seed}
  python scripts/22km/train.py --config-name training/mmoe     train.run_name=mmoe_v2_seed${seed}     train.seed=${seed}
done
```

### 3.4 Optional loader override on busy machine

```bash
python scripts/22km/train.py --config-name training/respatch \
  train.run_name=respatch_v2_seed0_nw2 train.seed=0 \
  data_loaders.num_workers=2
```

### 3.5 Center-crop ablation / future 1km-style supervision on current pipeline

```bash
python scripts/22km/train.py --config-name training/respatch \
  train.run_name=respatch_v2_centercrop32_seed0 train.seed=0 \
  train.supervision.mode=center_crop \
  train.supervision.center_crop_size=32
```

This keeps patch-context inputs and computes loss/metrics only on the central crop.

## 4) Testing

Testing can be called by `run_name` (checkpoint inferred from `checkpoint_dir`) or by explicit `checkpoint_path`.
`test.py` reads supervision and predictor-validity directly from the checkpoint config.

### 4.1 Test by run name

```bash
python scripts/22km/test.py run_name=respatch_v2_seed0
python scripts/22km/test.py run_name=mtl_v2_seed0
python scripts/22km/test.py run_name=mmoe_v2_seed0
```

### 4.2 Test multiple seeds

```bash
for seed in 0 1 2; do
  python scripts/22km/test.py run_name=respatch_v2_seed${seed}
  python scripts/22km/test.py run_name=mtl_v2_seed${seed}
  python scripts/22km/test.py run_name=mmoe_v2_seed${seed}
done
```

### 4.3 Explicit checkpoint path (alternative)

```bash
python scripts/22km/test.py checkpoint_path=/scratch3/plant-traits-v2/checkpoints/respatch_v2_seed0.pth
```

Test outputs now include:
- `n_valid_splot`
- `n_valid_gbif`
- per-trait source counts
- supervision and predictor-validity settings used

## 5) W&B Notes

Default training config logs to:
- `wandb.entity=plant-traits-v2`
- `wandb.project=Project_PTV2`

Key logged metrics:
- `train/splot_loss`, `train/gbif_loss`
- `val/splot_loss`, `val/gbif_loss`
- `val/splot/*` (selection is based on `val_splot_*` only)
- optional `val/gbif/*` diagnostics
