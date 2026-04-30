# 22km Pipeline (Splits -> Chips -> Train -> Test)

This is the current end-to-end pipeline for 22km using:
- separate `splot` + `gbif` targets,
- masking-aware chipping (`any_overlap`),
- `WeightedMaskedDenseLoss` (no calibration),
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

## 2) Create Chips

Default config writes to:
`/scratch3/plant-traits-v2/data/22km/chips_luca_new/patch7_stride3`

```bash
python scripts/22km/2_chipping/chip_rasters.py
```

Important defaults:
- `split_assignment: any_overlap`
- pixel masking by split (prevents leakage)
- targets: `splot` + `gbif` (no `comb`)

## 3) Training

### 3.1 Single runs

```bash
# ResPatch (STL baseline)
python scripts/22km/train.py --config-name training/respatch train.run_name=respatch_seed0 train.seed=0

# MTL
python scripts/22km/train.py --config-name training/mtl train.run_name=mtl_seed0 train.seed=0

# MMoE
python scripts/22km/train.py --config-name training/mmoe train.run_name=mmoe_seed0 train.seed=0
```

### 3.2 Multiple seeds (recommended)

```bash
for seed in 0 1 2; do
  python scripts/22km/train.py --config-name training/respatch train.run_name=respatch_seed${seed} train.seed=${seed}
  python scripts/22km/train.py --config-name training/mtl      train.run_name=mtl_seed${seed}      train.seed=${seed}
  python scripts/22km/train.py --config-name training/mmoe     train.run_name=mmoe_seed${seed}     train.seed=${seed}
done
```

### 3.3 Optional loader override on busy machine

If training is slow due to shared CPU/I/O load:

```bash
python scripts/22km/train.py --config-name training/respatch \
  train.run_name=respatch_seed0_nw2 train.seed=0 \
  data_loaders.num_workers=2
```

## 4) Testing

Testing can be called by `run_name` (checkpoint inferred from `checkpoint_dir`) or by explicit `checkpoint_path`.

### 4.1 Test by run name

```bash
python scripts/22km/test.py run_name=respatch_seed0
python scripts/22km/test.py run_name=mtl_seed0
python scripts/22km/test.py run_name=mmoe_seed0
```

### 4.2 Test multiple seeds

```bash
for seed in 0 1 2; do
  python scripts/22km/test.py run_name=respatch_seed${seed}
  python scripts/22km/test.py run_name=mtl_seed${seed}
  python scripts/22km/test.py run_name=mmoe_seed${seed}
done
```

### 4.3 Explicit checkpoint path (alternative)

```bash
python scripts/22km/test.py checkpoint_path=/scratch3/plant-traits-v2/checkpoints/respatch_seed0.pth
```

## 5) W&B Notes

Default training config logs to:
- `wandb.entity=plant-traits-v2`
- `wandb.project=Project_PTV2`

Logged metrics include compatible keys for comparison with older runs:
- `train/splot_loss`, `train/gbif_loss`
- `val/loss`, `val/mean_r`, `val/per_trait_r`
- `val/splot/*` and trait-level `val/traits/<trait>/pearson_r`
