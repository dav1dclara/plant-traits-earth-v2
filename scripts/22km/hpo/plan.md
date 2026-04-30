# Basic HPO Plan (current scope)

## Goal

- Keep setup minimal for now.
- Run 5 baseline seeds manually.
- Then run a basic Bayesian HPO on the existing training pipeline.

## 1) Baseline (manual)

- Use current default training config.
- Run seeds: `0 1 2 3 4`.
- Keep `train.epochs=50` and early stopping enabled.

Example commands:

```bash
python scripts/22km/train.py train.seed=0 train.epochs=50 train.early_stopping.enabled=true
python scripts/22km/train.py train.seed=1 train.epochs=50 train.early_stopping.enabled=true
python scripts/22km/train.py train.seed=2 train.epochs=50 train.early_stopping.enabled=true
python scripts/22km/train.py train.seed=3 train.epochs=50 train.early_stopping.enabled=true
python scripts/22km/train.py train.seed=4 train.epochs=50 train.early_stopping.enabled=true
```

## 2) Basic HPO

- Sweep config: `config/hpo/basic_sweep.yaml`.
- Training entry point: `scripts/22km/train.py`.
- Important integration detail:
  - Sweep uses `${args_no_hyphens}` so Hydra overrides are passed as `key=value`.

Start sweep:

```bash
wandb sweep config/hpo/basic_sweep.yaml
```

Then start agent (replace sweep id):

```bash
wandb agent plant-traits-v2/Project_PTV2/<sweep-id>
```

## 3) After HPO (later)

- Select top 3 HPO configs.
- Re-run each with seeds `0..4`.
- Compare mean and std against baseline mean and std.

This step is intentionally postponed.
