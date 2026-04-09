# ═══════════════════════════════════════════════════════════════════════════════
# QUICK START GUIDE - STEP BY STEP
# ═══════════════════════════════════════════════════════════════════════════════

This guide shows exactly what to run and in what order.

## STEP 1: INSPECT DATA (Optional but recommended)

```bash
cd /scratch3/plant-traits-v2/dsaini/plant-traits-earth-v2/scripts/__temp_dev/MTL
python inspect_zarr.py
```

Output will show:
- Data shapes
- Number of samples per split
- Sample npy files saved

## STEP 2: TRAIN MTL MODEL

Edit `train_v2.py` line ~30 to choose model:

```python
MODEL_TYPE = "mtl"  # Change to "stl" or "mmoe" to try others
```

Then run:

```bash
python train_v2.py
```

Expected runtime: ~50-60 minutes for 50 epochs on GPU
Expected output:
- Epoch by epoch: train loss, val loss, checkpoint saved
- Final: Best checkpoint path and training complete message
- Files saved to: `results/mtl/checkpoints/` and `results/mtl/metrics/`

## STEP 3: EVALUATE ON TEST SET

Edit `evaluate_v2.py` to match your training config (same MODEL_TYPE, architecture):

```python
MODEL_TYPE = "mtl"  # Must match training
```

Then run:

```bash
python evaluate_v2.py
```

Expected output:
- Pearson r and RMSE metrics
- Files saved to: `results/mtl/eval_results/`

## STEP 4: COMPARE MODELS (Optional)

Train and evaluate all three models:

```bash
# === STL ===
# Edit train_v2.py: MODEL_TYPE = "stl"
python train_v2.py
# Edit evaluate_v2.py: MODEL_TYPE = "stl"
python evaluate_v2.py

# === MTL ===
# Edit train_v2.py: MODEL_TYPE = "mtl"
python train_v2.py
# Edit evaluate_v2.py: MODEL_TYPE = "mtl"
python evaluate_v2.py

# === MMoE ===
# Edit train_v2.py: MODEL_TYPE = "mmoe"
python train_v2.py
# Edit evaluate_v2.py: MODEL_TYPE = "mmoe"
python evaluate_v2.py
```

Then compare results:
- Open `results/stl/eval_results/metrics.json`
- Open `results/mtl/eval_results/metrics.json`
- Open `results/mmoe/eval_results/metrics.json`

Look at "pearson_r_all" to compare models.

## EXPECTED RESULTS

Good results should have:
- **Pearson r > 0.65** (ideally > 0.75)
- **RMSE < 0.50** (ideally < 0.40)

Typical improvement:
- STL baseline: r ≈ 0.70
- MTL: r ≈ 0.72-0.73 (+2-3%)
- MMoE: r ≈ 0.74-0.75 (+4-5%)

## TUNE HYPERPARAMETERS

If results are poor, edit `train_v2.py` config section:

1. **Not converging?** Lower learning rate:
   - Change `LR = 1e-4` to `LR = 5e-5` or `1e-5`

2. **Loss exploding?** Increase gradient clipping:
   - Change `GRAD_CLIP = 5.0` to `GRAD_CLIP = 10.0`

3. **Bad test performance?** Increase sPlot weight:
   - Change `W_SPLOT = 8.0` to `W_SPLOT = 16.0`

4. **Out of memory?** Reduce batch size:
   - Change `BATCH_SIZE = 32` to `BATCH_SIZE = 16`

5. **Stopping too early?** Increase patience:
   - Change `EARLY_STOP_PATIENCE = 15` to `EARLY_STOP_PATIENCE = 20`

## FILE ORGANIZATION

After running all steps:

```
MTL/
├── results/
│   ├── mtl/
│   │   ├── checkpoints/
│   │   │   ├── mtl_best.pth
│   │   │   └── mtl_epoch_*.pth
│   │   ├── metrics/
│   │   │   └── train_history.json
│   │   └── eval_results/
│   │       ├── test_preds.npy
│   │       ├── test_targets.npy
│   │       └── metrics.json     ← Compare "pearson_r_all"
│   ├── stl/
│   │   └── ... (same structure)
│   └── mmoe/
│       └── ... (same structure)
├── train_v2.py
├── evaluate_v2.py
├── models_v2.py
├── loss_v2.py
└── ... (other files)
```

## NEXT STEPS

1. Train all three models and compare metrics
2. Choose best model based on Pearson r
3. Analyze per-trait performance to find strong/weak traits
4. Fine-tune W_SPLOT on best model for final performance
5. Document results in your analysis

Good luck! 🚀
