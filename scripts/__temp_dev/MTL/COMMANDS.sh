#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# MTL/MMOE TRAINING - COPY-PASTE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file contains exact commands you can copy-paste directly
# Each section is independent - pick what you want to run
#
# ═══════════════════════════════════════════════════════════════════════════════

# Navigate to MTL folder
cd /scratch3/plant-traits-v2/dsaini/plant-traits-earth-v2/scripts/__temp_dev/MTL

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 1: QUICK TEST (10 minutes)
# Trains MTL for a few epochs to verify everything works
# ═══════════════════════════════════════════════════════════════════════════════

# Edit train_v2.py and change:
#   EPOCHS = 5  (instead of 50)
# Then run:
python train_v2.py
python evaluate_v2.py

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 2: FULL TRAINING (50 minutes - ONE MODEL)
# Train MTL model fully
# ═══════════════════════════════════════════════════════════════════════════════

python train_v2.py
python evaluate_v2.py

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 3: COMPARE THREE MODELS (2.5 hours - RECOMMENDED)
# Train STL, MTL, and MMoE to beat STL baseline
# ═══════════════════════════════════════════════════════════════════════════════

# === Train STL ===
# Edit train_v2.py: MODEL_TYPE = "stl"
# Edit evaluate_v2.py: MODEL_TYPE = "stl"
python train_v2.py
python evaluate_v2.py

# === Train MTL ===
# Edit train_v2.py: MODEL_TYPE = "mtl"
# Edit evaluate_v2.py: MODEL_TYPE = "mtl"
python train_v2.py
python evaluate_v2.py

# === Train MMoE ===
# Edit train_v2.py: MODEL_TYPE = "mmoe"
# Edit evaluate_v2.py: MODEL_TYPE = "mmoe"
python train_v2.py
python evaluate_v2.py

# === Compare results ===
echo "=== STL ==="
cat results/stl/eval_results/metrics.json | grep -A5 pearson

echo "=== MTL ==="
cat results/mtl/eval_results/metrics.json | grep -A5 pearson

echo "=== MMoE ==="
cat results/mmoe/eval_results/metrics.json | grep -A5 pearson

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 4: OPTIMIZED TUNING (1 hour - FOR ONE MODEL)
# Train MTL with better hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════

# Edit train_v2.py and change:
#   MODEL_TYPE = "mtl"
#   EPOCHS = 100
#   W_SPLOT = 16.0  (instead of 8.0)
#   LR = 5e-5  (instead of 1e-4)
# Then run:
python train_v2.py
python evaluate_v2.py

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 5: DATA INSPECTION (10 seconds - OPTIONAL)
# View zarr file structure and create sample files
# ═══════════════════════════════════════════════════════════════════════════════

python inspect_zarr.py

# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 6: VISUALIZE PREDICTIONS (5 minutes - OPTIONAL)
# Create heatmaps of predictions vs targets
# ═══════════════════════════════════════════════════════════════════════════════

# First run evaluation to create test_preds.npy and test_targets.npy
python evaluate_v2.py
# Then create heatmaps
python heatmap.py

# ═══════════════════════════════════════════════════════════════════════════════
# KEY HYPERPARAMETERS TO EDIT IN train_v2.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# Line 30:  MODEL_TYPE = "mtl"     # Change to "stl" or "mmoe"
# Line 34:  EPOCHS = 50            # More epochs = better (50-100)
# Line 35:  LR = 1e-4              # Learning rate (1e-3 to 1e-5)
# Line 40:  W_GBIF = 1.0           # GBIF weight (usually 1.0)
# Line 41:  W_SPLOT = 8.0          # sPlot weight (1.0 to 16.0) - TUNE THIS!
# Line 43:  EARLY_STOP_PATIENCE = 15  # How many epochs before stopping
# Line 32:  BATCH_SIZE = 32        # Batch size (16, 32, or 64)
#
# ═══════════════════════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════
#
# Out of memory? Reduce BATCH_SIZE to 16 or 8
# Loss is NaN? Reduce LR to 1e-5 or increase GRAD_CLIP to 10.0
# No improvement? Increase W_SPLOT to 16.0
# Early stopping too early? Increase EARLY_STOP_PATIENCE to 25
#
# ═══════════════════════════════════════════════════════════════════════════════
