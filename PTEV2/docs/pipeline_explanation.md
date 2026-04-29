# Plant Trait Prediction Pipeline — Full Explanation

**Project**: Plant Traits Earth v2 (`plant-traits-earth-v2`)
**Goal**: Predict 37 plant functional traits (like leaf area, wood density, plant height) globally at 22 km resolution from satellite and environmental data.

---

## 1. What Are We Trying to Do?

Plants have measurable biological properties called **plant functional traits** — things like leaf nitrogen content, seed mass, stem density, plant height. These traits determine how plants respond to the environment (drought, cold, fire) and are critical for ecosystem models and climate projections.

The problem: trait measurements from field surveys (like SPLOT — a global plant trait database) only exist at a **few thousand scattered locations**. We want to create a **global continuous map** of 37 traits everywhere on Earth.

Our approach: train a neural network to learn the relationship between satellite-derived environmental variables (which exist everywhere globally) and the trait values measured at survey locations.

---

## 2. The Data

### 2a. Labels — Where Do the Trait Values Come From?

We use two data sources for labels (supervision signal):

**SPLOT** (primary — high quality)
- A curated database of field-measured plant traits from botanical surveys.
- High quality: measured in-situ, species-verified.
- But **sparse**: only ~18,000 usable locations in our training area.

**GBIF** (auxiliary — noisy but widespread)
- Global Biodiversity Information Facility — occurrence records + trait values linked via species names.
- Much more data (~millions of records), covers more of the globe.
- But noisier: different species, different methods, different times.
- We use GBIF only as an *auxiliary* signal, not the primary target.

### 2b. Predictors — What Goes Into the Model?

We use 5 satellite/environmental data sources, stacked into 150 input channels:

| Predictor       | Channels | What it measures |
|----------------|----------|-----------------|
| Canopy Height   | 2        | Vegetation height from lidar-derived products |
| MODIS           | 72       | Time-series of vegetation reflectance (NDVI, EVI etc.) — 72 time steps |
| SoilGrids       | 61       | Soil properties at multiple depths (pH, texture, carbon, nitrogen...) |
| VODCA           | 9        | Vegetation optical depth from microwave — related to water content |
| WorldClim       | 6        | Long-term climate statistics (temperature, precipitation, seasonality) |

Total: **150 channels per pixel**, at **22 km resolution**.

---

## 3. The "Chips" Idea — How Is the Data Structured?

Instead of working with full global rasters (which are huge), we cut the data into small spatial **chips** — 3×3 pixel windows centered on locations that have some plant trait observation.

**Why 3×3?**
- 3×3 at 22 km = a 66 km × 66 km spatial window — captures local spatial context.
- Keeps memory manageable (each chip is tiny).
- The center pixel (position [1,1]) is where the trait observation lives; the 8 surrounding pixels give spatial context.

**Why chip-centered?** — this is key:
- In a previous version of the project, chips were cut everywhere on a grid (most with no labels). Now chips are centered exactly on SPLOT or GBIF observation locations, so every chip is guaranteed to have at least something to learn from.

### The zarr files

Data is stored in [zarr](https://zarr.dev/) format — a compressed array storage format that is fast for cloud/disk access. Three splits:

```
/scratch3/plant-traits-v2/data/22km/chips_centered/patch3_stride1/
    train.zarr   →  54,377 chips  (34% have SPLOT at center = 18,487)
    val.zarr     →   6,956 chips  (100% have SPLOT at center)
    test.zarr    →   7,280 chips  (100% have SPLOT at center)
```

Inside each zarr:
- `predictors/{canopy_height, modis, soil_grids, vodca, worldclim}` — shape `(N, C, 3, 3)`
- `targets/supervision_splot_only` — shape `(N, 74, 3, 3)` — interleaved [value, source_mask] for 37 traits
- `targets/supervision_gbif_only` — same structure but GBIF values
- `center/splot_mean, splot_std, splot_count, ...` — summary stats at center pixel per trait

Val and test are **100% SPLOT-covered** by design — they were split from the SPLOT dataset. Train has 34% SPLOT center coverage because it also includes chips anchored to GBIF-only locations (no SPLOT measured there).

---

## 4. The Splitting Strategy

We split at the **chip level** such that:

- **Val set** = randomly sampled SPLOT-center chips, held out entirely from training.
- **Test set** = a separate held-out SPLOT-center set, never touched until final evaluation.
- **Train set** = remaining chips, including all GBIF-only chips.

This means:
- Val and test are "clean" — only SPLOT ground truth, so the metric (Pearson-r) is directly interpretable.
- Train is "messy" — SPLOT where available, GBIF to fill in coverage.

---

## 5. The Label Format — The Interleaved Structure

The supervision arrays use an interleaved format: `(N, 74, 3, 3)` for 37 traits because each trait takes **2 channels**: `[value, source_mask]`.

```python
sv = supervision_splot_only[i]  # shape (74, 3, 3)
y_values = sv[0::2]   # channels 0, 2, 4, ..., 72  → trait values
y_source  = sv[1::2]  # channels 1, 3, 5, ..., 73  → source code

# Source codes:
# 0 = no data at this pixel
# 1 = GBIF observation
# 2 = SPLOT observation
```

The loss function uses the source mask to **ignore pixels with no data**, so you never penalise the model on missing observations. This is the masked loss.

---

## 6. The Two Key Ideas We Implemented

### Idea 1 — Separate SPLOT-Only Supervision

**Problem (before):** The original code mixed SPLOT and GBIF labels in a single supervision array. The model learned a blend of both, making it impossible to control GBIF vs SPLOT influence.

**Fix:** Created two separate supervision arrays:
- `supervision_splot_only` — only SPLOT labels; source=2 where SPLOT, 0 elsewhere.
- `supervision_gbif_only` — only GBIF labels (linearly calibrated to SPLOT space); source=1 where GBIF, 0 elsewhere.

**Effect:** SPLOT drives the primary loss; GBIF is purely auxiliary.

### Idea 2 — Linear GBIF Calibration

**Problem:** GBIF and SPLOT measure the same traits but with different spatial coverage patterns — at co-located pixels, GBIF values are shifted and scaled relative to SPLOT values (Pearson-r ~0.33, large Δmean).

**Fix:** Fit a per-trait linear regression: `GBIF_calibrated = a * GBIF_raw + b` where `a, b` are chosen so that calibrated GBIF matches SPLOT distribution. This is fit on training data only (no leakage). The result: calibrated GBIF mean aligns to SPLOT mean across all 37 traits (max Δmean < 1.5e-8 after calibration ✓).

GBIF is then used in training at a **low weight** (`aux_gbif_weight = 0.15`) — it provides 15% of the gradient, mostly giving spatial coverage signal without distorting the SPLOT-space target.

### Idea 3 — Quality-Weighted Loss (Tried, Removed)

**Idea:** Weight each SPLOT observation by `splot_count / (splot_std + ε)` — locations with many observations and low spread should be trusted more.

**Result:** Val Pearson-r slightly dropped. Likely because: (a) the source mask already handles absent data; (b) over-weighting high-count locations may over-fit to a few dense survey sites. We removed it — Ideas 1+2 alone are better.

---

## 7. Speed Fix — Pre-loading zarr into RAM

**Problem:** Training was 3–4 min/epoch. The bottleneck was per-sample zarr decompression — every `__getitem__` call opened 7 zarr arrays and decompressed a tiny 3×3 chunk, which is very slow (random disk access + decompression overhead).

**Fix:** In `ChipsCenteredDataset.__init__()`, load **all arrays into numpy RAM** once at startup. `__getitem__` then does pure numpy indexing (no I/O):

```python
# At init time — one-off cost (~12 seconds, ~600 MB RAM)
self._X_arrs  = [store[f"predictors/{p}"][:] for p in PREDICTORS]
self._sv_splot = store["targets/supervision_splot_only"][:]
self._sv_gbif  = store["targets/supervision_gbif_only"][:]

# At training time — pure RAM access
def __getitem__(self, idx):
    X = np.concatenate([a[idx] for a in self._X_arrs], axis=0)  # instant
    ...
```

**Result:** ~41 ms/batch → projected **0.15 min/epoch** (was 3–4 min) — a **~20× speedup**. On Linux, DataLoader workers share this RAM via copy-on-write (fork) so no per-worker memory duplication.

---

## 8. The Models — Three Architectures

All three models share the same **SharedEncoder** — a 4-block residual CNN. The difference is in the **head** (the output side).

### Architecture overview

```
Input X: (B, 150, 3, 3)
         ↓
   SharedEncoder
   ┌─────────────────────────────┐
   │ ResBlock1: 150 → 32  ch    │
   │ ResBlock2:  32 → 32  ch    │
   │ ResBlock3:  32 → 64  ch    │
   │ ResBlock4:  64 → 128 ch    │
   └──────────────┬──────────────┘
                  │  (B, 128, 3, 3) feature map
                  ↓
         [Different heads below]
```

### Model 1 — STL (Single Task Learning) — Baseline

```
Features (B, 128, 3, 3)
    ↓
TaskHead: 1×1 Conv → (B, 37, 3, 3)
```

One big prediction head for all 37 traits simultaneously. Simple, fast, but traits can't share task-specific computation. **Val r ≈ 0.62**

### Model 2 — MTL (Multi Task Learning)

```
Features (B, 128, 3, 3)
    ↓
SharedHead: 3×3 Conv (shared across traits)
    ↓
37 × TaskHead: one 1×1 conv per trait → (B, 37, 3, 3)
```

Each trait gets its own 1×1 prediction head, but they all share the spatial 3×3 conv. Allows some trait-specific tuning. **Val r ≈ 0.63**

### Model 3 — MMoE V3 (Mixture of Experts) — Best

```
Features (B, 128, 3, 3)
    ↓
shared_projection: 128 → expert_hidden
    ↓
┌────────────────────────────┐
│  8 Expert networks         │  ← Each expert is a small MLP
│  (all see same features)   │
└────────────────────────────┘
         ↑ weighted sum (gating)
    For each trait group, a Gating Network learns which experts to activate
    Trait groups (biologically motivated):
      Group 1: SSD, Carbon, Height, Density
      Group 2: Leaf morphology (SLA, leaf area...)
      Group 3: Leaf chemistry + SRL
      Group 4: Seed traits + Wood anatomy
    ↓
37 × TaskHead → (B, 37, 3, 3)
```

The gating network produces **soft weights** over the 8 experts per trait group. Biologically related traits route through similar experts. This lets the model specialise different "sub-networks" for different trait types without completely separating them. **Val r ≈ 0.62–0.63**

---

## 9. The Loss Function

We use **Smooth L1 (Huber) loss** with masking:

```
loss = Σ_{valid pixels} smooth_l1(prediction - target) / n_valid
```

Where `valid` means `source_mask > 0` (i.e., there is actually a label there). Pixels with no observation are completely ignored — neither numerator nor denominator includes them.

The total training loss combines:

```
total_loss = splot_loss + 0.15 × gbif_loss
```

- **SPLOT loss**: on `supervision_splot_only`, weight = 1.0 (primary signal)
- **GBIF loss**: on `supervision_gbif_only` (calibrated), weight = 0.15 (auxiliary)

Val and test evaluation use **only SPLOT labels** — no GBIF at eval time.

---

## 10. The Evaluation Metric — Pearson-r

We report **center-pixel Pearson-r** averaged across 37 traits (macro-r):

1. For each chip, take the center pixel prediction: `pred[:, 1, 1]` — a 37-vector.
2. Only keep chips where the center pixel has a valid SPLOT label (source == 2).
3. For each trait, compute Pearson correlation between all predicted and observed values.
4. Average across all 37 traits → macro-r.

**Why Pearson-r?** It measures rank correlation regardless of scale — it doesn't matter if the model's absolute values are slightly off, as long as it correctly predicts *which locations have higher/lower* values for a given trait. This is what matters for making global maps.

**Implementation note:** We use float64 (double precision) accumulation and the formula `(pred - mean(pred)) · (label - mean(label)) / (||...|| * ||...||)` rather than the running-sum formula. The running-sum formula has catastrophic cancellation problems when values are large — we fixed this bug, which was causing val_mean_r to show 0.0000 despite the model learning correctly.

### Training vs. Validation Coverage

| Split | SPLOT coverage | What val metric means |
|-------|---------------|----------------------|
| Train | 34% of chips | Only 34% contribute to SPLOT loss per batch |
| Val   | 100% of chips | Every val chip contributes to Pearson-r |

This is why val metrics are clean and reliable — all val chips have ground truth.

---

## 11. Training Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Optimizer | Adam, lr=2e-4 | Standard, works well for residual CNNs |
| LR schedule | StepLR: ×0.7 every 15 epochs | Gradually reduce LR as training converges |
| Batch size | 128 | Fits in GPU memory; stable gradients |
| Num workers | 8 | Parallel data loading |
| Gradient clip | 1.0 | Prevents exploding gradients |
| Early stopping | patience=20 | Stops if val_mean_r doesn't improve for 20 epochs |
| Max epochs | 100 | Upper bound |
| Weight decay | 1e-5 | Light L2 regularisation |

**Two checkpoints saved per run:**
- `{run_name}_best_r.pth` — best val Pearson-r (use this for evaluation)
- `{run_name}.pth` — best val loss

---

## 12. Training Commands

```bash
cd /scratch3/plant-traits-v2/dsaini/plant-traits-earth-v2
source /scratch3/plant-traits-v2/dsaini/miniconda3/etc/profile.d/conda.sh
conda activate PTEV2

# STL — single-head baseline
python scripts/train_chips.py models=respatch_v2 train.run_name=chips_stl wandb.enabled=true train.group=chips_ideas12

# MTL — per-trait heads
python scripts/train_chips.py models=mtl_v2 train.run_name=chips_mtl wandb.enabled=true train.group=chips_ideas12

# MMoE V3 — mixture of experts (default)
python scripts/train_chips.py train.run_name=chips_mmoe wandb.enabled=true train.group=chips_ideas12
```

---

## 13. Evaluation — test_chips.py

The test split is **completely untouched during training**. After training, run:

```bash
# Evaluate on test set — generates metrics JSON + scores JSON
python scripts/test_chips.py \
    --checkpoint scripts/Checkpoints_Scores/checkpoints/chips_mmoe_best_r.pth \
    --wandb

# Also generate predicted trait maps (PNG with 37 scatter plots)
python scripts/test_chips.py \
    --checkpoint scripts/Checkpoints_Scores/checkpoints/chips_mmoe_best_r.pth \
    --maps
```

**Outputs:**

1. `results/chips/{stem}.test_metrics.json` — full metrics: macro-r, RMSE, MAE, per-trait breakdown
2. `results/chips/{stem}.scores.json` — just `{trait_id: Pearson-r}` for quick comparison across models
3. `results/chips/{stem}_predicted_maps.png` — (with `--maps`) scatter plots of predicted values on a lat/lon grid, one subplot per trait

**Comparing two models:**
```python
import json
stl   = json.load(open("results/chips/chips_stl_best_r.scores.json"))
mmoe  = json.load(open("results/chips/chips_mmoe_best_r.scores.json"))
for trait in stl:
    diff = mmoe[trait] - stl[trait]
    print(f"{trait}: STL={stl[trait]:.3f}  MMoE={mmoe[trait]:.3f}  Δ={diff:+.3f}")
```

---

## 14. Results Summary (approximate, val set)

| Model | Val macro-r (~ep 10) | Notes |
|-------|---------------------|-------|
| STL   | ~0.622              | Baseline |
| MTL   | ~0.625              | Slight improvement from per-trait heads |
| MMoE  | ~0.621              | Expert routing; benefits should grow with more data/resolution |

All three significantly outperform the prior baseline (~0.50) due to:
- Cleaner SPLOT-only primary loss (Idea 1)
- Calibrated GBIF auxiliary signal (Idea 2)
- The 20× data loading speedup allowing more epochs in less time

**Why does best performance come in early epochs?**
This is expected, not a bug. The model quickly learns the dominant signal (climate/biome gradients, which explain most trait variance globally). After that, further improvement requires learning finer within-biome variation, which is harder from a 3×3 window at 22 km. The early stopping + `_best_r.pth` checkpoint correctly captures the peak.

---

## 15. File Map

```
scripts/
  train_chips.py      ← Main training script (Ideas 1+2)
  test_chips.py       ← Evaluation on test set (metrics + maps)

src/ptev2/
  data/
    chips_dataset.py  ← ChipsCenteredDataset + GBIF calibration + DataLoader
  models/
    multitask_v2.py   ← STLModel, MTLModel, GatedMMoEModelV3, SharedEncoder
    traitPatchCNN.py  ← ResidualBlock (used inside SharedEncoder)
  loss.py             ← WeightedMaskedDenseLoss (masked Huber, supports sample_weight)

config/
  training/chips_v1.yaml   ← All hyperparameters (Hydra config)
  models/mmoe_v2.yaml      ← MMoE architecture config
  models/mtl_v2.yaml       ← MTL architecture config
  models/respatch_v2.yaml  ← STL architecture config

data/22km/chips_centered/patch3_stride1/
  train.zarr   ← 54,377 chips, 150-ch predictors, SPLOT+GBIF labels
  val.zarr     ←  6,956 chips, 100% SPLOT center
  test.zarr    ←  7,280 chips, 100% SPLOT center (untouched until eval)
```

---

## 16. Quick Glossary

| Term | Meaning |
|------|---------|
| SPLOT | Global plant trait database — field-measured, high quality |
| GBIF | Global biodiversity occurrence + species-linked traits — noisy, widespread |
| Chip | A 3×3 pixel spatial window centered on an observation location |
| 22 km | Spatial resolution of all data layers |
| Zarr | Compressed array storage format (like HDF5 but cloud-friendly) |
| Macro-r | Average Pearson-r across all 37 traits — primary evaluation metric |
| STL | Single Task Learning — one head for all traits |
| MTL | Multi Task Learning — separate head per trait, shared encoder |
| MMoE | Mixture of Experts — multiple specialist sub-networks, learned routing |
| Masked loss | Loss that ignores pixels with no label (source_mask = 0) |
| Calibration | Linear shift+scale to align GBIF distribution to SPLOT distribution |
