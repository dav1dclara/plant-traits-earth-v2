# Native Resolution Experiment Plan

## Motivation
- Current 22 km aggregation loses fine-grained spatial heterogeneity.
- EO data is available at native resolution (MODIS 250m-1km, SoilGrids 250m, WorldClim 30s).
- sPlot/GBIF labels are point/plot observations, not aggregated 22 km cells.
- Hypothesis: native EO with support-aware label matching can recover more signal.

---

## Three Variants to Compare

### Variant A: Baseline (Current 22 km)
- **EO:** Aggregated to 22 km grid.
- **Labels:** Rasterized and aggregated to 22 km.
- **Patches:** 15×15 at 22 km stride 10 (current standard).
- **Loss:** Weighted masked dense loss, source-aware masking.
- **Purpose:** Control, already running.

### Variant B: Native EO + Local Aggregation
- **EO:** At native resolution (no coarsening). Stack multiple sources at their native grids.
- **Labels:** For each sPlot/GBIF point, extract a **support region** around the point.
  - Typical support radius: 5 km (conservative) to 10 km (more aggressive).
  - Aggregate EO within that radius as context.
  - Use point value as label, but with spatial weighting kernel.
- **Patches:** Variable-sized patches centered on points, not fixed grid.
- **Loss:** MSE/RMSE with per-point weight based on label confidence/coverage.
- **Sampling:** Bootstrap from sPlot points with replacement; stratify by region/trait.
- **Purpose:** Assess whether fine-grained EO helps when labels stay local.

### Variant C: Native EO + Dense Rasterization
- **EO:** At native resolution (no coarsening).
- **Labels:** Rasterize sPlot/GBIF at native resolution.
  - Each point becomes a pixel-level label at the finest common EO resolution (e.g., 250 m).
  - Accept that some labels will be sparse in some regions.
  - Use distance-weighted labels: closer points have higher confidence.
- **Patches:** Fixed patches at 250 m resolution, but only in regions with label coverage.
- **Loss:** Masked loss that only optimizes on labeled pixels + regularization on unlabeled.
- **Sampling:** Chips sampled with stratification on label density and source.
- **Purpose:** Maximum fidelity to data; test if it helps or hurts due to increased noise.

---

## Technical Implementation Sketch

### Data Preparation

#### Variant B: Support-Aware Labels
```python
# Pseudocode
for point in splot_points:
    lon, lat, trait_value = point.lon, point.lat, point.trait
    # Extract EO in 5 km buffer around point
    eo_patch = extract_eo_buffer(lon, lat, radius_km=5, native_res=True)
    # Compute spatial weight: stronger near point, decay outward
    weights = spatial_decay_kernel(eo_patch, lon, lat, scale_km=2)
    # Store as: (eo_patch, trait_value, weights, source)
    samples.append({
        'eo': eo_patch,
        'label': trait_value,
        'weight': weights,
        'source': 'splot',
        'location': (lon, lat)
    })
```

#### Variant C: Native Rasterization
```python
# Pseudocode
# Create fine-grained target grid at 250 m resolution
target_grid = create_native_grid(bounds=global_bounds, res=250)
# For each trait, rasterize sPlot and GBIF as separate arrays
splot_raster = rasterize_points(splot_points, target_grid, decay_radius_m=1000)
gbif_raster = rasterize_points(gbif_points, target_grid, decay_radius_m=500)
# Stack EO bands at same native resolution
eo_cube = stack_native_eo(bounds, res=250)
# Patches: sample chips, only include if label coverage > threshold
chips = []
for i, j in grid_positions:
    eo_chip = eo_cube[i:i+H, j:j+W, :]
    label_chip = splot_raster[i:i+H, j:j+W]
    if label_coverage(label_chip) > 0.05:  # Only 5% valid pixels needed
        chips.append({'eo': eo_chip, 'label': label_chip, 'source': 'splot'})
```

### Loss Functions

#### Variant B: Point-Level Loss with Spatial Weight
```python
class SpatialWeightedMSE(nn.Module):
    def forward(self, pred, label, spatial_weight, source_mask=None):
        # pred: (B, C, H, W) predicted traits across patch
        # label: (B, C) scalar trait values
        # spatial_weight: (B, H, W) spatial confidence weights

        # Aggregate predictions across patch using spatial weight as pooling
        weighted_pred = (pred * spatial_weight.unsqueeze(1)).sum(dim=(2, 3)) / spatial_weight.sum(dim=(1, 2), keepdim=True)

        # MSE with optional source weighting
        loss = ((weighted_pred - label.unsqueeze(2)) ** 2).mean()

        if source_mask is not None:  # source_mask: 1.0 for sPlot, 0.2 for GBIF
            loss = (loss * source_mask).mean()

        return loss
```

#### Variant C: Dense Masked Loss + Regularization
```python
class NativeDenseLoss(nn.Module):
    def forward(self, pred, target, target_mask, source_mask=None, lambda_reg=0.01):
        # pred: (B, C, H, W) dense predictions
        # target: (B, C, H, W) dense target raster
        # target_mask: (B, C, H, W) binary mask where labels exist

        # Supervised loss: only on labeled pixels
        supervised = ((pred - target) ** 2)[target_mask].mean()

        # Regularization: smooth predictions on unlabeled pixels to reduce noise
        unlabeled = ~target_mask
        if unlabeled.any():
            # Gradient penalty (smoothness)
            smooth_loss = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).pow(2)[unlabeled[:, :, 1:, :]].mean()
            smooth_loss += (pred[:, :, :, 1:] - pred[:, :, :, :-1]).pow(2)[unlabeled[:, :, :, 1:]].mean()
        else:
            smooth_loss = 0.0

        # Combine
        total_loss = supervised + lambda_reg * smooth_loss

        # Apply source weighting if provided
        if source_mask is not None:
            total_loss = total_loss * source_mask.mean()

        return total_loss
```

---

## Experimental Setup

### Training Configuration

| Aspect | Variant A | Variant B | Variant C |
|--------|-----------|-----------|-----------|
| **EO Resolution** | 22 km | Native (250m–1km) | Native (250m–1km) |
| **Patch Size (spatial)** | 15×15 @ 22km | 5 km radius (variable) | 32×32 @ 250m (adaptive) |
| **Label Type** | Aggregated raster | Point with spatial weight | Dense raster |
| **Loss** | WeightedMaskedDenseLoss | SpatialWeightedMSE | NativeDenseLoss + smooth |
| **Batch Size** | 64 chips | 32 points | 32 chips |
| **Epochs** | 30 | 30 | 30 |
| **Optimizer** | Adam, lr=1e-3 | Adam, lr=1e-4 (slower convergence) | Adam, lr=5e-4 |
| **sPlot/GBIF Weight** | 1.0 / 0.2 | 1.0 / 0.1 | 1.0 / 0.1 |

### Sampling & Validation Strategy

**Training:**
- Variant A: Random chip sampling from 22 km grid.
- Variant B: Bootstrap from sPlot points with replacement; stratify by continent/trait variance.
- Variant C: Random chip sampling from grid, but only regions with ≥5% label coverage.

**Validation:**
- **All variants:** Split sPlot by region (holdout spatial regions, not random).
- **Separate eval on GBIF:** Run inference on GBIF-only regions to measure transfer.

**Test:**
- **Primary:** sPlot test set (source=2 in your current setup).
- **Secondary:** GBIF test set (separate evaluation).

---

## Evaluation Metrics

### Per-Variant Metrics (on sPlot Test)
- Macro Pearson r
- Macro RMSE
- Macro R²
- Macro MAE
- Per-trait RMSE (sorted by difficulty)

### Analysis
- **Variant A vs B:** Does fine-grained EO + point supervision beat 22 km aggregation?
- **Variant B vs C:** Does local weighting outperform dense rasterization?
- **Noise vs Signal:** Plot prediction std by trait. Finer resolution should reduce bias, not increase noise.
- **GBIF transfer:** How much does each variant's sPlot-trained model generalize to GBIF regions?

---

## Expected Outcomes

### Best Case (Variant B or C wins)
- RMSE improves by 5–15% on sPlot.
- Improvement is largest for traits with high local heterogeneity.
- GBIF transfer doesn't collapse.

### Worst Case (Variant A still best)
- Fine-grained EO adds noise without signal benefit.
- Suggests 22 km is the signal saturation point, or labels are inherently 22 km or coarser.
- This would validate the data-bottleneck hypothesis.

### Neutral (All similar)
- Improvement marginal (<2%).
- Suggests architectural or label-quality factors dominate.

---

## Implementation Checklist

- [ ] Extract native-res EO stacks (MODIS 250m, SoilGrids, WorldClim).
- [ ] Implement support-aware label sampling for Variant B.
- [ ] Implement dense rasterization for Variant C.
- [ ] Define spatial decay kernels (Gaussian or triangular).
- [ ] Write new loss functions (SpatialWeightedMSE, NativeDenseLoss).
- [ ] Adapt dataloader to handle variable-size patches (Variant B) and dense labels (Variant C).
- [ ] Run 3 seeds × 3 variants = 9 training runs.
- [ ] Generate comparison table (RMSE, Pearson r per variant).
- [ ] Plot RMSE by trait type (high-variance vs low-variance).
- [ ] Assess GBIF generalization.

---

## Timeline Estimate
- Data prep: 1–2 days.
- Loss/dataloader implementation: 1–2 days.
- Training (3 seeds × 3 variants): 5–7 days.
- Analysis & reporting: 1 day.
- **Total: 8–12 days.**

---

## Notes for Supervisor
- This experiment isolates the role of EO resolution and label spatial support.
- If Variant B or C wins, it justifies the effort to scale to native resolution.
- If Variant A wins, it shifts focus back to data quality, label curation, or alternative architectures.
- Either way, the result is actionable for the next phase.
