# Data Pipeline

## Overview

```
RasterDataset subclasses (TorchGeo)
    → IntersectionDataset (predictors & targets)
        → GridGeoSampler (tile full extent)
            → chips on disk (HDF5/zarr)
                → plain PyTorch Dataset (training)
```

## Steps

### 1. Define datasets (`src/ptev2/data/dataset.py`)
- Each EO source (MODIS, SoilGrids, etc.) is a `RasterDataset` subclass
- Same pattern for target trait maps
- TorchGeo handles CRS alignment and resolution matching

### 2. Combine (`create_dataset`)
- Combine all predictors into a `UnionDataset`
- Intersect with targets: `predictors & targets` → `IntersectionDataset`
- Only regions where both predictors and targets exist are sampled

### 3. Pre-chip (`scripts/preprocess.py`, to be implemented)
- Tile the full spatial extent with `GridGeoSampler(size=64, stride=64)`
- Iterate once and save all chips to a single HDF5/zarr file
- Apply spatial train/val split (hold out geographic blocks)

### 4. Train
- Load chips with a plain PyTorch `Dataset` — no TorchGeo in the training loop
- Fast sequential reads from HDF5/zarr

## Notes
- Pre-chipping is a one-time cost; after that TorchGeo is no longer needed
- Spatial (block) train/val split is preferred over random split to avoid spatial autocorrelation leakage
- Patch size (default 64×64 px) depends on model architecture
