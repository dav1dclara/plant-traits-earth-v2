# Better maps of plant functional traits – towards planttraits.earth v2

## Setup

### Repository

Clone the repository:

```bash
git clone git@github.com:dav1dclara/plant-traits-earth-v2.git
cd plant-traits-earth-v2/
```

### Dependencies

Create a conda environment, then install the dependencies:

```bash
conda create -n PTEV2 python=3.12
conda activate PTEV2
pip install -r requirements.txt
pip install -e .
```

Install the pre-commit hooks for automatic code formatting and linting on each commit:

```bash
pre-commit install
```

## Project structure

*(create a tree at the end of the project)*

## Data preparation

*(write about data preparation...)*

### Splitting

To split ..., run:

```python
python scripts/splitting/create_splits.py
```

### Data chipping

To chip predictor and target raster data into per-split [Zarr](https://zarr.dev/) stores, run:

```python
python scripts/chipping/chip_rasters.py
```

The script slides a window of `patch_size × patch_size` pixels across all rasters with a given `stride`, producing one chip per position. Each chip is routed to `train`, `val`, or `test` based on which H3 hexagonal cell its center falls in. Pixels within a chip that belong to a different split are masked to `NaN` to prevent data leakage across boundaries. The output is one zarr store per split (e.g. `train.zarr`). Each store contains two groups — `predictors/` and `targets/` — with arrays of shape `(n_chips, n_bands, patch_size, patch_size)`, plus a `bounds` array with the geographic bounding box of each chip.

Patch size, stride, data paths, and which predictors/targets to include are controlled via `config/chipping/default.yaml`.

## Model training

The current 22 km model variants are split into dedicated modules:

- `ptev2.models.pixel_mlp.PixelMLP` for the tiny 1x1-conv baseline.
- `ptev2.models.stl.STLModel` for the single-task ResPatch-style baseline used by the `stl` config.
- `ptev2.models.mtl.MTLModel` for the multi-task shared-encoder variant.
- `ptev2.models.mmoe.GatedMMoEModelV3` for the dense MMoE variant.

Legacy imports through `ptev2.models.models` and `ptev2.models.multitask` remain available for old checkpoints and configs.

### Supervision protocol

- 22 km benchmark: patch-context / center-pixel supervision.
  The model sees a spatial predictor patch, but training/validation/testing loss and metrics are computed only on the center target cell.
- Future 1 km benchmark: patch-context / center-crop supervision.
  The model sees a larger patch, and loss/metrics are computed on a central supervised crop (for example `center_crop_size: 32`), while outer pixels provide context only.
- sPlot and GBIF are treated as separate supervision sources.
  sPlot is primary ground truth; GBIF is auxiliary weak supervision.
  They are not collapsed into one merged target.
  We train a source-aware CNN using sPlot as the primary supervised source and GBIF as a down-weighted auxiliary source, and we validate/test against held-out sPlot.
- Dense supervision is supported only as an ablation.
  With overlapping chips, dense mode can duplicate target pixels unless unique-pixel de-duplication is implemented.
