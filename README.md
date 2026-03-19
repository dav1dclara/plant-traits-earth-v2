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

### Data chipping

To chip predictor and target raster data into per-split [Zarr](https://zarr.dev/) stores, run:

```python
python scripts/preprocessing/chip_rasters.py
```

The script slides a window of `patch_size × patch_size` pixels across all rasters with a given `stride`, producing one chip per position. Each chip is routed to `train`, `val`, or `test` based on which H3 hexagonal cell its center falls in. Pixels within a chip that belong to a different split are masked to `NaN` to prevent data leakage across boundaries. The output is one zarr store per split (e.g. `train.zarr`). Each store contains two groups — `predictors/` and `targets/` — with arrays of shape `(n_chips, n_bands, patch_size, patch_size)`, plus a `bounds` array with the geographic bounding box of each chip.

Patch size, stride, data paths, and which predictors/targets to include are controlled via `config/chipping/default.yaml`.

## Model training
