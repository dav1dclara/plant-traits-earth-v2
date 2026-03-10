# Data sources

This file describes the data sources used in this project.

## Location and structure

All data used in this project is located in `/scratch3/plant-traits-v2/data/`. Total size: ~29 GB, 518 files.

```
data/
├── 1km/                         # EO data + CV splits at 1 km resolution (~28.6 GB)
│   ├── eo_data/
│   │   ├── canopy_height/
│   │   ├── modis/
│   │   ├── soilgrids/
│   │   ├── vodca/
│   │   └── worldclim/
│   └── skcv_splits/
├── 22km/                        # EO data + targets + CV splits at 22 km resolution (~433 MB)
│   ├── eo_data/
│   │   ├── canopy_height/
│   │   ├── modis/
│   │   ├── soilgrids/
│   │   ├── vodca/
│   │   └── worldclim/
│   ├── gbif/
│   ├── gbif_original/
│   ├── splot/
│   ├── splot_original/
│   └── skcv_splits/
├── power_transformer_params.csv  # Yeo-Johnson transformation parameters per trait
├── trait_mapping.json            # TRY trait IDs → descriptions + units (37 traits)
└── README.md                     # Back-transformation guide
```

---

## Earth Observation (EO) predictors

EO data is available at both 1 km and 22 km resolution. Each resolution follows the same structure under `eo_data/`.

### Canopy height

**Files:** `ETH_GlobalCanopyHeight_2020_v1.tif`, `ETH_GlobalCanopyHeightSD_2020_v1.tif`

> **Q1:** What is the source of the canopy height data (ETH product)? What year does it represent and what is the original resolution before resampling?

### MODIS

**Files:** 72 GeoTIFFs. Surface reflectance bands b01–b05 and NDVI, each as 12 monthly means (2001–2024).
Naming: `sur_refl_[band]_2001-2024_m[1-12]_mean.tif`

> **Q2:** Which MODIS product(s) were used (e.g. MOD09A1, MYD09)? Were bands from Terra, Aqua, or combined? Were the monthly means computed from all years 2001–2024, or a specific subset?

### SoilGrids

**Files:** 62 GeoTIFFs. 11 soil properties at 6 depth intervals (0–5, 5–15, 15–30, 30–60, 60–100, 100–200 cm).
Properties: `bdod`, `cec`, `cfvo`, `clay`, `nitrogen`, `ocd`, `ocs`, `phh2o`, `sand`, `silt`, `soc`.

> **Q3:** Which version of SoilGrids was used (v1 / v2.0)? Were all depth intervals used as separate features, or were they aggregated in any way?

### VODCA

**Files:** 9 GeoTIFFs. Three microwave bands (C, K, X), each with mean, 5th percentile, and 95th percentile.

> **Q4:** What time period was used to compute the VODCA statistics (mean, p5, p95)? What does each band represent physically and why were these three bands selected?

### WorldClim

**Files:** 6 GeoTIFFs. Bioclimatic variables: `bio_1`, `bio_4`, `bio_7`, `bio_12`, `bio_13`, `bio_14`, `bio_15`.

> **Q5:** Which WorldClim version was used (v1 / v2.1)? Why were these specific bioclimatic variables selected rather than the full set of 19?

---

## Target variables

Target data is only available at 22 km resolution. There are 37 traits in total, identified by TRY trait IDs.

**Trait IDs:** 4, 6, 11, 13, 14, 15, 18, 21, 26, 27, 46, 47, 50, 55, 78, 95, 138, 144, 145, 146, 163, 169, 223, 224, 237, 281, 282, 289, 297, 351, 614, 1080, 3106, 3107, 3112, 3113, 3114, 3117, 3120

Each trait has a raster file `X[trait_id].tif` in `gbif/` and `splot/`, plus unprocessed originals in `gbif_original/` and `splot_original/`.

### GBIF (`gbif/`, `gbif_original/`)

> **Q6:** What exactly is stored in the GBIF target rasters? Are these community-weighted mean (CWM) trait values computed from species occurrence records + TRY traits? What is the aggregation method (e.g. mean over species in each pixel)?

> **Q7:** What is the difference between `gbif/` and `gbif_original/`? What processing was applied to go from original to processed (e.g. power transformation, masking, resampling)?

### sPlot (`splot/`, `splot_original/`)

> **Q8:** What exactly is stored in the sPlot target rasters? Are these plot-level observations aggregated to the 22 km grid (e.g. mean of plots within each pixel)? How many sPlot observations were used?

> **Q9:** What is the difference between `splot/` and `splot_original/`? Same question as Q7 for the sPlot pipeline.

> **Q10:** During training, are GBIF and sPlot targets used together or separately? Is one considered more reliable than the other?

---

## Cross-validation splits (`skcv_splits/`)

Stratified k-fold cross-validation splits stored as Parquet files.

- `1km/skcv_splits/`: 12 files (`X[split_id]_mean.parquet`)
- `22km/skcv_splits/`: 38 files (`X[split_id]_mean.parquet`)

> **Q11:** What stratification variable was used for the spatial cross-validation (e.g. climate zone, biome, geographic region)? What does each Parquet file contain — pixel coordinates, fold assignments, or something else?

> **Q12:** Why are there different numbers of splits at 1 km (12) vs 22 km (38)?

---

## Trait metadata

### `trait_mapping.json`

Maps TRY trait IDs to descriptions and units for all 37 traits.

> **Q13:** Is the trait selection (37 traits) based on data availability in TRY, coverage in GBIF/sPlot, or scientific relevance? Is there a reference paper for this trait set?

### `power_transformer_params.csv`

Yeo-Johnson transformation parameters for each trait (used to normalize target values).

> **Q14:** Were traits power-transformed before being rasterized, or after? Are the parameters trait-specific only, or also source-specific (separate params for GBIF vs sPlot)?
