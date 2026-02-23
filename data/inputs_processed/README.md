# Processed input data

The processed input data is provided by Daniel Lusk and available for download [here](https://kattenborn.go.bwsfs.uni-freiburg.de:11443/web/client/pubshares/4bKqRZEjNagxYEEUcp3XxM/browse).

**Notes by Daniel:**
- Each grid cell represents the community-weighted mean value for the given trait.
- When we have GBIF and sPlot observations for the same grid cell, we prefer to use the sPlot observation.
- Each trait has been power-transformed, so the values are in the transformed space.
- Many of the Earth observation rasters have been quantized into integer format, so, when loading the data, pay attention to the `scale`, `add_offset`, and `_FillValue` (nodata value) attributes when loading the them.
- In the original experiment, validation was done by dividing the globe up into a hexagonal grid, randomly assigning fold IDs to the hexagons, and then propagating the fold ID to the pixels contained within each hexagon to perform spatial k-fold cross validation.
- If following the same cross validation approach, please make sure to only use the sPlot labels in the held-out fold for validation.
- I've provided a JSON file which maps the trait IDs to their actual names.


## Earth Observation Data

### Canopy height (`canopy_height/`)

- `ETH_GlobalCanopyHeight_2020_v1`: the actual estimated canopy height globally (from ETH Zurich, year 2020)
- `ETH_GlobalCanopyHeightSD_2020_v1`: the standard deviation (uncertainty) of that estimate

### MODIS (`modis/`)

MODIS is a NASA satellite sensor. `sur_refl` = surface reflectance (how much light bounces off the land surface in different wavelengths).

**File naming**:
`sur_refl_{band}_2001-2024_m{month}_mean.tif`: Monthly mean surface reflectance for 6 bands across 12 months (72 files total).

**Bands:**
`b01` through `b05` = spectral bands (different wavelengths of light). b01 is red, b02 is near-infrared, b03 is blue, b04 is green, b05 is shortwave infrared. Plants reflect very differently in these bands depending on their health, structure, and traits.

`ndvi` = Normalized Difference Vegetation Index — a derived index combining red and near-infrared bands. It's basically a "greenness" measure, very commonly used in ecology.

**Months:**
`m1` through `m12` = months of the year (January through December). So you get seasonal information — how reflectance changes across the year, which captures things like leaf-on/leaf-off cycles.
2001-2024 = these are long-term averages computed across those years, for each month.

### SoilGrids (`soilgrids/`)

SoilGrids is a global gridded soil information system from ISRIC. It provides predictions of soil properties at multiple depth intervals.

**File naming:**
`{variable}_{depth}_mean.tif`: Mean value for a given soil property at a specific depth interval (61 files total).

**Variables (11):**
- `bdod` = Bulk density of the fine earth fraction (how compact the soil is)
- `cec` = Cation Exchange Capacity (soil's ability to hold nutrients)
- `cfvo` = Coarse fragments volumetric fraction (rocks/gravel content)
- `clay` = Clay content
- `nitrogen` = Total nitrogen
- `ocd` = Organic carbon density
- `ocs` = Organic carbon stocks (only available at 0–30cm)
- `phh2o` = Soil pH in water
- `sand` = Sand content
- `silt` = Silt content
- `soc` = Soil organic carbon content

**Depths:**
6 standard depth intervals: `0-5cm`, `5-15cm`, `15-30cm`, `30-60cm`, `60-100cm`, `100-200cm`. Most variables have all 6 depths (10 × 6 = 60 files), except `ocs` which only has the `0-30cm` layer (+1 = 61 files total).


### VODCA

VODCA = Vegetation Optical Depth Climate Archive. This is a microwave-based satellite product that measures how much water is in vegetation (vegetation water content), which is related to plant structure and function.

**Microwave frequencies:**
C-band, K-band, X-band = different microwave frequencies, each sensitive to different aspects of vegetation (C and X more sensitive to woody biomass, K more to leaf water)

**Statistics:**
mean, p5, p95 = the long-term mean, 5th percentile, and 95th percentile — capturing typical conditions and extremes

### WorldClim

These are bioclimatic variables — long-term climate summaries. The bio_ numbers are standard bioclim variables:

- bio_1 = Annual Mean Temperature
- bio_4 = Temperature Seasonality (how much temperature varies across the year)
- bio_7 = Temperature Annual Range (max month minus min month temp)
- bio_12 = Annual Precipitation
- bio_13-14 = Precipitation of Wettest Month and Driest Month
- bio_15 = Precipitation Seasonality (how variable rainfall is across the year)

## GBIF (Global Biodiversity Information Facility)

...


## sPlot

...
