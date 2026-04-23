# Plan for 1km data

## Motivation

- Problem 1 - Targets/Labels: The combined approach (prefer sPlot where sPlot is available) creates a distribution shift in the targets. Naively combinining them confuses the model.

- Problem 2 - Predictors: Predictors are aggregated to 1km/22km. Native resolution gets reduced to single averages and distribution information is thrown away.

- Problem 3 - Modelling: Our current approach (Respatch at 15x15 patches and 22km resolution) doesn't really exploit spatial context, but at 22km, larger patches also don't make sense.


## Key decisions

- Resolution: Focus exclusively on 1km. This is where CNNs can add the most value.
- Target integration: Do not naively combine sPlot and GBIF into one target. Treat them as related but distinct quantities and let the model learn the difference.
- Predictors enhancement: If time and compute allows, extract data at higher resolution (for example canopy height).
- Spatial split: Stratified splitting by H3 cells and biomes. Evaluate on sPlot only.

## Preprocessing

### Data splitting

Assign all labeled 1km pixels to H3 cells. Classify each hexagon by dominant biome. Within each biome, randomly assign hexagons to train/val/test at 80/10/10. Run 50 random assignments, select the one that maximizes biome balance while ensuring minimum sPlot representation in test.

### Predictor pre-processing

Apply per-predictor normalization. If time allows, process more data at native resolution. Integrate more data sources.

##
