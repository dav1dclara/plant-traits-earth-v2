"""
Export Google Satellite Embedding V1 (AlphaEarth) aggregated to ~1km to Google Drive.

Usage:
    python export_satellite_embeddings.py [--year 2022]

Requires: earthengine-api  (authenticated via `earthengine authenticate`)
"""

import argparse

import ee

BBOX = [5.9, 45.8, 10.5, 47.8]  # Switzerland
DRIVE_FOLDER = "plant-traits-earth-v2/embeddings"
COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, default=2022)
args = parser.parse_args()

ee.Authenticate()
ee.Initialize(project="plant-traits-earth-v2")

col = ee.ImageCollection(COLLECTION).filterDate(
    f"{args.year}-01-01", f"{args.year + 1}-01-01"
)
embeddings = (
    col.mosaic()
    .setDefaultProjection(col.first().projection())
    .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=10000)
)

region = ee.Geometry.Rectangle(BBOX)
name = f"satellite_embedding_v1_{args.year}_switzerland"

task = ee.batch.Export.image.toDrive(
    image=embeddings,
    description=name,
    folder=DRIVE_FOLDER,
    fileNamePrefix=name,
    scale=1000,
    region=region,
    maxPixels=int(1e10),
    fileFormat="GeoTIFF",
    formatOptions={"cloudOptimized": True},
)
task.start()
print(f"Started {name}  (task {task.id})")
print("Monitor at: https://code.earthengine.google.com/tasks")
