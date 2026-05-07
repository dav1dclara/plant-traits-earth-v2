"""
Export Google Satellite Embedding V1 (AlphaEarth) aggregated to ~1km in EPSG:4326.

GEE does not support EPSG:6933 / EASE-Grid 2.0 internally, so we export in
EPSG:4326 at ~0.009 degrees (~1km at equator) and warp to the reference grid
locally afterwards using reproject_to_reference.py.

Usage:
    # Test export (small region, quick turnaround):
    python export_satellite_embeddings.py --mode test

    # Full global export:
    python export_satellite_embeddings.py --mode global

Requires: earthengine-api, authenticated via `earthengine authenticate`
"""

import argparse

import ee

# ~1km in degrees; used for GEE-side aggregation and export resolution.
EXPORT_SCALE_DEG = 0.008983  # 1000m / 111320m per degree at equator

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
YEAR = 2022


def get_embedding(year: int) -> ee.Image:
    col = ee.ImageCollection(COLLECTION).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    native_proj = col.first().projection()
    return col.mosaic().setDefaultProjection(native_proj)


def aggregate_to_1km(image: ee.Image, region: ee.Geometry | None = None) -> ee.Image:
    """Mean-pool 10m embeddings to ~1km in EPSG:4326."""
    aggregated = image.reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=10000
    ).reproject(crs="EPSG:4326", scale=EXPORT_SCALE_DEG * 111320)
    if region is not None:
        aggregated = aggregated.clip(region)
    return aggregated


def export_to_drive(
    image: ee.Image, description: str, folder: str, region: ee.Geometry
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        crs="EPSG:4326",
        scale=EXPORT_SCALE_DEG * 111320,
        region=region,
        maxPixels=int(1e11),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "global"], default="test")
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--drive-folder", default="plant_traits_earth_embeddings")
    args = parser.parse_args()

    ee.Authenticate()
    ee.Initialize(project="plant-traits-earth-v2")

    embedding = get_embedding(args.year)

    if args.mode == "test":
        # Central Europe test region (roughly Alps + Germany, ~600x400 km)
        region = ee.Geometry.Rectangle([5.0, 44.0, 17.0, 52.0])
        description = f"satellite_embedding_v1_{args.year}_wgs84_test"
    else:
        region = ee.Geometry.Rectangle(
            [-180, -90, 180, 90], proj="EPSG:4326", evenOdd=False
        )
        description = f"satellite_embedding_v1_{args.year}_wgs84_global"

    image = aggregate_to_1km(embedding, region=region)
    task = export_to_drive(image, description, args.drive_folder, region)
    print(f"Started export: {description}")
    print(f"Task ID: {task.id}")
    print("Monitor at: https://code.earthengine.google.com/tasks")
    print("After download, run: python reproject_to_reference.py <downloaded_file.tif>")


if __name__ == "__main__":
    main()
