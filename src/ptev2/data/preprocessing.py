from pathlib import Path

import numpy as np
import rasterio


def _read_bands(path: Path, band_order: list[str]) -> np.ndarray:
    """Read raster bands reordered to match band_order. Returns (n_bands, H, W) float32."""
    with rasterio.open(path) as src:
        descriptions = [d.lower() for d in src.descriptions]
        data = src.read().astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    # Reorder bands to match band_order
    indices = [descriptions.index(b) for b in band_order]
    return data[indices]


def combine_traits(
    trait: str,
    gbif_dir: Path,
    splot_dir: Path,
    output_dir: Path,
    output_bands: list[str],
) -> None:
    gbif_path = gbif_dir / f"{trait}.tif"
    splot_path = splot_dir / f"{trait}.tif"

    gbif = _read_bands(gbif_path, output_bands)  # (n_bands, H, W)
    splot = _read_bands(splot_path, output_bands)  # (n_bands, H, W)

    gbif_valid = np.isfinite(gbif[0])  # (H, W)
    splot_valid = np.isfinite(splot[0])  # (H, W)

    # Source band: NaN=nodata, 1=GBIF only, 2=SPLOT (±GBIF)
    source = np.full(gbif.shape[1:], np.nan, dtype=np.float32)
    source[gbif_valid] = 1.0
    source[splot_valid] = 2.0

    # Merged output: start with GBIF, overwrite with SPLOT where available
    merged = gbif.copy()
    merged[:, ~gbif_valid] = np.nan
    merged[:, splot_valid] = splot[:, splot_valid]

    # Stack trait bands + source band
    output = np.concatenate([merged, source[np.newaxis]], axis=0)  # (n_bands+1, H, W)

    with rasterio.open(gbif_path) as ref:
        profile = ref.profile.copy()

    all_bands = output_bands + ["source"]
    profile.update(dtype=np.float32, count=len(all_bands), nodata=np.nan)

    out_path = output_dir / f"{trait}.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(output)
        dst.descriptions = tuple(all_bands)
