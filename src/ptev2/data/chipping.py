import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
import zarr
from rasterio.features import rasterize
from shapely.geometry import Point
from tqdm import tqdm

BUFFER_SIZE = 512
SPLIT_ENCODING = {"train": 0, "val": 1, "test": 2}  # -1 = unknown


def compute_split_labels(
    ref_tif: Path,
    patch_size: int,
    stride: int,
    h3_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """Assign a split label to every chip extracted from a raster.

    Chips are defined by sliding a window of size `patch_size` across the raster
    with a given `stride`. Each chip is assigned the split of the H3 cell that
    contains its center point. Chips whose center falls outside all H3 cells are
    labeled -1 (unknown).

    Args:
        ref_tif: Path to a reference GeoTIFF defining the raster grid (CRS,
            transform, and dimensions). Only metadata is read, not pixel data.
        patch_size: Side length of each chip in pixels.
        stride: Step size between consecutive chips in pixels.
        h3_gdf: GeoDataFrame of H3 hexagonal cells with a "split" column
            containing split name string labels.

    Returns:
        Int8 array of shape (n_chips,) with values from SPLIT_ENCODING, or -1
        for chips not covered by any H3 cell. Ordered row-major (left-to-right,
        top-to-bottom).
    """
    # Read the raster grid metadata from the reference TIF
    with rasterio.open(ref_tif) as src:
        transform = src.transform
        height, width = src.height, src.width

    # Compute the number of chips along each axis
    n_cols = math.ceil((width - patch_size) / stride) + 1
    n_rows = math.ceil((height - patch_size) / stride) + 1

    # Compute the geographic coordinates of each chip's center point.
    # transform.c / transform.f are the top-left corner coordinates,
    # transform.a / transform.e are the pixel width and height (e is negative).
    xs, ys = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            cx = transform.c + (col * stride + patch_size / 2) * transform.a
            cy = transform.f + (row * stride + patch_size / 2) * transform.e
            xs.append(cx)
            ys.append(cy)

    # Spatial join: find which H3 cell each chip center falls within
    centers = gpd.GeoDataFrame(
        {"chip_idx": range(len(xs))},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=h3_gdf.crs,
    )
    joined = gpd.sjoin(
        centers, h3_gdf[["split", "geometry"]], how="left", predicate="within"
    )

    # Map split names to integer codes; chips with no matching H3 cell stay -1
    labels = np.full(len(xs), -1, dtype=np.int8)
    matched = joined["split"].notna()
    labels[joined.loc[matched, "chip_idx"].values] = (
        joined.loc[matched, "split"]
        .map(SPLIT_ENCODING)
        .fillna(-1)
        .astype(np.int8)
        .values
    )

    return labels


def compute_pixel_split_mask(ref_tif: Path, h3_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Rasterize the H3 split grid onto the reference raster grid.

    Projects H3 hexagonal cells onto the pixel grid of the reference raster,
    burning each cell's split code into the corresponding pixels. Used during
    chipping to mask out pixels that belong to a different split than the chip.

    Args:
        ref_tif: Path to a reference GeoTIFF defining the target raster grid
            (CRS, transform, and dimensions).
        h3_gdf: GeoDataFrame of H3 hexagonal cells with a "split" column
            containing split name string labels.

    Returns:
        Int8 array of shape (height, width) with values from SPLIT_ENCODING,
        or -1 where no H3 cell covers the pixel.
    """
    with rasterio.open(ref_tif) as src:
        transform = src.transform
        height, width = src.height, src.width
        h3_proj = h3_gdf.to_crs(src.crs)

    split_codes = h3_proj["split"].map(SPLIT_ENCODING).fillna(-1).astype(int)
    shapes = [
        (geom, int(code))
        for geom, code in zip(h3_proj.geometry, split_codes)
        if geom is not None
    ]
    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype=np.int8,
    )


def _init_zarr_store(
    path: Path,
    n_chips: int,
    split_name: str,
    predictors: dict,
    targets: dict,
    all_srcs: dict,
    patch_size: int,
    stride: int,
    crs,
    transform,
) -> tuple[zarr.Group, dict, zarr.Array]:
    """Create and pre-allocate a zarr store for a single split.

    Opens a new zarr group at `path`, writes raster metadata as group
    attributes, and pre-allocates one array per predictor and target of shape
    (n_chips, n_bands, patch_size, patch_size). Also allocates a bounds array
    of shape (n_chips, 4) for storing chip bounding boxes.

    Args:
        path: Output path for the zarr store (e.g. output_dir/train.zarr).
        n_chips: Number of chips that will be written to this split.
        split_name: Name of the split (e.g. "train"), stored as an attribute.
        predictors: Dict mapping predictor name to list of source TIF paths.
        targets: Dict mapping target name to list of source TIF paths.
        all_srcs: Dict mapping name to list of open rasterio datasets.
        patch_size: Side length of each chip in pixels.
        stride: Step size between consecutive chips in pixels.
        crs: Coordinate reference system of the raster grid.
        transform: Affine transform of the raster grid.

    Returns:
        Tuple of (zarr group, dict of pre-allocated arrays, bounds array).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.attrs["split"] = split_name
    store.attrs["crs_epsg"] = crs.to_epsg()
    store.attrs["res_km"] = [transform.a, abs(transform.e)]
    store.attrs["transform"] = list(transform)
    store.attrs["patch_size"] = patch_size
    store.attrs["stride"] = stride
    store.attrs["creation_date"] = datetime.now().isoformat(timespec="seconds")

    arrays = {}
    pred_group = store.require_group("predictors")
    tgt_group = store.require_group("targets")

    for name, group, group_name in [
        (n, pred_group, "predictors") for n in predictors
    ] + [(n, tgt_group, "targets") for n in targets]:
        srcs = all_srcs[name]
        n_bands = sum(src.count for src in srcs)
        arrays[name] = group.create_array(
            name,
            shape=(n_chips, n_bands, patch_size, patch_size),
            chunks=(1, n_bands, patch_size, patch_size),
            dtype="f4",
        )
        arrays[name].attrs["files"] = [Path(src.name).name for src in srcs]
        print(
            f"  {group_name}/{name}: {n_chips} × {n_bands} bands -> {arrays[name].shape}"
        )

    bounds_arr = store.create_array(
        "bounds", shape=(n_chips, 4), chunks=(1024, 4), dtype="f8"
    )

    return store, arrays, bounds_arr


def chip_rasters_to_zarr(
    predictors: dict[str, list[Path]],
    targets: dict[str, list[Path]],
    output_dir: Path,
    patch_size: int,
    stride: int,
    h3_file: Path,
) -> None:
    """Chip predictor and target rasters into one zarr store per split.

    Computes per-chip split labels and a pixel-level split mask from the H3
    grid, then slides a window of size `patch_size` across all rasters with
    the given `stride`, routing each chip to the zarr store of its split.
    Pixels that belong to a different split than the chip are set to NaN.

    Args:
        predictors: Dict mapping predictor name to list of source TIF paths.
        targets: Dict mapping target name to list of source TIF paths.
        output_dir: Directory where split zarr stores will be written.
        patch_size: Side length of each chip in pixels.
        stride: Step size between consecutive chips in pixels.
        h3_file: Path to a GeoPackage of H3 hexagonal cells with a "split"
            column used to assign each chip and pixel to a train/val/test split.
    """
    # get reference raster metadata
    all_srcs = {
        name: [rasterio.open(f) for f in files]
        for name, files in {**predictors, **targets}.items()
    }
    for name, srcs in all_srcs.items():
        if not srcs:
            raise ValueError(f"No files found for '{name}'. Check your data paths.")

    ref = next(iter(all_srcs.values()))[0]
    print(f"Reference dataset:\n  {Path(ref.name)}")

    height, width, crs, transform = ref.height, ref.width, ref.crs, ref.transform
    print("Reference grid")
    print(f"  Resolution:   {abs(transform.a):.2f} × {abs(transform.e):.2f}")
    print(f"  Height:       {height} px")
    print(f"  Width:        {width} px")
    print(f"  CRS:          EPSG:{crs.to_epsg()}")

    for name, srcs in all_srcs.items():
        for src in srcs:
            assert src.height == height and src.width == width, (
                f"{name}: shape mismatch ({src.height}x{src.width} vs {height}x{width})"
            )
            assert src.crs == crs, f"{name}: CRS mismatch"
            assert src.transform == transform, f"{name}: transform mismatch"
    print("All datasets match reference grid!")

    print()

    # compute split labels and pixel mask from H3 grid  TODO: rewrite with our own splits as soon as we have them
    print(f"H3 split cells:\n  {h3_file}")
    h3_gdf = gpd.read_file(h3_file)

    print("Computing chip-level split labels...")
    split_labels = compute_split_labels(ref.name, patch_size, stride, h3_gdf)
    print("Rasterizing H3 split grid to pixel mask...")
    pixel_split_mask = compute_pixel_split_mask(ref.name, h3_gdf)

    n_cols = math.ceil((width - patch_size) / stride) + 1
    n_rows = math.ceil((height - patch_size) / stride) + 1
    n_chips = n_rows * n_cols
    print(f"Chips: {n_rows} rows × {n_cols} cols = {n_chips} total\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    code_to_name = {code: name for name, code in SPLIT_ENCODING.items()}
    n_per_split = {
        name: int((split_labels == code).sum()) for name, code in SPLIT_ENCODING.items()
    }
    print("Chips per split:")
    for name, n in n_per_split.items():
        print(f"  {name}: {n:,}")
    unknown = int((split_labels == -1).sum())
    if unknown:
        print(f"  unknown (skipped): {unknown:,}")
    print()

    # Pre-allocate one zarr store per split
    split_arrays = {}
    split_bounds_arrs = {}
    for split_name, n_split in n_per_split.items():
        print(f"Initialising {split_name}.zarr ({n_split:,} chips)...")
        _, split_arrays[split_name], split_bounds_arrs[split_name] = _init_zarr_store(
            path=output_dir / f"{split_name}.zarr",
            n_chips=n_split,
            split_name=split_name,
            predictors=predictors,
            targets=targets,
            all_srcs=all_srcs,
            patch_size=patch_size,
            stride=stride,
            crs=crs,
            transform=transform,
        )

    # Per-split buffers
    split_bufs = {
        split_name: {
            name: np.empty(
                (
                    BUFFER_SIZE,
                    sum(s.count for s in all_srcs[name]),
                    patch_size,
                    patch_size,
                ),
                dtype=np.float32,
            )
            for name in all_srcs
        }
        for split_name in SPLIT_ENCODING
    }
    split_bounds_bufs = {
        split_name: np.empty((BUFFER_SIZE, 4), dtype=np.float64)
        for split_name in SPLIT_ENCODING
    }
    split_buf_pos = {split_name: 0 for split_name in SPLIT_ENCODING}
    split_buf_start = {split_name: 0 for split_name in SPLIT_ENCODING}

    def flush(split_name: str, count: int) -> None:
        start = split_buf_start[split_name]
        for name in all_srcs:
            split_arrays[split_name][name][start : start + count] = split_bufs[
                split_name
            ][name][:count]
        split_bounds_arrs[split_name][start : start + count] = split_bounds_bufs[
            split_name
        ][:count]

    print("\nChipping...")
    chip_global_idx = 0
    for row in tqdm(range(n_rows), desc="Rows"):
        for col in range(n_cols):
            split_code = int(split_labels[chip_global_idx])
            chip_global_idx += 1

            if split_code not in code_to_name:
                continue  # unknown — skip

            split_name = code_to_name[split_code]
            y_px, x_px = row * stride, col * stride
            window = rasterio.windows.Window(x_px, y_px, patch_size, patch_size)

            # Read chip
            chip_data = {}
            for name, srcs in all_srcs.items():
                chip_data[name] = np.concatenate(
                    [
                        src.read(window=window, boundless=True, fill_value=0).astype(
                            np.float32
                        )
                        for src in srcs
                    ],
                    axis=0,
                )

            # Mask pixels that belong to a different split
            if pixel_split_mask is not None:
                y_end = min(y_px + patch_size, height)
                x_end = min(x_px + patch_size, width)
                mask_chip = np.full((patch_size, patch_size), -1, dtype=np.int8)
                mask_chip[: y_end - y_px, : x_end - x_px] = pixel_split_mask[
                    y_px:y_end, x_px:x_end
                ]
                outside = mask_chip != split_code  # (patch_size, patch_size)
                for name in chip_data:
                    chip_data[name][:, outside] = np.nan

            buf_pos = split_buf_pos[split_name]
            for name in all_srcs:
                split_bufs[split_name][name][buf_pos] = chip_data[name]

            win_t = rasterio.windows.transform(window, transform)
            split_bounds_bufs[split_name][buf_pos] = [
                win_t.c,
                win_t.f + patch_size * transform.e,
                win_t.c + patch_size * transform.a,
                win_t.f,
            ]

            split_buf_pos[split_name] += 1
            if split_buf_pos[split_name] == BUFFER_SIZE:
                flush(split_name, BUFFER_SIZE)
                split_buf_start[split_name] += BUFFER_SIZE
                split_buf_pos[split_name] = 0

    # Final flush
    for split_name in SPLIT_ENCODING:
        pos = split_buf_pos[split_name]
        if pos > 0:
            flush(split_name, pos)

    for srcs in all_srcs.values():
        for src in srcs:
            src.close()

    print(f"\nDone. Splits written to {output_dir}")
    for split_name, n_split in n_per_split.items():
        print(f"  {split_name}.zarr: {n_split:,} chips")
