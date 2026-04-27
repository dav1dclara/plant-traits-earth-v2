import math
from concurrent.futures import ThreadPoolExecutor
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
    raster_height: int,
    raster_width: int,
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
    store.attrs["raster_height"] = raster_height
    store.attrs["raster_width"] = raster_width
    store.attrs["creation_date"] = datetime.now().isoformat(timespec="seconds")

    arrays = {}
    pred_group = store.require_group("predictors")
    tgt_group = store.require_group("targets")

    # Store band names once on the targets group (all targets share the same band order)
    first_tgt_srcs = next(iter(all_srcs[n] for n in targets))
    tgt_group.attrs["band_names"] = list(first_tgt_srcs[0].descriptions)

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
    stride_per_split: dict[str, int],
    h3_file: Path,
    save_all: bool = False,
    stride_all: int | None = None,
) -> None:
    """Chip predictor and target rasters into one zarr store per split.

    Computes a pixel-level split mask from the H3 grid, then for each split
    slides a window of size `patch_size` with that split's stride. Each chip is
    assigned to exactly one split based on its center pixel. Pixels from other
    splits inside that chip are set to NaN.

    Args:
        predictors: Dict mapping predictor name to list of source TIF paths.
        targets: Dict mapping target name to list of source TIF paths.
        output_dir: Directory where split zarr stores will be written.
        patch_size: Side length of each chip in pixels.
        stride_per_split: Step size per split (e.g. {"train": 10, "val": 10, "test": 15}).
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

    print(f"H3 split cells:\n  {h3_file}")
    h3_gdf = gpd.read_file(h3_file)

    print("Rasterizing H3 split grid to pixel mask...")
    pixel_split_mask = compute_pixel_split_mask(ref.name, h3_gdf)

    output_dir.mkdir(parents=True, exist_ok=True)
    code_to_name = {code: name for name, code in SPLIT_ENCODING.items()}

    # Group splits by stride so we do one pass per unique stride
    stride_groups: dict[int, list[str]] = {}
    for split_name in SPLIT_ENCODING:
        stride_groups.setdefault(stride_per_split[split_name], []).append(split_name)

    # "all" gets its own stride entry so it runs in a separate pass
    _stride_all = (
        stride_all if stride_all is not None else min(stride_per_split.values())
    )
    if save_all:
        stride_groups.setdefault(_stride_all, [])
        if "all" not in stride_groups[_stride_all]:
            stride_groups[_stride_all].append("all")

    for stride, group_splits in stride_groups.items():
        n_cols = math.ceil((width - patch_size) / stride) + 1
        n_rows = math.ceil((height - patch_size) / stride) + 1
        print(f"\nStride={stride} ({', '.join(group_splits)}), grid={n_rows}×{n_cols}")

        # Count chips per split in this group
        named_splits = [s for s in group_splits if s != "all"]
        group_codes = {SPLIT_ENCODING[s] for s in named_splits}
        n_chips_per = {s: 0 for s in group_splits}
        center_row = patch_size // 2
        center_col = patch_size // 2
        for row in range(n_rows):
            for col in range(n_cols):
                y_px, x_px = row * stride, col * stride
                y_end = min(y_px + patch_size, height)
                x_end = min(x_px + patch_size, width)
                mask_chip = np.full((patch_size, patch_size), -1, dtype=np.int8)
                mask_chip[: y_end - y_px, : x_end - x_px] = pixel_split_mask[
                    y_px:y_end, x_px:x_end
                ]
                center_code = int(mask_chip[center_row, center_col])
                if center_code in group_codes:
                    n_chips_per[code_to_name[center_code]] += 1
                unique_codes = np.unique(mask_chip)
                if "all" in group_splits and any(c >= 0 for c in unique_codes):
                    n_chips_per["all"] += 1
        for s, n in n_chips_per.items():
            print(f"  {s}: {n:,} chips")

        # Allocate zarr stores and buffers for each split in this group
        split_arrays = {}
        split_bounds_arrs = {}
        for split_name in group_splits:
            if n_chips_per[split_name] == 0:
                continue
            _, split_arrays[split_name], split_bounds_arrs[split_name] = (
                _init_zarr_store(
                    path=output_dir / f"{split_name}.zarr",
                    n_chips=n_chips_per[split_name],
                    split_name=split_name,
                    predictors=predictors,
                    targets=targets,
                    all_srcs=all_srcs,
                    patch_size=patch_size,
                    stride=stride,
                    crs=crs,
                    transform=transform,
                    raster_height=height,
                    raster_width=width,
                )
            )

        bufs = {
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
            for split_name in group_splits
        }
        bounds_bufs = {
            s: np.empty((BUFFER_SIZE, 4), dtype=np.float64) for s in group_splits
        }
        buf_pos = {s: 0 for s in group_splits}
        buf_start = {s: 0 for s in group_splits}

        def flush(split_name: str, count: int) -> None:
            start = buf_start[split_name]
            for name in all_srcs:
                split_arrays[split_name][name][start : start + count] = bufs[
                    split_name
                ][name][:count]
            split_bounds_arrs[split_name][start : start + count] = bounds_bufs[
                split_name
            ][:count]

        # Read one full-width strip per chip row; chips are sliced from it in memory.
        # All source files for a strip are read in parallel with ThreadPoolExecutor.
        strip_width = (n_cols - 1) * stride + patch_size
        all_srcs_flat = [(name, src) for name, srcs in all_srcs.items() for src in srcs]
        n_files = len(all_srcs_flat)

        def _read_strip(args: tuple) -> np.ndarray:
            src, window = args
            return src.read(window=window, boundless=True, fill_value=0).astype(
                np.float32
            )

        print("  Chipping...")
        with ThreadPoolExecutor(max_workers=min(32, n_files)) as executor:
            for row in tqdm(range(n_rows), desc=f"  stride={stride}"):
                y_px = row * stride
                strip_window = rasterio.windows.Window(0, y_px, strip_width, patch_size)

                # Read all source files for this strip in parallel
                strips_flat = list(
                    executor.map(
                        _read_strip, [(src, strip_window) for _, src in all_srcs_flat]
                    )
                )

                # Reconstruct name -> strip (n_bands, patch_size, strip_width)
                strips: dict[str, np.ndarray] = {}
                i = 0
                for name, srcs in all_srcs.items():
                    strips[name] = np.concatenate(
                        strips_flat[i : i + len(srcs)], axis=0
                    )
                    i += len(srcs)

                for col in range(n_cols):
                    x_px = col * stride
                    y_end = min(y_px + patch_size, height)
                    x_end = min(x_px + patch_size, width)
                    mask_chip = np.full((patch_size, patch_size), -1, dtype=np.int8)
                    mask_chip[: y_end - y_px, : x_end - x_px] = pixel_split_mask[
                        y_px:y_end, x_px:x_end
                    ]

                    center_code = int(mask_chip[center_row, center_col])
                    unique_codes = np.unique(mask_chip)
                    chip_splits = []
                    if center_code in group_codes:
                        chip_splits.append(code_to_name[center_code])
                    has_any_valid = any(c >= 0 for c in unique_codes)
                    if "all" in group_splits and has_any_valid:
                        chip_splits.append("all")
                    if not chip_splits:
                        continue

                    # Slice chip from the in-memory strip
                    chip_data = {
                        name: strip[:, :, x_px : x_px + patch_size].copy()
                        for name, strip in strips.items()
                    }

                    window = rasterio.windows.Window(x_px, y_px, patch_size, patch_size)
                    win_t = rasterio.windows.transform(window, transform)
                    bounds = [
                        win_t.c,
                        win_t.f + patch_size * transform.e,
                        win_t.c + patch_size * transform.a,
                        win_t.f,
                    ]

                    for split_name in chip_splits:
                        pos = buf_pos[split_name]
                        for name in all_srcs:
                            chip = chip_data[name].copy()
                            if split_name == "all":
                                chip[:, mask_chip < 0] = np.nan
                            else:
                                split_code = SPLIT_ENCODING[split_name]
                                chip[:, mask_chip != split_code] = np.nan
                            bufs[split_name][name][pos] = chip
                        bounds_bufs[split_name][pos] = bounds
                        buf_pos[split_name] += 1
                        if buf_pos[split_name] == BUFFER_SIZE:
                            flush(split_name, BUFFER_SIZE)
                            buf_start[split_name] += BUFFER_SIZE
                            buf_pos[split_name] = 0

        for split_name in group_splits:
            if buf_pos[split_name] > 0:
                flush(split_name, buf_pos[split_name])

    for srcs in all_srcs.values():
        for src in srcs:
            src.close()

    print(f"\nDone. Splits written to {output_dir}")
