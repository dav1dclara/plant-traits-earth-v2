import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
import zarr
from rasterio.features import rasterize
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

BUFFER_SIZE = 2048

# Thread-local storage for rasterio file handles used during parallel strip reads.
# Each worker thread keeps its own open handle per path so concurrent prefetch reads
# never share a file handle (rasterio/GDAL file handles are not thread-safe).
_strip_reader_tls = threading.local()
SPLIT_ENCODING = {"train": 0, "val": 1, "test": 2, "none": 3}  # -1 = unknown


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
    overwrite: bool,
    target_bands: dict[str, list[int]] | None = None,
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
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing zarr store: {path}. "
            "Set settings.overwrite=true to allow replacing existing chips."
        )
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
    first_tgt_name = next(iter(targets))
    first_tgt_srcs = all_srcs[first_tgt_name]
    if target_bands and first_tgt_name in target_bands:
        bands_sel = target_bands[first_tgt_name]
        band_names = []
        for src in first_tgt_srcs:
            descs = list(src.descriptions)
            band_names.extend(descs[b - 1] or f"band{b}" for b in bands_sel)
        tgt_group.attrs["band_names"] = band_names
    else:
        band_names = []
        for src in first_tgt_srcs:
            band_names.extend(
                d or f"band{i + 1}" for i, d in enumerate(src.descriptions)
            )
        tgt_group.attrs["band_names"] = band_names

    for name, group, group_name in [
        (n, pred_group, "predictors") for n in predictors
    ] + [(n, tgt_group, "targets") for n in targets]:
        srcs = all_srcs[name]
        if name in targets and target_bands and name in target_bands:
            n_bands = len(srcs) * len(target_bands[name])
        else:
            n_bands = sum(src.count for src in srcs)
        arrays[name] = group.create_array(
            name,
            shape=(n_chips, n_bands, patch_size, patch_size),
            chunks=(1, n_bands, patch_size, patch_size),
            dtype="f4",
        )
        arrays[name].attrs["files"] = [Path(src.name).name for src in srcs]
        console.print(
            f"  [cyan]{group_name}/{name}[/cyan]: {n_chips} × {n_bands} bands "
            f"[dim]→ {arrays[name].shape}[/dim]"
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
    validity_mask_file: Path | None = None,
    mask_predictors_by_split: bool = True,
    mask_targets_by_split: bool = True,
    split_assignment: str = "any_overlap",
    min_split_pixels: int = 1,
    require_valid_target: bool = True,
    overwrite: bool = False,
    target_bands: dict[str, list[int]] | None = None,
) -> None:
    """Chip predictor and target rasters into one zarr store per split.

    Computes a pixel-level split mask from the H3 grid, then for each split
    slides a window of size `patch_size` with that split's stride. A chip is
    assigned to every split it overlaps with; pixels outside that split are
    set to NaN.

    Args:
        predictors: Dict mapping predictor name to list of source TIF paths.
        targets: Dict mapping target name to list of source TIF paths.
        output_dir: Directory where split zarr stores will be written.
        patch_size: Side length of each chip in pixels.
        stride_per_split: Step size per split (e.g. {"train": 10, "val": 10, "test": 15}).
        h3_file: Path to a GeoPackage of H3 hexagonal cells with a "split"
            column used to assign each chip and pixel to a train/val/test split.
    """
    all_srcs = {
        name: [rasterio.open(f) for f in files]
        for name, files in {**predictors, **targets}.items()
    }
    predictor_names = set(predictors.keys())
    target_names = set(targets.keys())
    for name, srcs in all_srcs.items():
        if not srcs:
            raise ValueError(f"No files found for '{name}'. Check your data paths.")

    ref = next(iter(all_srcs.values()))[0]
    console.print(f"Reference dataset:\n  [cyan]{Path(ref.name)}[/cyan]")

    height, width, crs, transform = ref.height, ref.width, ref.crs, ref.transform
    console.print("Reference grid")
    console.print(
        f"  Resolution:   [cyan]{abs(transform.a):.2f} × {abs(transform.e):.2f}[/cyan]"
    )
    console.print(f"  Height:       [cyan]{height} px[/cyan]")
    console.print(f"  Width:        [cyan]{width} px[/cyan]")
    console.print(f"  CRS:          [cyan]EPSG:{crs.to_epsg()}[/cyan]")

    for name, srcs in all_srcs.items():
        for src in srcs:
            if src.height != height or src.width != width:
                raise ValueError(
                    f"{name}: shape mismatch ({src.height}x{src.width} vs {height}x{width})"
                )
            if src.crs != crs:
                raise ValueError(f"{name}: CRS mismatch ({src.crs} vs {crs})")
            if src.transform != transform:
                raise ValueError(
                    f"{name}: transform mismatch ({src.transform} vs {transform})"
                )
    console.print("[green]All datasets match reference grid![/green]\n")

    with console.status(f"Reading H3 split cells from [cyan]{h3_file.name}[/cyan]..."):
        h3_gdf = gpd.read_file(h3_file)
    console.print(
        f"[green]✓[/green] Loaded {len(h3_gdf):,} H3 cells from [cyan]{h3_file}[/cyan]"
    )

    with console.status("Rasterizing H3 split grid to pixel mask..."):
        pixel_split_mask = compute_pixel_split_mask(ref.name, h3_gdf)
    console.print("[green]✓[/green] Pixel split mask ready")

    if save_all and validity_mask_file is not None:
        with console.status(
            f"Loading validity mask from [cyan]{validity_mask_file.name}[/cyan]..."
        ):
            with rasterio.open(validity_mask_file) as vm:
                predictor_valid_mask = vm.read(1).astype(bool)
        console.print(
            f"[green]✓[/green] Validity mask ready ({predictor_valid_mask.sum():,} valid pixels)"
        )
    else:
        predictor_valid_mask = None

    output_dir.mkdir(parents=True, exist_ok=True)
    code_to_name = {code: name for name, code in SPLIT_ENCODING.items()}

    # Group splits by stride so we do one pass per unique stride.
    # Splits absent from stride_per_split (e.g. "none") are burned into the
    # pixel mask for the "all" zarr but don't produce their own zarr.
    stride_groups: dict[int, list[str]] = {}
    for split_name in SPLIT_ENCODING:
        if split_name in stride_per_split:
            stride_groups.setdefault(stride_per_split[split_name], []).append(
                split_name
            )

    # "all" gets its own stride entry so it runs in a separate pass
    _stride_all = (
        stride_all if stride_all is not None else min(stride_per_split.values())
    )
    if save_all:
        stride_groups.setdefault(_stride_all, [])
        if "all" not in stride_groups[_stride_all]:
            stride_groups[_stride_all].append("all")

    valid_assignment_modes = {"center", "any_overlap"}
    if split_assignment not in valid_assignment_modes:
        raise ValueError(
            f"split_assignment must be one of {sorted(valid_assignment_modes)}, got '{split_assignment}'"
        )
    if min_split_pixels < 1:
        raise ValueError(f"min_split_pixels must be >= 1, got {min_split_pixels}")

    for stride, group_splits in stride_groups.items():
        n_cols = math.ceil((width - patch_size) / stride) + 1
        n_rows = math.ceil((height - patch_size) / stride) + 1
        console.print(
            f"\nStride=[cyan]{stride}[/cyan] ({', '.join(group_splits)}), "
            f"grid=[cyan]{n_rows}×{n_cols}[/cyan]"
        )

        named_splits = [s for s in group_splits if s != "all"]
        group_codes = {SPLIT_ENCODING[s] for s in named_splits}
        n_chips_per = {s: 0 for s in group_splits}

        # Build per-code overlap grids row-by-row using a 1D cumsum on each strip.
        # Resulting grids are (n_rows, n_cols) booleans — O(1) lookup during chipping.
        # Peak memory is one strip of the split mask at a time, not the full raster.
        x_starts = np.arange(n_cols) * stride
        x_ends = np.minimum(x_starts + patch_size, width)

        overlap_grids: dict[int, np.ndarray] = {
            code: np.zeros((n_rows, n_cols), dtype=bool) for code in group_codes
        }
        if "all" in group_splits:
            overlap_grids[-1] = np.zeros((n_rows, n_cols), dtype=bool)

        with console.status("Counting chips per split..."):
            for row in range(n_rows):
                y_px = row * stride
                y_end = min(y_px + patch_size, height)
                strip = pixel_split_mask[y_px:y_end, :]  # (h, W)

                for code in group_codes:
                    col_has = (strip == code).any(axis=0).astype(np.int32)
                    cs = np.zeros(width + 1, dtype=np.int32)
                    cs[1:] = np.cumsum(col_has)
                    overlap_grids[code][row] = (cs[x_ends] - cs[x_starts]) > 0

                if "all" in group_splits:
                    if predictor_valid_mask is not None:
                        valid_strip = predictor_valid_mask[y_px:y_end, :]
                    else:
                        valid_strip = strip >= 0
                    col_has_valid = valid_strip.any(axis=0).astype(np.int32)
                    cs_valid = np.zeros(width + 1, dtype=np.int32)
                    cs_valid[1:] = np.cumsum(col_has_valid)
                    overlap_grids[-1][row] = (cs_valid[x_ends] - cs_valid[x_starts]) > 0

        for code in group_codes:
            n_chips_per[code_to_name[code]] = int(np.sum(overlap_grids[code]))
        if "all" in group_splits:
            n_chips_per["all"] = int(np.sum(overlap_grids[-1]))

        # Allocate zarr stores and buffers for each split in this group
        split_arrays = {}
        split_bounds_arrs = {}
        for split_name in group_splits:
            if n_chips_per[split_name] == 0:
                continue
            console.print(
                f"[bold]{split_name}[/bold]: {n_chips_per[split_name]:,} chips"
            )
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
                    overwrite=overwrite,
                    target_bands=target_bands,
                )
            )

        bufs = {
            split_name: {
                name: np.empty(
                    (
                        BUFFER_SIZE,
                        len(all_srcs[name]) * len(target_bands[name])
                        if (
                            name in target_names
                            and target_bands
                            and name in target_bands
                        )
                        else sum(s.count for s in all_srcs[name]),
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
        written_count = {s: 0 for s in group_splits}
        skipped_invalid_target = {s: 0 for s in group_splits}

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
        # Use path strings (not open handles) so each worker thread can maintain its
        # own handle via _strip_reader_tls — concurrent prefetch reads are then safe.
        # Attach per-file band selection so each worker reads only the needed bands.
        # None means "read all bands" (predictors); a list means specific 1-indexed bands.
        all_paths_flat = [
            (
                name,
                str(src.name),
                list(target_bands[name])
                if (name in target_names and target_bands and name in target_bands)
                else None,
            )
            for name, srcs in all_srcs.items()
            for src in srcs
        ]
        n_files = len(all_paths_flat)

        def _read_strip(args: tuple) -> np.ndarray:
            path_str, window, bands = args
            if not hasattr(_strip_reader_tls, "files"):
                _strip_reader_tls.files = {}
            if path_str not in _strip_reader_tls.files:
                _strip_reader_tls.files[path_str] = rasterio.open(path_str)
            data = (
                _strip_reader_tls.files[path_str]
                .read(indexes=bands, window=window, boundless=True, masked=True)
                .astype(np.float32)
            )
            return data.filled(np.nan)

        # Precompute chip bounds for all (row, col) pairs (avoids per-chip rasterio calls).
        # Assumes no raster rotation (transform.b == transform.d == 0), which holds for
        # standard equal-area projections like EPSG:6933.
        _x_starts = np.arange(n_cols) * stride
        _y_starts = np.arange(n_rows) * stride
        _x_min = transform.c + _x_starts * transform.a
        _y_max = transform.f + _y_starts * transform.e
        _x_max = _x_min + patch_size * transform.a
        _y_min = _y_max + patch_size * transform.e
        all_bounds = np.empty((n_rows, n_cols, 4), dtype=np.float64)
        all_bounds[:, :, 0] = _x_min[np.newaxis, :]
        all_bounds[:, :, 1] = _y_min[:, np.newaxis]
        all_bounds[:, :, 2] = _x_max[np.newaxis, :]
        all_bounds[:, :, 3] = _y_max[:, np.newaxis]

        # Reusable mask buffer — avoids a new allocation per chip at edge rows/cols.
        mask_chip_buf = np.full((patch_size, patch_size), -1, dtype=np.int8)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        with progress:
            task = progress.add_task(f"Chipping stride={stride}", total=n_rows)
            with ThreadPoolExecutor(max_workers=min(32, n_files)) as executor:

                def _submit_strip(row: int) -> list:
                    y_px = row * stride
                    window = rasterio.windows.Window(0, y_px, strip_width, patch_size)
                    return [
                        executor.submit(_read_strip, (path_str, window, bands))
                        for _, path_str, bands in all_paths_flat
                    ]

                # Prefetch row 0 before entering the loop
                strip_futures = _submit_strip(0)

                for row in range(n_rows):
                    y_px = row * stride

                    # Submit next strip's reads before blocking on the current one,
                    # overlapping I/O with the column-processing loop below.
                    if row + 1 < n_rows:
                        next_futures = _submit_strip(row + 1)

                    strips_flat = [f.result() for f in strip_futures]
                    if row + 1 < n_rows:
                        strip_futures = next_futures

                    # Reconstruct name -> strip (n_bands, patch_size, strip_width)
                    strips: dict[str, np.ndarray] = {}
                    i = 0
                    for name, srcs in all_srcs.items():
                        n = len(srcs)
                        strips[name] = (
                            strips_flat[i]
                            if n == 1
                            else np.concatenate(strips_flat[i : i + n], axis=0)
                        )
                        i += n

                    for col in range(n_cols):
                        x_px = col * stride
                        y_end = min(y_px + patch_size, height)
                        x_end = min(x_px + patch_size, width)

                        # For interior chips use a direct view; edge chips need padding.
                        h_slice = y_end - y_px
                        w_slice = x_end - x_px
                        if h_slice == patch_size and w_slice == patch_size:
                            mask_chip = pixel_split_mask[y_px:y_end, x_px:x_end]
                        else:
                            mask_chip_buf[:] = -1
                            mask_chip_buf[:h_slice, :w_slice] = pixel_split_mask[
                                y_px:y_end, x_px:x_end
                            ]
                            mask_chip = mask_chip_buf

                        chip_splits = [
                            code_to_name[c]
                            for c in group_codes
                            if overlap_grids[c][row, col]
                        ]
                        if "all" in group_splits and overlap_grids[-1][row, col]:
                            chip_splits.append("all")
                        if not chip_splits:
                            continue

                        for split_name in chip_splits:
                            if require_valid_target and split_name != "all":
                                if split_name == "all":
                                    valid_region = mask_chip >= 0
                                else:
                                    valid_region = (
                                        mask_chip == SPLIT_ENCODING[split_name]
                                    )
                                has_valid_target = False
                                if bool(valid_region.any()):
                                    for tname in target_names:
                                        tchip = strips[tname][
                                            :, :, x_px : x_px + patch_size
                                        ]
                                        if np.isfinite(tchip[:, valid_region]).any():
                                            has_valid_target = True
                                            break
                                if not has_valid_target:
                                    skipped_invalid_target[split_name] += 1
                                    continue

                            pos = buf_pos[split_name]
                            for name in all_srcs:
                                # Write directly into the buffer — no intermediate copy.
                                buf_slice = bufs[split_name][name][pos]
                                np.copyto(
                                    buf_slice,
                                    strips[name][:, :, x_px : x_px + patch_size],
                                )

                                should_mask = (
                                    name in predictor_names and mask_predictors_by_split
                                ) or (name in target_names and mask_targets_by_split)

                                if should_mask:
                                    if split_name == "all":
                                        # When using a predictor validity mask, the
                                        # rasterio boundless read already fills nodata
                                        # with NaN — no extra masking needed. Fall back
                                        # to the H3 mask only if no validity mask was
                                        # provided (legacy behaviour).
                                        if predictor_valid_mask is None:
                                            buf_slice[:, mask_chip < 0] = np.nan
                                    else:
                                        split_code = SPLIT_ENCODING[split_name]
                                        buf_slice[:, mask_chip != split_code] = np.nan

                            bounds_bufs[split_name][pos] = all_bounds[row, col]
                            buf_pos[split_name] += 1
                            written_count[split_name] += 1
                            if buf_pos[split_name] == BUFFER_SIZE:
                                flush(split_name, BUFFER_SIZE)
                                buf_start[split_name] += BUFFER_SIZE
                                buf_pos[split_name] = 0

                    progress.advance(task)

        for split_name in group_splits:
            if buf_pos[split_name] > 0:
                flush(split_name, buf_pos[split_name])
            if split_name in split_arrays:
                n_written = int(written_count[split_name])
                for name in all_srcs:
                    arr = split_arrays[split_name][name]
                    arr.resize((n_written,) + arr.shape[1:])
                split_bounds_arrs[split_name].resize((n_written, 4))
                if require_valid_target:
                    console.print(
                        f"  [bold]{split_name}[/bold]: kept {n_written:,}, "
                        f"skipped {int(skipped_invalid_target[split_name]):,} invalid-target chips"
                    )

    for srcs in all_srcs.values():
        for src in srcs:
            src.close()

    console.print(f"\n[green]Done.[/green] Splits written to [cyan]{output_dir}[/cyan]")
