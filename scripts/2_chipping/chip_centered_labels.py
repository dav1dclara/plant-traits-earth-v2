"""Build centered chips with simple supervision views (sPlot, GBIF, combined)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import rasterio
import zarr
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track
from shapely.geometry import box

from ptev2.data.chipping import SPLIT_ENCODING, compute_pixel_split_mask

console = Console()
SPLITS = ("train", "val", "test")
SUPERVISION_DATASETS = (
    "supervision",  # combined: sPlot priority, else GBIF
    "supervision_splot_only",
    "supervision_gbif_only",
)


def _resolve_data_root(candidates: list[str], resolution_km: int) -> Path:
    for root in candidates:
        path = Path(root) / f"{resolution_km}km"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve data root from candidates: {candidates} (for {resolution_km}km)."
    )


def _trait_tif_path(root: Path, trait: str) -> Path:
    direct = root / f"{trait}.tif"
    if direct.exists():
        return direct
    prefixed = root / f"X{trait}.tif"
    if prefixed.exists():
        return prefixed
    return direct


def _read_raster_band(path: Path, band: int) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def _sample_band_matrix(
    paths: list[Path],
    bands: list[int],
    rows: np.ndarray,
    cols: np.ndarray,
    progress_desc: str,
) -> np.ndarray:
    if len(paths) != len(bands):
        raise ValueError("paths and bands must have the same length")
    out = np.full((rows.size, len(paths)), np.nan, dtype=np.float32)
    for i, (path, band) in enumerate(
        track(list(zip(paths, bands)), description=progress_desc)
    ):
        out[:, i] = _read_raster_band(path, int(band))[rows, cols]
    return out


def _resolve_band_indices(
    paths: list[Path], band_name: str, fallback_band: int
) -> list[int]:
    out: list[int] = []
    band_name = str(band_name).strip().lower()
    for path in paths:
        with rasterio.open(path) as src:
            descriptions = [
                str(d).strip().lower() if d is not None else ""
                for d in src.descriptions
            ]
            name_to_idx = {name: i + 1 for i, name in enumerate(descriptions) if name}
        out.append(int(name_to_idx.get(band_name, fallback_band)))
    return out


def _build_any_valid_mask(paths: list[Path], progress_desc: str) -> np.ndarray:
    any_valid = None
    for path in track(paths, description=progress_desc):
        valid = np.isfinite(_read_raster_band(path, 1))
        any_valid = valid if any_valid is None else (any_valid | valid)
    assert any_valid is not None
    return any_valid


def _load_predictor_stacks(
    predictors_dir: Path,
    predictor_groups: list[str],
) -> tuple[dict[str, np.ndarray], rasterio.Affine, int, int, object]:
    stacks: dict[str, np.ndarray] = {}
    ref_transform = None
    ref_height = None
    ref_width = None
    ref_crs = None

    for group in predictor_groups:
        files = sorted((predictors_dir / group).glob("*.tif"))
        if not files:
            raise FileNotFoundError(
                f"No predictor GeoTIFF files found in {predictors_dir / group}"
            )

        bands = []
        for path in files:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan

                if ref_transform is None:
                    ref_transform = src.transform
                    ref_height, ref_width = src.height, src.width
                    ref_crs = src.crs
                else:
                    if (src.height, src.width) != (ref_height, ref_width):
                        raise ValueError(f"Shape mismatch for predictor {path}.")
                    if src.transform != ref_transform:
                        raise ValueError(f"Transform mismatch for predictor {path}.")
                    if src.crs != ref_crs:
                        raise ValueError(f"CRS mismatch for predictor {path}.")
                bands.append(arr)
        stacks[group] = np.stack(bands, axis=0)

    assert (
        ref_transform is not None and ref_height is not None and ref_width is not None
    )
    assert ref_crs is not None
    return stacks, ref_transform, ref_height, ref_width, ref_crs


def _pack_supervision(
    mean_center: np.ndarray, source_center: np.ndarray, patch_size: int
) -> np.ndarray:
    n_traits = mean_center.shape[0]
    center = patch_size // 2
    out = np.full((n_traits * 2, patch_size, patch_size), np.nan, dtype=np.float32)
    out[1::2, :, :] = 0.0
    out[0::2, center, center] = mean_center.astype(np.float32)
    out[1::2, center, center] = source_center.astype(np.float32)
    return out


def _supervision_views(
    *,
    split_name: str,
    splot_mean: np.ndarray,
    splot_valid: np.ndarray,
    gbif_mean: np.ndarray,
    gbif_valid: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    use_gbif = (~splot_valid) & gbif_valid
    splot_view = (
        np.where(splot_valid, splot_mean, np.nan).astype(np.float32),
        np.where(splot_valid, 2.0, 0.0).astype(np.float32),
    )
    gbif_view = (
        np.where(gbif_valid, gbif_mean, np.nan).astype(np.float32),
        np.where(gbif_valid, 1.0, 0.0).astype(np.float32),
    )
    comb_view = (
        np.where(splot_valid, splot_mean, np.where(use_gbif, gbif_mean, np.nan)).astype(
            np.float32
        ),
        (2.0 * splot_valid.astype(np.float32) + 1.0 * use_gbif.astype(np.float32)),
    )

    views = {
        "supervision": comb_view,
        "supervision_splot_only": splot_view,
        "supervision_gbif_only": gbif_view,
    }
    return views


def _chip_bounds(
    row: int, col: int, patch_size: int, transform: rasterio.Affine
) -> tuple[float, float, float, float]:
    half = patch_size // 2
    row0, row1 = row - half, row + half + 1
    col0, col1 = col - half, col + half + 1
    min_x, max_y = transform * (col0, row0)
    max_x, min_y = transform * (col1, row1)
    return float(min_x), float(min_y), float(max_x), float(max_y)


def _write_split_gpkg(
    out_path: Path,
    split_name: str,
    rows: np.ndarray,
    cols: np.ndarray,
    bounds: np.ndarray,
    raster_epsg: int,
    out_crs: str | None,
) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "chip_id": np.arange(bounds.shape[0], dtype=np.int64),
            "split": [split_name] * bounds.shape[0],
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
        },
        geometry=[box(*b) for b in bounds],
        crs=f"EPSG:{raster_epsg}",
    )
    if out_crs:
        gdf = gdf.to_crs(out_crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")


def _select_split_centers(
    split_mask: np.ndarray,
    splot_any: np.ndarray,
    gbif_only_any: np.ndarray,
    *,
    split_name: str,
    gbif_to_splot_ratio: float,
    rng: np.random.Generator,
    max_centers: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    split_pixels = split_mask == SPLIT_ENCODING[split_name]
    splot_rows, splot_cols = np.where(split_pixels & splot_any)
    rows, cols = splot_rows, splot_cols

    if split_name == "train":
        gbif_rows, gbif_cols = np.where(split_pixels & gbif_only_any)
        n_keep = min(gbif_rows.size, int(round(gbif_to_splot_ratio * splot_rows.size)))
        if n_keep > 0:
            idx = rng.choice(gbif_rows.size, size=n_keep, replace=False)
            rows = np.concatenate([splot_rows, gbif_rows[idx]])
            cols = np.concatenate([splot_cols, gbif_cols[idx]])

    if max_centers is not None and max_centers > 0 and rows.size > max_centers:
        idx = rng.choice(rows.size, size=max_centers, replace=False)
        rows, cols = rows[idx], cols[idx]

    return rows.astype(np.int32), cols.astype(np.int32), int(splot_rows.size)


@hydra.main(
    version_base=None,
    config_path="../../config/preprocessing",
    config_name="centered_chipping",
)
def main(cfg: DictConfig) -> None:
    resolution_km = int(cfg.settings.resolution_km)
    patch_size = int(cfg.settings.patch_size)
    stride = int(cfg.settings.stride)
    if patch_size % 2 == 0:
        raise ValueError("Centered chipping requires an odd patch_size.")
    pad = patch_size // 2

    if cfg.paths.get("data_root"):
        data_root = Path(str(cfg.paths.data_root)) / f"{resolution_km}km"
        if not data_root.exists():
            raise FileNotFoundError(f"Configured data_root does not exist: {data_root}")
    else:
        data_root = _resolve_data_root(
            [str(v) for v in cfg.paths.data_root_candidates], resolution_km
        )
    predictors_dir = data_root / str(cfg.paths.predictors_dirname)
    splot_dir = data_root / str(cfg.paths.targets_splot_dirname)
    gbif_dir = data_root / str(cfg.paths.targets_gbif_dirname)
    split_file = data_root / str(cfg.paths.splits_relpath)
    chips_dir = (
        data_root / str(cfg.paths.chips_subdir) / f"patch{patch_size}_stride{stride}"
    )
    chips_dir.mkdir(parents=True, exist_ok=True)

    predictor_groups = [
        str(name)
        for name, pred_cfg in cfg.data.predictors.items()
        if bool(pred_cfg.use)
    ]
    if not predictor_groups:
        raise ValueError("No predictor groups enabled.")

    traits = [str(v) for v in cfg.data.traits] if cfg.data.traits else []
    if not traits:
        traits = sorted(p.stem for p in splot_dir.glob("X*.tif"))
    if not traits:
        raise ValueError(f"No traits found in {splot_dir}")

    splot_paths = [_trait_tif_path(splot_dir, t) for t in traits]
    gbif_paths = [_trait_tif_path(gbif_dir, t) for t in traits]

    console.rule("[bold]CHIP CENTERED LABELS[/bold]")
    console.print(f"Data root: [cyan]{data_root}[/cyan]")
    console.print(f"Patch size: [cyan]{patch_size}[/cyan]")
    console.print(f"Predictor groups: [cyan]{predictor_groups}[/cyan]")
    console.print(f"Traits: [cyan]{len(traits)}[/cyan]")
    console.print(f"Split file: [cyan]{split_file}[/cyan]")
    console.print(f"Output dir: [cyan]{chips_dir}[/cyan]")

    predictor_stacks, transform, height, width, crs = _load_predictor_stacks(
        predictors_dir, predictor_groups
    )
    predictor_padded = {
        name: np.pad(
            arr,
            ((0, 0), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=np.nan,
        )
        for name, arr in predictor_stacks.items()
    }

    h3_gdf = gpd.read_file(split_file)
    split_mask = compute_pixel_split_mask(splot_paths[0], h3_gdf)
    splot_any = _build_any_valid_mask(splot_paths, progress_desc="Build validity splot")
    gbif_any = _build_any_valid_mask(gbif_paths, progress_desc="Build validity gbif")
    gbif_only_any = gbif_any & (~splot_any)

    rng = np.random.default_rng(int(cfg.settings.random_seed))
    ratio = float(cfg.settings.train_gbif_to_splot_ratio)
    export_split_gpkg = bool(cfg.settings.get("export_split_gpkg", False))
    export_split_gpkg_crs = cfg.settings.get("export_split_gpkg_crs")
    band_names = cfg.settings.get("target_band_names")
    splot_mean_bands = _resolve_band_indices(
        splot_paths, str(band_names.mean if band_names else "mean"), 1
    )
    splot_std_bands = _resolve_band_indices(
        splot_paths, str(band_names.std if band_names else "std"), 2
    )
    splot_q05_bands = _resolve_band_indices(
        splot_paths, str(band_names.q05 if band_names else "q05"), 4
    )
    splot_q95_bands = _resolve_band_indices(
        splot_paths, str(band_names.q95 if band_names else "q95"), 5
    )
    splot_count_bands = _resolve_band_indices(
        splot_paths, str(band_names.count if band_names else "count"), 6
    )
    gbif_mean_bands = _resolve_band_indices(
        gbif_paths, str(band_names.mean if band_names else "mean"), 1
    )

    split_coords: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in SPLITS:
        max_centers = cfg.settings.get(f"max_{split_name}_centers")
        max_centers = int(max_centers) if max_centers is not None else None
        rows, cols, n_splot_anchors = _select_split_centers(
            split_mask,
            splot_any,
            gbif_only_any,
            split_name=split_name,
            gbif_to_splot_ratio=ratio,
            rng=rng,
            max_centers=max_centers,
        )
        split_coords[split_name] = (rows, cols)
        console.print(
            f"Split {split_name}: [cyan]{rows.size}[/cyan] centers "
            f"(sPlot anchors: [cyan]{n_splot_anchors}[/cyan])"
        )

    for split_name in SPLITS:
        rows, cols = split_coords[split_name]
        n_chips = int(rows.size)
        if n_chips == 0:
            console.print(f"[yellow]Skipping {split_name}: no centers.[/yellow]")
            continue

        console.rule(f"[bold]{split_name.upper()}[/bold]")
        splot_mean = _sample_band_matrix(
            splot_paths, splot_mean_bands, rows, cols, "Sample splot mean"
        )
        splot_std = _sample_band_matrix(
            splot_paths, splot_std_bands, rows, cols, "Sample splot std"
        )
        splot_q05 = _sample_band_matrix(
            splot_paths, splot_q05_bands, rows, cols, "Sample splot q05"
        )
        splot_q95 = _sample_band_matrix(
            splot_paths, splot_q95_bands, rows, cols, "Sample splot q95"
        )
        splot_count = _sample_band_matrix(
            splot_paths, splot_count_bands, rows, cols, "Sample splot count"
        )
        gbif_mean = _sample_band_matrix(
            gbif_paths, gbif_mean_bands, rows, cols, "Sample gbif mean"
        )

        splot_valid = np.isfinite(splot_mean)
        gbif_valid = np.isfinite(gbif_mean)
        supervision_views = _supervision_views(
            split_name=split_name,
            splot_mean=splot_mean,
            splot_valid=splot_valid,
            gbif_mean=gbif_mean,
            gbif_valid=gbif_valid,
        )

        zarr_path = chips_dir / f"{split_name}.zarr"
        store = zarr.open_group(str(zarr_path), mode="w")
        store.attrs.update(
            {
                "split": split_name,
                "crs_epsg": int(crs.to_epsg()),
                "transform": list(transform),
                "raster_height": int(height),
                "raster_width": int(width),
                "res_km": [float(transform.a), float(abs(transform.e))],
                "patch_size": patch_size,
                "stride": stride,
                "creation_date": datetime.now().isoformat(timespec="seconds"),
                "chip_mode": "centered",
            }
        )

        pred_group = store.require_group("predictors")
        for name in predictor_groups:
            arr = pred_group.create_array(
                name,
                shape=(
                    n_chips,
                    predictor_stacks[name].shape[0],
                    patch_size,
                    patch_size,
                ),
                chunks=(64, predictor_stacks[name].shape[0], patch_size, patch_size),
                dtype="f4",
            )
            arr.attrs["files"] = [
                p.name for p in sorted((predictors_dir / name).glob("*.tif"))
            ]

        tgt_group = store.require_group("targets")
        tgt_group.attrs["band_names"] = ["mean", "source"]
        tgt_group.attrs["trait_names"] = traits
        supervision_arrays = {}
        for dataset_name in SUPERVISION_DATASETS:
            arr = tgt_group.create_array(
                dataset_name,
                shape=(n_chips, len(traits) * 2, patch_size, patch_size),
                chunks=(64, len(traits) * 2, patch_size, patch_size),
                dtype="f4",
            )
            arr.attrs["files"] = [f"{trait}.tif" for trait in traits]
            supervision_arrays[dataset_name] = arr

        center_group = store.require_group("center")
        center_group.create_array(
            "splot_mean", data=splot_mean.astype(np.float32), chunks=(256, len(traits))
        )
        center_group.create_array(
            "gbif_mean", data=gbif_mean.astype(np.float32), chunks=(256, len(traits))
        )
        center_group.create_array(
            "splot_valid", data=splot_valid.astype(np.uint8), chunks=(256, len(traits))
        )
        center_group.create_array(
            "gbif_valid", data=gbif_valid.astype(np.uint8), chunks=(256, len(traits))
        )
        center_group.create_array(
            "splot_std", data=splot_std.astype(np.float32), chunks=(256, len(traits))
        )
        center_group.create_array(
            "splot_q05", data=splot_q05.astype(np.float32), chunks=(256, len(traits))
        )
        center_group.create_array(
            "splot_q95", data=splot_q95.astype(np.float32), chunks=(256, len(traits))
        )
        center_group.create_array(
            "splot_count",
            data=splot_count.astype(np.float32),
            chunks=(256, len(traits)),
        )

        store.create_array("row", data=rows, chunks=(1024,))
        store.create_array("col", data=cols, chunks=(1024,))
        bounds_arr = store.create_array(
            "bounds", shape=(n_chips, 4), chunks=(1024, 4), dtype="f8"
        )

        for i in track(range(n_chips), description=f"Write {split_name} chips"):
            r, c = int(rows[i]), int(cols[i])
            for name in predictor_groups:
                pred_group[name][i] = predictor_padded[name][
                    :, r : r + patch_size, c : c + patch_size
                ].astype(np.float32)
            for dataset_name, (mean_center, source_center) in supervision_views.items():
                supervision_arrays[dataset_name][i] = _pack_supervision(
                    mean_center=mean_center[i],
                    source_center=source_center[i],
                    patch_size=patch_size,
                )
            bounds_arr[i] = _chip_bounds(r, c, patch_size, transform)

        console.print(
            f"Wrote [cyan]{n_chips}[/cyan] centered chips to [cyan]{zarr_path}[/cyan]"
        )
        if export_split_gpkg:
            out_gpkg = chips_dir / "gpkg" / f"{split_name}.gpkg"
            _write_split_gpkg(
                out_path=out_gpkg,
                split_name=split_name,
                rows=rows,
                cols=cols,
                bounds=bounds_arr[:],
                raster_epsg=int(crs.to_epsg()),
                out_crs=str(export_split_gpkg_crs) if export_split_gpkg_crs else None,
            )
            console.print(f"Wrote split GPKG: [cyan]{out_gpkg}[/cyan]")


if __name__ == "__main__":
    main()
