from pathlib import Path

import hydra
import numpy as np
import rasterio
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rasterio.windows import Window
from tqdm import tqdm

from ptev2.utils import (
    checkpoint_paths_from_cfg,
    run_name_from_cfg,
    seed_all,
)


def _list_predictor_files(cfg: DictConfig) -> list[Path]:
    predictors_root_cfg = cfg.prediction.predictors_dir
    if predictors_root_cfg:
        predictors_root = Path(str(predictors_root_cfg))
    else:
        predictors_root = Path(cfg.paths.data_root_dir) / "22km" / "eo_data_processed"

    selected_predictors = [
        name for name, pred_cfg in cfg.training.data.predictors.items() if pred_cfg.use
    ]
    if not selected_predictors:
        raise ValueError("No predictors enabled in training.data.predictors.")

    predictor_files: list[Path] = []
    for name in selected_predictors:
        files = sorted((predictors_root / name).glob("*.tif"))
        if not files:
            raise FileNotFoundError(
                f"No predictor .tif files found for '{name}' in {predictors_root / name}."
            )
        predictor_files.extend(files)

    return predictor_files


def _resolve_reference_tif(cfg: DictConfig) -> Path:
    explicit = cfg.prediction.reference_tif
    if explicit:
        ref = Path(str(explicit))
        if not ref.exists():
            raise FileNotFoundError(f"prediction.reference_tif does not exist: {ref}")
        return ref

    target_source = str(cfg.training.data.target.source)
    source_dir = Path(cfg.paths.data_root_dir) / "22km" / target_source
    candidates = sorted(source_dir.glob("X*.tif"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not find a default reference tif. "
        "Set +prediction.reference_tif=/absolute/path/to/reference.tif"
    )


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_override = cfg.prediction.checkpoint_path
    if checkpoint_override:
        return Path(str(checkpoint_override))
    best_path, last_path = checkpoint_paths_from_cfg(cfg)
    if best_path.exists():
        return best_path
    return last_path


def _infer_checkpoint_output_channels(
    state_dict: dict[str, torch.Tensor],
) -> int | None:
    for key, tensor in state_dict.items():
        if key.endswith("head.bias") and tensor.ndim == 1:
            return int(tensor.shape[0])
    for key, tensor in state_dict.items():
        if key.endswith("head.weight") and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _validate_grids(
    reference: rasterio.DatasetReader, predictor_files: list[Path]
) -> int:
    total_bands = 0
    ref_shape = (reference.height, reference.width)
    ref_crs = reference.crs
    ref_transform = reference.transform

    for path in predictor_files:
        with rasterio.open(path) as src:
            total_bands += int(src.count)
            if (src.height, src.width) != ref_shape:
                raise ValueError(
                    f"Shape mismatch for {path}: {(src.height, src.width)} vs {ref_shape}"
                )
            if src.crs != ref_crs:
                raise ValueError(f"CRS mismatch for {path}: {src.crs} vs {ref_crs}")
            if src.transform != ref_transform:
                raise ValueError(f"Transform mismatch for {path}")

    return total_bands


def _target_layout_from_cfg(
    cfg: DictConfig,
) -> tuple[int, list[str], int, list[str]]:
    target_cfg = cfg.training.data.target
    selected_statistics = [str(v) for v in target_cfg.selected_statistics]
    if not selected_statistics:
        raise ValueError("training.data.target.selected_statistics must not be empty.")

    selected_trait_ids = (
        [int(v) for v in target_cfg.trait_ids] if target_cfg.trait_ids else []
    )
    n_traits = (
        len(selected_trait_ids) if selected_trait_ids else int(target_cfg.n_traits)
    )
    if n_traits <= 0:
        raise ValueError("training.data.target.n_traits must be > 0.")

    if selected_trait_ids and len(selected_trait_ids) == n_traits:
        trait_labels = [f"X{tid}" for tid in selected_trait_ids]
    else:
        trait_labels = [f"trait_{idx:03d}" for idx in range(n_traits)]

    out_channels = n_traits * len(selected_statistics)
    return n_traits, selected_statistics, out_channels, trait_labels


def predict_global(cfg: DictConfig) -> Path:
    seed_all(cfg.training.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA ist erforderlich, aber nicht verfuegbar.")
    device = torch.device("cuda")

    n_traits, band_names, out_channels, trait_labels = _target_layout_from_cfg(cfg)

    patch_h = int(cfg.data.patch_h)
    patch_w = int(cfg.data.patch_w)
    if patch_h != patch_w:
        raise ValueError(f"Only square patches are supported, got {patch_h}x{patch_w}.")

    halo_cfg = cfg.prediction.halo
    min_safe_halo = patch_h // 2 + 1
    halo = int(halo_cfg) if halo_cfg is not None else min_safe_halo
    if halo < min_safe_halo:
        raise ValueError(
            f"prediction.halo={halo} is too small for patch size {patch_h}. "
            f"Use halo >= {min_safe_halo} to avoid seam artifacts."
        )
    tile_size = int(cfg.prediction.tile_size)
    if tile_size < 1:
        raise ValueError(f"prediction.tile_size must be >= 1, got {tile_size}")

    apply_predictor_valid_mask = bool(cfg.prediction.apply_predictor_valid_mask)
    apply_reference_mask = bool(cfg.prediction.apply_reference_mask)

    predictor_files = _list_predictor_files(cfg)
    reference_tif = _resolve_reference_tif(cfg)

    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Set +prediction.checkpoint_path=/absolute/path/model.pth"
        )

    with rasterio.open(reference_tif) as ref:
        ref_mask = None
        if apply_reference_mask:
            count_band_idx = (
                band_names.index("count") if "count" in band_names else None
            )
            if count_band_idx is not None:
                count_arr = ref.read(count_band_idx + 1).astype(np.float32)
                ref_mask = np.isfinite(count_arr) & (count_arr > 0)
            else:
                # Fallback: if no count semantics exist, use finite values of band 1.
                ref_mask = np.isfinite(ref.read(1).astype(np.float32))

        total_in_channels = _validate_grids(ref, predictor_files)
        if total_in_channels != int(cfg.data.in_channels):
            raise ValueError(
                "Predictor channel count mismatch: "
                f"cfg.data.in_channels={int(cfg.data.in_channels)} vs actual={total_in_channels}."
            )

        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]

        checkpoint_out = _infer_checkpoint_output_channels(state)
        if checkpoint_out is not None and checkpoint_out != out_channels:
            raise ValueError(
                "Checkpoint/model channel mismatch: "
                f"checkpoint_output_channels={checkpoint_out}, expected={out_channels}."
            )

        model = instantiate(cfg.models.active, out_channels=out_channels).to(device)
        model.load_state_dict(state)
        model.eval()

        height, width = ref.height, ref.width
        predictions = np.full((out_channels, height, width), np.nan, dtype=np.float32)

        n_rows = (height + tile_size - 1) // tile_size
        n_cols = (width + tile_size - 1) // tile_size
        n_tiles = n_rows * n_cols
        max_tiles_cfg = cfg.prediction.max_tiles
        max_tiles = int(max_tiles_cfg) if max_tiles_cfg is not None else n_tiles
        n_tiles_to_run = min(n_tiles, max_tiles)

        srcs = [rasterio.open(path) for path in predictor_files]
        try:
            with torch.no_grad():
                for tile_idx in tqdm(range(n_tiles_to_run), desc="Predicting tiles"):
                    row = tile_idx // n_cols
                    col = tile_idx % n_cols

                    y0 = row * tile_size
                    x0 = col * tile_size
                    core_h = min(tile_size, height - y0)
                    core_w = min(tile_size, width - x0)

                    read_window = Window(
                        col_off=x0 - halo,
                        row_off=y0 - halo,
                        width=core_w + 2 * halo,
                        height=core_h + 2 * halo,
                    )

                    x_parts: list[np.ndarray] = []
                    valid_parts: list[np.ndarray] = []
                    for src in srcs:
                        arr = src.read(
                            window=read_window,
                            boundless=True,
                            fill_value=0,
                        ).astype(np.float32)
                        x_parts.append(arr)

                        # Use dataset masks to distinguish true data from boundless/no-data fill.
                        band_valid = (
                            src.read_masks(
                                window=read_window,
                                boundless=True,
                            )
                            > 0
                        )
                        valid_parts.append(band_valid)

                    x_np = np.concatenate(x_parts, axis=0)
                    valid_core = np.concatenate(valid_parts, axis=0).any(axis=0)
                    x = (
                        torch.from_numpy(x_np)
                        .unsqueeze(0)
                        .to(device=device, dtype=torch.float32)
                    )
                    x = torch.nan_to_num(x)

                    y_pred = model(x).squeeze(0).detach().cpu().numpy()
                    y_core = y_pred[:, halo : halo + core_h, halo : halo + core_w]
                    valid_core = valid_core[halo : halo + core_h, halo : halo + core_w]
                    if apply_predictor_valid_mask:
                        y_core[:, ~valid_core] = np.nan
                    predictions[:, y0 : y0 + core_h, x0 : x0 + core_w] = y_core
        finally:
            for src in srcs:
                src.close()

        output_cfg = cfg.prediction.output_tif
        if output_cfg:
            output_tif = Path(str(output_cfg))
        else:
            output_tif = (
                Path(cfg.training.checkpoint.dir)
                / f"{run_name_from_cfg(cfg)}_global_prediction.tif"
            )
        output_tif.parent.mkdir(parents=True, exist_ok=True)

        profile = ref.profile.copy()
        profile.update(
            dtype="float32",
            count=out_channels,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
            predictor=3,
        )

        with rasterio.open(output_tif, "w", **profile) as dst:
            if apply_reference_mask and ref_mask is not None:
                predictions[:, ~ref_mask] = np.nan
            dst.write(predictions)

            band_idx = 1
            for trait_label in trait_labels:
                for semantic in band_names:
                    dst.set_band_description(band_idx, f"{trait_label}_{semantic}")
                    band_idx += 1

    print("Prediction finished")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  reference:  {reference_tif}")
    print(f"  output:     {output_tif}")
    print(f"  shape:      ({out_channels}, {height}, {width})")
    print(f"  n_traits:   {n_traits}")
    print(f"  band_order: {band_names}")
    print(
        "  masks:      "
        f"predictor_valid={apply_predictor_valid_mask}, reference={apply_reference_mask}"
    )

    return output_tif


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    predict_global(cfg)


if __name__ == "__main__":
    main()
