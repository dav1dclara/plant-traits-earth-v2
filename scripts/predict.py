from pathlib import Path

import hydra
import numpy as np
import rasterio
import torch
import zarr
from affine import Affine
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rasterio.crs import CRS
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader

from ptev2.data.dataloader import PlantTraitDataset

console = Console()


@hydra.main(
    config_path="../config/prediction/", config_name="default", version_base=None
)
def main(cfg: DictConfig) -> None:
    console.rule("[bold cyan]PREDICTION[/bold cyan]")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"Device: [cyan]{device}[/cyan]")

    # Fetch training config and checkpoint path from W&B
    assert cfg.wandb.run_id, "wandb.run_id must be set (e.g. 'grateful-sunset-21')"
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_path = ckpt_dir / f"{cfg.wandb.run_id}.pth"
    cfg_path = ckpt_dir / f"{cfg.wandb.run_id}.yaml"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    train_cfg = OmegaConf.load(cfg_path)
    score_str = f", val_loss={ckpt['val_loss']:.6f}" if "val_loss" in ckpt else ""
    console.print(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']}{score_str})")
    console.print(f"Loaded config: {cfg_path}")

    # Build and load model
    # TODO: older checkpoints don't store in_channels/out_channels — make sure
    # all future checkpoints include them and remove these fallback defaults.
    model = instantiate(
        train_cfg.models,
        in_channels=ckpt.get("in_channels", 150),
        out_channels=ckpt.get("out_channels", 37),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    console.print(f"Model: [cyan]{train_cfg.models._target_}[/cyan]")
    console.print(f"  In channels:  [cyan]{ckpt.get('in_channels', 150)}[/cyan]")
    console.print(f"  Out channels: [cyan]{ckpt.get('out_channels', 37)}[/cyan]")

    # Load all.zarr (same structure as training script)
    root_dir = Path(train_cfg.data.root_dir)
    resolution_km = train_cfg.data.resolution_km
    patch_size = train_cfg.data.patch_size
    stride = train_cfg.data.stride
    zarr_dir = (
        root_dir / f"{resolution_km}km" / "chips" / f"patch{patch_size}_stride{stride}"
    )
    assert zarr_dir.exists(), f"Zarr directory does not exist: {zarr_dir}"
    all_zarr_path = zarr_dir / "all.zarr"
    assert all_zarr_path.exists(), f"all.zarr not found: {all_zarr_path}"

    all_store = zarr.open_group(str(all_zarr_path), mode="r")

    predictors = [
        name
        for name, predictor_cfg in train_cfg.data.predictors.items()
        if bool(predictor_cfg.use)
    ]

    target_cfg = train_cfg.data.targets
    target_dataset = str(target_cfg.dataset)
    zarr_band_names = all_store["targets"].attrs["band_names"]
    band_to_idx = {name: idx for idx, name in enumerate(zarr_band_names)}
    zarr_all_traits = [
        f.replace("X", "").replace(".tif", "")
        for f in all_store[f"targets/{target_dataset}"].attrs["files"]
    ]
    traits = (
        [str(v) for v in target_cfg.traits] if target_cfg.traits else zarr_all_traits
    )
    cfg_bands = [str(v) for v in target_cfg.bands]
    n_bands = len(zarr_band_names)
    target_indices = [
        trait_pos * n_bands + band_to_idx[band]
        for trait_pos in range(len(traits))
        for band in cfg_bands
    ]
    source_indices = [
        trait_pos * n_bands + band_to_idx["source"] for trait_pos in range(len(traits))
    ]

    dataset = PlantTraitDataset(
        all_zarr_path,
        predictors=predictors,
        target=target_dataset,
        target_indices=target_indices,
        source_indices=source_indices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.data_loaders.batch_size,
        shuffle=False,
        num_workers=train_cfg.data_loaders.num_workers,
    )
    console.print(f"Zarr: [cyan]{all_zarr_path}[/cyan]")
    console.print(f"Samples: [cyan]{len(dataset)}[/cyan]")

    # Inference
    all_preds = []

    with torch.no_grad():
        for X, _, _ in track(dataloader, description="Predicting"):
            X = X.to(device, dtype=torch.float32)
            valid = torch.isfinite(X).all(dim=1, keepdim=True)  # (B, 1, H, W)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.clamp(X, min=-1e4, max=1e4)
            y_pred = model(X).cpu()
            y_pred = torch.where(
                valid.cpu().expand_as(y_pred), y_pred, torch.tensor(float("nan"))
            )
            all_preds.append(y_pred)

    preds = torch.cat(all_preds)  # (N, out_channels, H_out, W_out)
    console.print(f"Predictions shape: [cyan]{tuple(preds.shape)}[/cyan]")

    # Reconstruct continuous GeoTIFF from chips
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = all_store.attrs["transform"]
    geo_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs_epsg = all_store.attrs["crs_epsg"]
    pixel_w = geo_transform.a
    pixel_h = abs(geo_transform.e)

    bounds = all_store["bounds"][:]  # (N, 4): min_x, min_y, max_x, max_y

    preds_np = preds.numpy()  # (N, out_channels, H_out, W_out)
    n_chips, out_ch, h_out, w_out = preds_np.shape

    # Anchor the canvas to the original raster's top-left corner (stored in the
    # zarr transform), NOT to global_min_x/global_max_y derived from bounds.
    # The two should be equal when all-zarr chips start at pixel (0,0), but if
    # the top/left border chips are absent (e.g. ocean, no valid H3 cell) the
    # bounds-derived origin is shifted inward by one or more stride steps.
    canvas_origin_x = geo_transform.c  # = transform.c of the source raster
    canvas_origin_y = geo_transform.f  # = transform.f of the source raster (top)
    canvas_rows = all_store.attrs["raster_height"]
    canvas_cols = all_store.attrs["raster_width"]

    # Weight map: pixel weight = distance to nearest chip edge (in each axis),
    # so center pixels contribute more than edge pixels when patches overlap.
    # With patch_size=15, stride=10 the overlap is 5 px; center pixels get
    # weight 8*8=64 while corner pixels get 1*1=1.
    _yw = np.minimum(np.arange(1, h_out + 1), np.arange(h_out, 0, -1)).astype(
        np.float32
    )
    _xw = np.minimum(np.arange(1, w_out + 1), np.arange(w_out, 0, -1)).astype(
        np.float32
    )
    weight_map = np.outer(_yw, _xw)  # (h_out, w_out)

    canvas = np.zeros((out_ch, canvas_rows, canvas_cols), dtype=np.float32)
    weight_sum = np.zeros((canvas_rows, canvas_cols), dtype=np.float32)

    for i in range(n_chips):
        min_x, _, _, max_y = bounds[i]
        col = round((min_x - canvas_origin_x) / pixel_w)
        row = round((canvas_origin_y - max_y) / pixel_h)

        # Clip to canvas bounds (edge chips may extend beyond the raster boundary)
        r0, r1 = max(row, 0), min(row + h_out, canvas_rows)
        c0, c1 = max(col, 0), min(col + w_out, canvas_cols)
        if r0 >= r1 or c0 >= c1:
            continue

        chip = preds_np[i, :, r0 - row : r1 - row, c0 - col : c1 - col]
        w = weight_map[r0 - row : r1 - row, c0 - col : c1 - col]
        w = np.where(np.isfinite(chip[0]), w, 0.0)  # zero-out invalid pixels
        canvas[:, r0:r1, c0:c1] += chip * w
        weight_sum[r0:r1, c0:c1] += w

    nonzero = weight_sum > 0
    canvas[:, nonzero] /= weight_sum[nonzero]
    canvas[:, ~nonzero] = np.nan

    out_tif_path = out_dir / f"{cfg.wandb.run_id}.tif"
    out_geo_transform = Affine(
        pixel_w, 0, canvas_origin_x, 0, -pixel_h, canvas_origin_y
    )
    band_names = [f"{trait}_{band}" for trait in traits for band in cfg_bands]
    with rasterio.open(
        out_tif_path,
        "w",
        driver="GTiff",
        height=canvas_rows,
        width=canvas_cols,
        count=out_ch,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=out_geo_transform,
        compress="deflate",
        tiled=True,
        nodata=np.nan,
    ) as dst:
        dst.write(canvas)
        for b_idx, name in enumerate(band_names, start=1):
            dst.set_band_description(b_idx, name)
    console.print(f"Saved tif:  [cyan]{out_tif_path}[/cyan]")
    console.print(
        f"  shape: [cyan]{canvas_rows} x {canvas_cols}[/cyan]  bands: [cyan]{out_ch}[/cyan]  CRS: [cyan]EPSG:{crs_epsg}[/cyan]"
    )
    console.rule("[bold cyan]DONE[/bold cyan]")


if __name__ == "__main__":
    main()
