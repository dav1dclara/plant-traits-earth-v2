from pathlib import Path

import hydra
import numpy as np
import rasterio
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from rasterio.windows import Window
from rich.console import Console
from rich.progress import track

console = Console()


@hydra.main(config_path="../config", config_name="training/david", version_base=None)
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

    # Collect predictor files (same order as training)
    predictors_root = Path(cfg.prediction.predictors_dir)
    predictor_files: list[Path] = []
    for name, pred_cfg in cfg.data.predictors.items():
        if pred_cfg.use:
            files = sorted((predictors_root / name).glob("*.tif"))
            if not files:
                raise FileNotFoundError(
                    f"No .tif files for predictor '{name}' in {predictors_root / name}"
                )
            predictor_files.extend(files)

    total_bands = 0
    for f in predictor_files:
        with rasterio.open(f) as src:
            total_bands += src.count
    console.print(
        f"Predictors: [cyan]{len(predictor_files)} files, {total_bands} bands[/cyan]"
    )

    # Load checkpoint
    checkpoint_path = Path(cfg.prediction.checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Infer out_channels from the final layer bias in the checkpoint
    out_channels = None
    for key, tensor in state.items():
        if key.endswith("head.bias") and tensor.ndim == 1:
            out_channels = int(tensor.shape[0])
            break
    if out_channels is None:
        raise ValueError(
            "Could not infer out_channels from checkpoint. Set cfg.prediction.out_channels explicitly."
        )
    console.print(f"Output channels: [cyan]{out_channels}[/cyan]")

    # Build and load model
    model = instantiate(
        cfg.models, in_channels=total_bands, out_channels=out_channels
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    console.print(f"Model: [cyan]{cfg.models._target_}[/cyan]")

    # Tiling parameters
    tile_size = int(cfg.prediction.tile_size)
    halo = int(cfg.prediction.halo)
    console.print(f"Tile size: [cyan]{tile_size}[/cyan]  Halo: [cyan]{halo}[/cyan]")

    # Reference raster defines the output grid
    reference_tif = Path(cfg.prediction.reference_tif)
    assert reference_tif.exists(), f"Reference tif not found: {reference_tif}"

    output_tif = Path(cfg.prediction.output_tif)
    assert output_tif.parent.exists(), (
        f"Output directory does not exist: {output_tif.parent}"
    )

    with rasterio.open(reference_tif) as ref:
        height, width = ref.height, ref.width
        console.print(f"Grid: [cyan]{height} x {width}[/cyan]")

        predictions = np.full((out_channels, height, width), np.nan, dtype=np.float32)

        n_rows = (height + tile_size - 1) // tile_size
        n_cols = (width + tile_size - 1) // tile_size

        srcs = [rasterio.open(f) for f in predictor_files]
        try:
            with torch.no_grad():
                for tile_idx in track(range(n_rows * n_cols), description="Predicting"):
                    row = tile_idx // n_cols
                    col = tile_idx % n_cols

                    y0 = row * tile_size
                    x0 = col * tile_size
                    core_h = min(tile_size, height - y0)
                    core_w = min(tile_size, width - x0)

                    window = Window(
                        x0 - halo, y0 - halo, core_w + 2 * halo, core_h + 2 * halo
                    )
                    x_parts = [
                        src.read(window=window, boundless=True, fill_value=0).astype(
                            np.float32
                        )
                        for src in srcs
                    ]
                    x_np = np.concatenate(x_parts, axis=0)
                    x = (
                        torch.from_numpy(x_np)
                        .unsqueeze(0)
                        .to(device=device, dtype=torch.float32)
                    )
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                    x = torch.clamp(x, min=-1e4, max=1e4)

                    y_pred = model(x).squeeze(0).detach().cpu().numpy()
                    predictions[:, y0 : y0 + core_h, x0 : x0 + core_w] = y_pred[
                        :, halo : halo + core_h, halo : halo + core_w
                    ]
        finally:
            for src in srcs:
                src.close()

        profile = ref.profile.copy()
        profile.update(
            dtype="float32",
            count=out_channels,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
        )
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(predictions)

    console.print(f"Output: [cyan]{output_tif}[/cyan]")
    console.rule("[bold cyan]DONE[/bold cyan]")


if __name__ == "__main__":
    main()
