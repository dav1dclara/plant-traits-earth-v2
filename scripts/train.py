import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import wandb
import importlib.util

from ptev2.data.dataset import CanopyHeight, Modis, SoilGrids, Vodca, WorldClim
from ptev2.utils import seed_all


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _validate_split_config(cfg: DictConfig) -> None:
    split_cfg = cfg.data.splits
    train_folds = list(split_cfg.train_folds)
    val_fold = int(split_cfg.val_fold)
    test_fold = int(split_cfg.test_fold)
    n_folds = int(split_cfg.n_folds)

    used_folds = train_folds + [val_fold, test_fold]
    if len(set(used_folds)) != len(used_folds):
        raise ValueError(f"Fold assignment overlaps: {used_folds}")
    if sorted(used_folds) != list(range(n_folds)):
        raise ValueError(
            f"Expected complete fold coverage 0..{n_folds - 1}, got {sorted(used_folds)}"
        )
    if len(train_folds) != n_folds - 2:
        raise ValueError(
            f"Expected n_folds-2 training folds, got {len(train_folds)} for n_folds={n_folds}"
        )


def _validate_data_sources(cfg: DictConfig) -> None:
    paths = cfg.data.sources
    required_dirs = {
        "eo_processed_dir": Path(paths.eo_processed_dir),
        "merged_targets_dir": Path(paths.merged_targets_dir),
        "source_mask_dir": Path(paths.source_mask_dir),
        "gbif_targets_dir": Path(paths.gbif_targets_dir),
        "splot_targets_dir": Path(paths.splot_targets_dir),
        "skcv_splits_dir": Path(paths.skcv_splits_dir),
    }
    for label, directory in required_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {directory}")

    if paths.esa_worldcover_water_mask:
        water_mask = Path(paths.esa_worldcover_water_mask)
        if not water_mask.is_file():
            raise FileNotFoundError(
                f"esa_worldcover_water_mask does not exist: {water_mask}"
            )


def _count_files(directory: Path, pattern: str) -> int:
    return len(sorted(directory.glob(pattern)))


def train(cfg: DictConfig) -> None:
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    seed_all(cfg.training.seed)

    device = _resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    model = instantiate(cfg.models.active).to(device)
    total_params, trainable_params = _count_parameters(model)
    print(f"Instantiated model: {model.__class__.__name__}")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    _validate_split_config(cfg)
    _validate_data_sources(cfg)

    paths = cfg.data.sources
    eo_data_dir = Path(paths.eo_processed_dir)

    datasets = {
        "CanopyHeight": CanopyHeight(paths=eo_data_dir / "canopy_height"),
        "Modis": Modis(paths=eo_data_dir / "modis"),
        "SoilGrids": SoilGrids(paths=eo_data_dir / "soil_grids"),
        "Vodca": Vodca(paths=eo_data_dir / "vodca"),
        "WorldClim": WorldClim(paths=eo_data_dir / "worldclim"),
    }

    predictor_bands = 0
    for name, ds in datasets.items():
        print(f"{name}: {len(ds.all_bands)} bands, {len(ds)} files, res={ds.res}")
        predictor_bands += len(ds.all_bands)

    if predictor_bands != int(cfg.data.in_channels):
        raise ValueError(
            f"Configured data.in_channels={cfg.data.in_channels} but found {predictor_bands} EO bands."
        )

    trait_pattern = cfg.data.trait_pattern
    merged_n = _count_files(Path(paths.merged_targets_dir), trait_pattern)
    gbif_n = _count_files(Path(paths.gbif_targets_dir), trait_pattern)
    splot_n = _count_files(Path(paths.splot_targets_dir), trait_pattern)
    source_mask_n = _count_files(Path(paths.source_mask_dir), "*_source.tif")
    split_n = _count_files(Path(paths.skcv_splits_dir), cfg.data.split_parquet_pattern)

    print(
        f"Targets: merged={merged_n}, gbif={gbif_n}, splot={splot_n}, "
        f"source_masks={source_mask_n}, split_files={split_n}"
    )
    print(
        "Source mask encoding: "
        f"no_data={cfg.data.source_mask_encoding.no_data}, "
        f"cit={cfg.data.source_mask_encoding.cit}, "
        f"sci={cfg.data.source_mask_encoding.sci}, "
        f"raster_nodata={cfg.data.source_mask_encoding.raster_nodata}"
    )

    expected_traits = int(cfg.data.n_traits)
    if not all(
        x == expected_traits for x in [merged_n, gbif_n, splot_n, source_mask_n]
    ):
        raise ValueError(
            f"Trait/mask count mismatch: expected {expected_traits}, got "
            f"merged={merged_n}, gbif={gbif_n}, splot={splot_n}, source_mask={source_mask_n}"
        )
    if split_n != expected_traits:
        raise ValueError(
            f"Expected {expected_traits} split parquet files, found {split_n}."
        )

    if not importlib.util.find_spec("pyarrow") and not importlib.util.find_spec(
        "fastparquet"
    ):
        print(
            "Warning: pyarrow/fastparquet not installed. Parquet column validation is skipped."
        )

    # Dataloader / training loop handled separately.
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    # val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    loss_fn = instantiate(cfg.training.loss)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer)
    _ = (loss_fn, optimizer, scheduler)

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        # Training and validation logic goes here
        # For example:
        # model.train()
        # for batch in train_dl:
        #     optimizer.zero_grad()
        #     inputs, targets = batch     #     outputs = model(inputs.to(device))
        #     loss = loss_fn(outputs, targets.to(device))
        #     loss.backward()
        #     optimizer.step()
        # scheduler.step()
        # model.eval()
        # with torch.no_grad():
        #     for batch in val_dl:
        #         inputs, targets = batch
        #         outputs = model(inputs.to(device))
        #         val_loss = loss_fn(outputs, targets.to(device))
        #         # Log val_loss to wandb or print it as needed


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    available_models = set(cfg.models.registry.keys())
    if cfg.models.name not in available_models:
        raise ValueError(
            f"Unsupported model_name '{cfg.models.name}'. "
            f"Available: {sorted(available_models)}"
        )

    selected_model_cfg = cfg.models.registry[cfg.models.name]
    if selected_model_cfg != cfg.models.active:
        raise ValueError(
            "Config mismatch: cfg.models.active does not match cfg.models.registry[cfg.models.name]."
        )

    print(f"Selected model_name: {cfg.models.name}")
    print(f"Selected target: {selected_model_cfg._target_}")
    train(cfg)


if __name__ == "__main__":
    main()
