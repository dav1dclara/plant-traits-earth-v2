from pathlib import Path

# import h5py
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchgeo.samplers import GridGeoSampler

from ptev2.data.dataset import (
    _collect_bands,
    create_predictors_dataset,
    create_targets_dataset,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_size = cfg.training.preprocess.patch_size
    stride = cfg.training.preprocess.stride
    batch_size = cfg.training.preprocess.batch_size
    num_workers = cfg.training.preprocess.num_workers
    output_path = Path(cfg.training.preprocess.output_path)

    # Build datasets separately to track the predictor/target band split
    predictors = create_predictors_dataset(cfg)
    targets = create_targets_dataset(cfg)
    combined = predictors & targets

    predictor_bands = _collect_bands(predictors)
    target_bands = _collect_bands(targets)
    n_predictor_bands = len(predictor_bands)

    print(
        f"\nPreprocessing {n_predictor_bands} predictor bands + {len(target_bands)} target bands"
    )
    print(f"Patch size: {patch_size}x{patch_size}, stride: {stride}")

    sampler = GridGeoSampler(combined, size=patch_size, stride=stride)
    dataloader = DataLoader(
        combined,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # keep as list of dicts
    )

    n_samples = len(sampler)
    print(f"Total patches: {n_samples}")
    print(f"Output: {output_path}")

    # output_path.parent.mkdir(parents=True, exist_ok=True)

    # with h5py.File(output_path, "w") as f:
    #     pred_ds = f.create_dataset(
    #         "predictors",
    #         shape=(n_samples, n_predictor_bands, patch_size, patch_size),
    #         dtype=np.float32,
    #         chunks=(1, n_predictor_bands, patch_size, patch_size),
    #     )
    #     tgt_ds = f.create_dataset(
    #         "targets",
    #         shape=(n_samples, len(target_bands), patch_size, patch_size),
    #         dtype=np.float32,
    #         chunks=(1, len(target_bands), patch_size, patch_size),
    #     )

    #     # Store band names as metadata
    #     f["predictors"].attrs["bands"] = predictor_bands
    #     f["targets"].attrs["bands"] = target_bands

    #     idx = 0
    #     for batch in dataloader:
    #         for sample in batch:
    #             image = sample["image"]  # (C, H, W)
    #             pred_ds[idx] = image[:n_predictor_bands].numpy()
    #             tgt_ds[idx] = image[n_predictor_bands:].numpy()
    #             idx += 1

    #         print(f"\r  {idx}/{n_samples} patches saved", end="", flush=True)

    # print(f"\nDone. Chips saved to {output_path}")


if __name__ == "__main__":
    main()
