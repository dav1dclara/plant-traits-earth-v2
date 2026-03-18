"""
This file is OUTDATED.
"""

from pathlib import Path

from omegaconf import DictConfig
from torchgeo.datasets import IntersectionDataset, RasterDataset

from functools import reduce

TRAIT_IDS = (
    4,
    6,
    13,
    14,
    15,
    21,
    26,
    27,
    46,
    47,
    50,
    55,
    78,
    95,
    138,
    144,
    145,
    146,
    163,
    169,
    223,
    224,
    237,
    281,
    282,
    289,
    297,
    351,
    614,
    1080,
    3106,
    3107,
    3112,
    3113,
    3114,
    3117,
    3120,
)


class CanopyHeight(RasterDataset):
    """Canopy height & canopy height SD (2 bands total)"""

    name = "canopy_height"
    filename_glob = "ETH_GlobalCanopy*.tif"
    filename_regex = r"ETH_GlobalCanopy(?P<band>.+)_2020_v1\.tif"
    separate_files = True
    all_bands = ("Height", "HeightSD")
    is_image = True


class Modis(RasterDataset):
    """MODIS surface reflectance: 5 bands + NDVI, 12 monthly means (72 bands total)"""

    name = "modis"
    filename_glob = "sur_refl_*.tif"
    filename_regex = r"sur_refl_(?P<band>.+)\.tif"
    separate_files = True
    all_bands = tuple(
        f"{band}_2001-2024_m{month}_mean"
        for band in ["b01", "b02", "b03", "b04", "b05", "ndvi"]
        for month in range(1, 13)
    )
    is_image = True


class SoilGrids(RasterDataset):
    """SoilGrids: 10 properties at 6 depths + ocs at 1 depth (61 bands total)"""

    name = "soilgrids"
    filename_glob = "*.tif"
    filename_regex = r"(?P<band>.+)_mean\.tif"
    separate_files = True
    all_bands = tuple(
        f"{prop}_{depth}"
        for prop in [
            "bdod",
            "cec",
            "cfvo",
            "clay",
            "nitrogen",
            "ocd",
            "phh2o",
            "sand",
            "silt",
            "soc",
        ]
        for depth in ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
    ) + ("ocs_0-30cm",)
    is_image = True


class Vodca(RasterDataset):
    """VODCA — C/K/X microwave bands, mean/p5/p95 (9 bands total)."""

    name = "vodca"
    filename_glob = "vodca_*.tif"
    filename_regex = r"vodca_(?P<band>.+)\.tif"
    separate_files = True
    all_bands = tuple(
        f"{freq}-band_{stat}"
        for freq in ["c", "k", "x"]
        for stat in ["mean", "p5", "p95"]
    )
    is_image = True


class WorldClim(RasterDataset):
    """WorldClim bioclimatic variables (6 bands)."""

    name = "worldclim"
    filename_glob = "wc2.1_30s_bio_*.tif"
    filename_regex = r"wc2\.1_30s_bio_(?P<band>.+)\.tif"
    separate_files = True
    all_bands = ("1", "4", "7", "12", "13-14", "15")
    is_image = True


class GbifTraits(RasterDataset):
    """GBIF occurrence-based trait maps — 37 traits (one file per trait)."""

    name = "gbif"
    filename_glob = "X*.tif"
    filename_regex = r"X(?P<band>\d+)\.tif"
    separate_files = True
    all_bands = tuple(str(tid) for tid in TRAIT_IDS)
    is_image = True


class SplotTraits(RasterDataset):
    """sPlot community-weighted mean trait maps — 37 traits (one file per trait)."""

    name = "splot"
    filename_glob = "X*.tif"
    filename_regex = r"X(?P<band>\d+)\.tif"
    separate_files = True
    all_bands = tuple(str(tid) for tid in TRAIT_IDS)
    is_image = True


def _validate_datasets(datasets: dict) -> None:  # TODO: might need more checks here
    """Validate CRS and resolution consistency across datasets."""
    ref_name, ref = next(iter(datasets.items()))
    for name, ds in datasets.items():
        n_files = len(ds.files) if hasattr(ds, "files") else "N/A"
        print(f"Dataset: {name}")
        print(f"  - Files: {n_files}")
        print(f"  - CRS: EPSG:{ds.crs.to_epsg()}")
        print(f"  - Resolution: {ds.res}")
        if name == ref_name:
            continue
        if ds.crs != ref.crs:
            raise ValueError(
                f"CRS mismatch: {ref_name}=EPSG:{ref.crs.to_epsg()}, {name}=EPSG:{ds.crs.to_epsg()}"
            )
        if ds.res != ref.res:
            raise ValueError(
                f"Resolution mismatch: {ref_name}={ref.res}, {name}={ds.res}"
            )


def _collect_bands(dataset: RasterDataset | IntersectionDataset) -> list[str]:
    """Recursively collect all band names from a (possibly nested) dataset."""
    if isinstance(dataset, IntersectionDataset):
        return _collect_bands(dataset.datasets[0]) + _collect_bands(dataset.datasets[1])
    return list(dataset.bands)


def create_predictors_dataset(cfg: DictConfig) -> RasterDataset | IntersectionDataset:
    """Instantiate EO datasets listed in cfg.training.data.predictors."""
    print("Creating predictors dataset...")
    data_root = Path(cfg.data.root_dir)
    registry = {
        cls.name: cls for cls in RasterDataset.__subclasses__() if hasattr(cls, "name")
    }

    datasets = {}
    for name, predictor in cfg.training.data.predictors.items():
        if not predictor.use:
            continue
        kwargs = {"paths": data_root / cfg.data[name].path}
        if predictor.bands:
            kwargs["bands"] = list(predictor.bands)
        datasets[name] = registry[name](**kwargs)

    _validate_datasets(datasets)

    # Combine datasets
    dataset_list = list(datasets.values())
    combined = reduce(lambda a, b: a | b, dataset_list)

    return combined


def create_targets_dataset(cfg: DictConfig) -> RasterDataset:
    """Instantiate the target trait dataset from cfg.training.data.targets."""
    print("Creating targets dataset...")
    data_root = Path(cfg.data.root_dir)
    source = cfg.training.data.targets.source  # gbif or splot
    registry = {"gbif": GbifTraits, "splot": SplotTraits}

    kwargs = {"paths": data_root / cfg.data[source].path}
    trait_ids = list(cfg.training.data.targets.trait_ids)
    if trait_ids:
        kwargs["bands"] = [str(tid) for tid in trait_ids]
    dataset = registry[source](**kwargs)

    _validate_datasets({source: dataset})

    return dataset


def create_dataset(cfg: DictConfig) -> IntersectionDataset:
    """Create the combined predictor & target dataset."""
    predictors = create_predictors_dataset(cfg)
    targets = create_targets_dataset(cfg)

    _validate_datasets({"predictors": predictors, "targets": targets})

    combined = predictors & targets

    # all_bands = _collect_bands(combined)
    # print(f"\nCombined dataset: {len(all_bands)} bands total")
    # print(
    #     f"  - Predictor bands ({len(_collect_bands(predictors))}): {_collect_bands(predictors)}"
    # )
    # print(
    #     f"  - Target bands   ({len(_collect_bands(targets))}): {_collect_bands(targets)}"
    # )

    return combined
