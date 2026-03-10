from pathlib import Path
from torchgeo.datasets import RasterDataset


class CanopyHeight(RasterDataset):
    """Canopy height & canopy height SD (2 bands total)"""

    filename_glob = "ETH_GlobalCanopy*.tif"
    filename_regex = r"ETH_GlobalCanopy(?P<band>.+)_2020_v1\.tif"
    separate_files = True
    all_bands = ("Height", "HeightSD")
    is_image = True


class Modis(RasterDataset):
    """MODIS surface reflectance: 5 bands + NDVI, 12 monthly means (72 bands total)"""

    filename_glob = "sur_refl_*.tif"
    filename_regex = r"sur_refl_(?P<band>.+)\.tif"
    separate_files = True
    all_bands = tuple(
        f"{band}_2001-2024_m{month}_mean"
        for band in ["b01", "b02", "b03", "b04", "b05", "ndvi"]
        for month in range(1, 13)
    )
    is_image = True


# TODO: save these somewhere else
SOILGRIDS_PROPERTIES = [
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
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


class SoilGrids(RasterDataset):  # TODO: check this again
    """SoilGrids: 10 properties at 6 depths + ocs at 1 depth (61 bands total)"""

    filename_glob = "*.tif"
    filename_regex = r"(?P<band>.+)_mean\.tif"
    separate_files = True
    all_bands = tuple(
        f"{prop}_{depth}" for prop in SOILGRIDS_PROPERTIES for depth in SOILGRIDS_DEPTHS
    ) + ("ocs_0-30cm",)
    is_image = True


class Vodca(RasterDataset):
    """VODCA — C/K/X microwave bands, mean/p5/p95 (9 bands total)."""

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

    filename_glob = "wc2.1_30s_bio_*.tif"
    filename_regex = r"wc2\.1_30s_bio_(?P<band>.+)\.tif"
    separate_files = True
    all_bands = ("1", "4", "7", "12", "13-14", "15")
    is_image = True


# TODO: implement classes for targets
# class SPlotTraits(RasterDataset):
#     """sPlot community-weighted mean traits — one trait at a time."""

#     is_image = True

#     def __init__(self, paths: str | Path, trait_id: int, **kwargs):
#         self.filename_glob = f"X{trait_id}.tif"
#         self.filename_regex = rf"X{trait_id}\.tif"
#         super().__init__(paths=paths, **kwargs)


# class GBIFTraits(RasterDataset):
#     """GBIF occurrence-based trait maps — one trait at a time."""

#     is_image = True

#     def __init__(self, paths: str | Path, trait_id: int, **kwargs):
#         self.filename_glob = f"X{trait_id}.tif"
#         self.filename_regex = rf"X{trait_id}\.tif"
#         super().__init__(paths=paths, **kwargs)


# TODO: write a function that combines all datasets needed for training (as specified in the training config) -> the implementation below is not correct
def build_eo_dataset(eo_data_dir: Path):
    # datasets = {
    #     "CanopyHeight": CanopyHeight(paths=eo_data_dir / "canopy_height"),
    #     "Modis": Modis(paths=eo_data_dir / "modis"),
    #     "SoilGrids": SoilGrids(paths=eo_data_dir / "soilgrids"),
    #     "Vodca": Vodca(paths=eo_data_dir / "vodca"),
    #     "WorldClim": WorldClim(paths=eo_data_dir / "worldclim"),
    # }

    # for name, ds in datasets.items():
    #     print(f"{name}: {len(ds.all_bands)} bands, {len(ds)} files, res={ds.res}, bounds={ds.bounds}")

    # combined = None
    # for ds in datasets.values():
    #     combined = ds if combined is None else combined & ds
    # return combined
    pass


# TODO: implement a dataloader
def make_dataloader(dataset, batch_size):
    # return DataLoader()
    pass
