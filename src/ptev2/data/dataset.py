from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.samplers import RandomGeoSampler


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


# TODO: re-implement this function - think about which sampler to use
def make_dataloader(
    dataset: RasterDataset | IntersectionDataset,
    patch_size: int,
    batch_size: int,
    num_samples: int,
    num_workers: int = 0,
) -> DataLoader:
    sampler = RandomGeoSampler(dataset, size=patch_size, length=num_samples)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
