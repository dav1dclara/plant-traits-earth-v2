"""
Merge GBIF and SPLOT trait rasters into a single set of rasters.

Merging rule (per pixel):
  - SPLOT only or both GBIF+SPLOT  →  use SPLOT
  - GBIF only                      →  use GBIF
  - Neither                        →  NaN

Output bands: mean, std, median, q05, q95, count, source
  source: NaN = no data, 1 = GBIF only, 2 = SPLOT (with or without GBIF)

Outputs saved to:
  /scratch3/plant-traits-v2/data/22km/merged/X*.tif
"""

from pathlib import Path

from rich.progress import track

from ptev2.data.preprocessing import combine_traits


def main() -> None:
    gbif_dir = Path("/scratch3/plant-traits-v2/data/22km/gbif")
    splot_dir = Path("/scratch3/plant-traits-v2/data/22km/splot")
    bands = ["mean", "std", "median", "q05", "q95", "count"]
    out_dir = Path("/scratch3/plant-traits-v2/data/22km/targets/comb")

    out_dir.mkdir(parents=True, exist_ok=True)

    traits = sorted(p.stem for p in gbif_dir.glob("*.tif"))
    splot_traits = {p.stem for p in splot_dir.glob("*.tif")}
    missing = [t for t in traits if t not in splot_traits]
    if missing:
        print(f"Warning: {len(missing)} trait(s) missing from SPLOT: {missing}")

    for trait in track(traits, description="Merging rasters..."):
        combine_traits(trait, gbif_dir, splot_dir, out_dir, bands)

    print(f"Done. Merged {len(traits)} traits → {out_dir}")


if __name__ == "__main__":
    main()
