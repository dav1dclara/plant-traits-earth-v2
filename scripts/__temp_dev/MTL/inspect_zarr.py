"""
Inspect Zarr files to understand the data structure.
This script prints structure information and optionally exports summaries or sample arrays.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def inspect_store(zarr_path: Path, export_dir: Path | None = None) -> dict:
    print(f"Inspecting {zarr_path}")
    store = zarr.open_group(str(zarr_path), mode="r")
    summary = {"path": str(zarr_path), "groups": []}

    for key in store.keys():
        item = store[key]
        if isinstance(item, zarr.Array):
            summary["groups"].append(
                {
                    "name": key,
                    "type": "array",
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                    "chunks": item.chunks,
                }
            )
        else:
            summary["groups"].append({"name": key, "type": "group"})

    if "predictors" in store:
        summary["predictors"] = []
        for name in store["predictors"].keys():
            arr = store[f"predictors/{name}"]
            summary["predictors"].append(
                {"name": name, "shape": arr.shape, "dtype": str(arr.dtype)}
            )

    if "targets" in store:
        summary["targets"] = []
        for name in store["targets"].keys():
            arr = store[f"targets/{name}"]
            summary["targets"].append(
                {"name": name, "shape": arr.shape, "dtype": str(arr.dtype)}
            )

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        summary_path = export_dir / f"{zarr_path.name}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved summary to {summary_path}")

        if "predictors" in store and "targets" in store:
            predictors = list(store["predictors"].keys())
            targets = list(store["targets"].keys())
            if predictors and targets:
                X = np.array(store[f"predictors/{predictors[0]}"][0])
                y = np.array(store[f"targets/{targets[0]}"][0])
                np.save(export_dir / f"{zarr_path.name}_X_sample.npy", X)
                np.save(export_dir / f"{zarr_path.name}_y_sample.npy", y)
                print(f"Saved sample arrays for {zarr_path.name}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Zarr stores and optionally export summaries."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to a Zarr store or folder containing split zarr stores",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory to save JSON summaries and sample arrays",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    zarr_path = args.zarr_path
    export_dir = args.export_dir

    if zarr_path.is_dir():
        for split in ["train", "val", "test"]:
            candidate = zarr_path / f"{split}.zarr"
            if candidate.exists():
                inspect_store(candidate, export_dir=export_dir)
            else:
                print(f"Skipping missing split {candidate}")
    else:
        inspect_store(zarr_path, export_dir=export_dir)


if __name__ == "__main__":
    main()
