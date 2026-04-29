"""
Generate a per-trait score table (CSV) from test metric JSON files.

Usage:
    # Use current (v1) runs:
    python scripts/make_trait_table.py

    # Use corrected (v2) runs — run after the v2 training is complete:
    python scripts/make_trait_table.py --v2

    # Custom output path:
    python scripts/make_trait_table.py --output analysis/my_table.csv

Output columns:
    trait_id, trait_name, trait_unit,
    stl_weighted_r,   stl_weighted_r2,
    stl_gradnorm_r,   stl_gradnorm_r2,
    mtl_weighted_r,   mtl_weighted_r2,
    mtl_gradnorm_r,   mtl_gradnorm_r2,
    mmoe_weighted_r,  mmoe_weighted_r2,
    mmoe_gradnorm_r,  mmoe_gradnorm_r2,
    best_model
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRICS_DIR = Path("scripts/Checkpoints_Scores/metrics")
TRAIT_MAPPING = Path("/scratch3/plant-traits-v2/data/trait_mapping.json")
OUTPUT_DEFAULT = Path("analysis/trait_scores.csv")

# Run name → [v1 stem, v2 stem]
# v1 = original (possibly broken hyperparams) runs
# v2 = corrected runs (suffixed with "2")
MODEL_STEMS = {
    "stl_weighted": ["stl_weighted_best_r", "stl_weighted2_best_r"],
    "stl_gradnorm": ["stl_gradnorm_best_r", "stl_gradnorm2_best_r"],
    "mtl_weighted": ["mtl_weighted_best_r", "mtl_weighted2_best_r"],
    "mtl_gradnorm": ["mtl_gradnorm_best_r", "mtl_gradnorm2_best_r"],
    "mmoe_weighted": ["mmoe_weighted_best_r", "mmoe_weighted2_best_r"],
    "mmoe_gradnorm": ["mmoe_gradnorm_best_r", "mmoe_gradnorm2_best_r"],
}

# Column display order
MODEL_ORDER = [
    "stl_weighted",
    "stl_gradnorm",
    "mtl_weighted",
    "mtl_gradnorm",
    "mmoe_weighted",
    "mmoe_gradnorm",
]


def load_trait_mapping(path: Path) -> dict:
    if not path.exists():
        print(f"[WARNING] Trait mapping not found: {path}", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def load_metric_file(stem: str, metrics_dir: Path) -> dict | None:
    path = metrics_dir / f"{stem}.test_metrics_full.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("trait_metrics", {})


def pick_stem(col: str, use_v2: bool) -> str:
    stems = MODEL_STEMS[col]
    if use_v2:
        return stems[1]  # v2 (corrected) run
    return stems[0]  # v1 (original) run


def main(args: argparse.Namespace) -> None:
    use_v2 = args.v2
    metrics_dir = Path(args.metrics_dir)
    out_path = Path(args.output)

    trait_map = load_trait_mapping(TRAIT_MAPPING)

    # Load per-trait metric dicts
    model_data: dict[str, dict] = {}
    for col in MODEL_ORDER:
        stem = pick_stem(col, use_v2)
        d = load_metric_file(stem, metrics_dir)
        if d is None:
            # Fall back to the other version if preferred one doesn't exist
            fallback = pick_stem(col, not use_v2)
            d = load_metric_file(fallback, metrics_dir)
            if d is not None:
                print(
                    f"[INFO] {col}: '{stem}' not found — using '{fallback}'",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[WARNING] {col}: no metric file found for either v1/v2 — column will be empty.",
                    file=sys.stderr,
                )
                d = {}
        model_data[col] = d

    # Union of all trait IDs (sorted numerically where possible)
    all_trait_ids: set[str] = set()
    for d in model_data.values():
        all_trait_ids |= set(d.keys())

    def sort_key(tid: str) -> int | str:
        return int(tid) if tid.isdigit() else tid

    sorted_traits = sorted(all_trait_ids, key=sort_key)

    # Print macro-r summary
    print("\n=== MACRO Pearson-r (SPLOT-only) ===")
    print(f"{'Model':<20} {'macro_r':>8}")
    for col in MODEL_ORDER:
        stem = pick_stem(col, use_v2)
        path = metrics_dir / f"{stem}.test_metrics_full.json"
        if not path.exists():
            # try fallback
            stem = pick_stem(col, not use_v2)
            path = metrics_dir / f"{stem}.test_metrics_full.json"
        if path.exists():
            with open(path) as f:
                top = json.load(f)
            print(f"  {col:<18} {top.get('macro_pearson_r', float('nan')):>8.4f}")
        else:
            print(f"  {col:<18} {'N/A':>8}")
    print()

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["trait_id", "trait_name", "trait_unit"]
    for col in MODEL_ORDER:
        header += [f"{col}_r", f"{col}_r2"]
    header += ["best_model_by_r"]

    rows_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for tid in sorted_traits:
            info = trait_map.get(str(tid), {})
            name = info.get("short", tid)
            unit = info.get("unit", "")

            row: list = [tid, name, unit]
            r_scores: dict[str, float] = {}

            for col in MODEL_ORDER:
                m = model_data[col].get(str(tid), {})
                r_val = m.get("pearson_r", None)
                r2_val = m.get("r2", None)
                row.append(f"{r_val:.4f}" if r_val is not None else "")
                row.append(f"{r2_val:.4f}" if r2_val is not None else "")
                if r_val is not None:
                    r_scores[col] = r_val

            best = max(r_scores, key=r_scores.__getitem__) if r_scores else ""
            row.append(best)
            writer.writerow(row)
            rows_written += 1

    print(f"Saved {rows_written} traits → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-trait score table from test metric JSONs."
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use v2 (corrected) run metric files (suffix '2').",
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(METRICS_DIR),
        help="Directory containing test_metrics_full.json files.",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_DEFAULT), help="Output CSV path."
    )
    args = parser.parse_args()
    main(args)
