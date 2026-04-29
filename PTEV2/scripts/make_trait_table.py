"""
Generate a per-trait score comparison table (CSV) from test metric JSON files.

Auto-discovers all *.test_metrics.json files in the metrics directory and
produces a CSV with one row per trait and one column-pair (r, rmse) per run.

Usage:
    # Auto-discover all runs in default metrics dir:
    python scripts/make_trait_table.py

    # Custom metrics dir and output path:
    python scripts/make_trait_table.py \
        --metrics_dir results/metrics \
        --output analysis/comparison_table.csv

    # Restrict to specific runs (stems without suffix):
    python scripts/make_trait_table.py \
        --runs chips_stl_best_r chips_mtl_best_r chips_mmoe_best_r

Output columns:
    trait_id, {run}_r, {run}_rmse, {run}_n, ..., best_run_r
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

METRICS_DIR = Path("results/metrics")
OUTPUT_DEFAULT = Path("analysis/comparison_table.csv")


def load_metric_file(path: Path) -> dict | None:
    """Load a .test_metrics.json and return it, or None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARNING] Could not load {path}: {exc}", file=sys.stderr)
        return None


def discover_runs(
    metrics_dir: Path, filter_stems: list[str] | None
) -> list[tuple[str, Path]]:
    """Return sorted (stem, path) pairs for all *.test_metrics.json files found."""
    all_files = sorted(metrics_dir.glob("*.test_metrics.json"))
    if not all_files:
        print(
            f"[ERROR] No *.test_metrics.json files found in {metrics_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if filter_stems:
        filter_set = set(filter_stems)
        all_files = [
            p
            for p in all_files
            if p.name.replace(".test_metrics.json", "") in filter_set
        ]
        if not all_files:
            print(
                f"[ERROR] None of the requested runs were found in {metrics_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
    return [(p.name.replace(".test_metrics.json", ""), p) for p in all_files]


def main(args: argparse.Namespace) -> None:
    metrics_dir = Path(args.metrics_dir)
    out_path = Path(args.output)
    filter_stems = args.runs if args.runs else None

    # Auto-discover JSON files
    runs = discover_runs(metrics_dir, filter_stems)
    print(f"Found {len(runs)} metric file(s) in {metrics_dir}:")
    for stem, _ in runs:
        print(f"  {stem}")

    # Load all JSONs
    run_data: dict[str, dict] = {}
    for stem, path in runs:
        d = load_metric_file(path)
        if d is not None:
            run_data[stem] = d

    if not run_data:
        print("[ERROR] No metric files could be loaded.", file=sys.stderr)
        sys.exit(1)

    # Print macro-r summary
    print("\n=== Summary (macro Pearson-r on test split) ===")
    print(f"  {'Run':<45} {'macro_r':>8}  {'macro_rmse':>10}  {'epoch':>5}")
    for stem, _ in runs:
        d = run_data.get(stem, {})
        macro_r = d.get("macro_r", float("nan"))
        macro_rmse = d.get("macro_rmse", float("nan"))
        epoch = d.get("epoch", "?")
        print(f"  {stem:<45} {macro_r:>8.4f}  {macro_rmse:>10.4f}  {epoch:>5}")
    print()

    # Collect union of all trait IDs
    all_trait_ids: set[str] = set()
    for d in run_data.values():
        per_trait = d.get("per_trait", {})
        all_trait_ids |= set(per_trait.keys())

    def sort_key(tid: str) -> tuple:
        # Sort by numeric suffix if trait has form X{number}
        if tid.startswith("X") and tid[1:].isdigit():
            return (0, int(tid[1:]))
        return (1, tid)

    sorted_traits = sorted(all_trait_ids, key=sort_key)

    # Build CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem_list = [s for s, _ in runs if s in run_data]

    header = ["trait_id"]
    for stem in stem_list:
        header += [f"{stem}_r", f"{stem}_rmse", f"{stem}_n"]
    header.append("best_run_by_r")

    rows_written = 0
    with open(out_path, "w", newline="") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(header)

        for tid in sorted_traits:
            row: list = [tid]
            r_scores: dict[str, float] = {}

            for stem in stem_list:
                per_trait = run_data[stem].get("per_trait", {})
                m = per_trait.get(tid, {})
                r_val = m.get("r", None)
                rmse_val = m.get("rmse", None)
                n_val = m.get("n", None)
                row.append(f"{r_val:.4f}" if r_val is not None else "")
                row.append(f"{rmse_val:.4f}" if rmse_val is not None else "")
                row.append(str(n_val) if n_val is not None else "")
                if r_val is not None:
                    r_scores[stem] = r_val

            best = max(r_scores, key=r_scores.__getitem__) if r_scores else ""
            row.append(best)
            writer.writerow(row)
            rows_written += 1

    print(f"Saved {rows_written} traits → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-trait score comparison table from test metric JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(METRICS_DIR),
        help=f"Directory containing *.test_metrics.json files (default: {METRICS_DIR}).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help=f"Output CSV path (default: {OUTPUT_DEFAULT}).",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="STEM",
        default=None,
        help=(
            "Optional: restrict to specific run stems, e.g. "
            "--runs chips_stl_best_r chips_mtl_best_r chips_mmoe_best_r"
        ),
    )
    args = parser.parse_args()
    main(args)
