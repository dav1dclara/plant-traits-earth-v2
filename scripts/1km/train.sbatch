#!/bin/bash
#SBATCH --job-name=plant-traits-1km-simple
#SBATCH --output=/cluster/home/ldominiak/plant-traits-earth-v2/logs/%x_%j.log
#SBATCH --partition=gpuhe.24h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --time=12:00:00
#SBATCH --account=es_schin

set -euo pipefail

# Load modules and activate venv
module restore plant-traits
source ~/venvs/plant-traits/bin/activate

MODEL=${MODEL:-}
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        MODEL=*)
            MODEL="${1#MODEL=}"
            shift
            ;;
        --job-name|-J)
            echo "Ignoring '$1 ${2:-}': pass Slurm job names before the script path, e.g. sbatch --job-name=... scripts/1km/train.sbatch ..." >&2
            shift
            if [[ $# -gt 0 ]]; then
                shift
            fi
            ;;
        --job-name=*|-J*)
            echo "Ignoring '$1': pass Slurm job names before the script path, e.g. sbatch --job-name=... scripts/1km/train.sbatch ..." >&2
            shift
            ;;
        training=*)
            echo "Invalid override '$1' for scripts/1km/train.py." >&2
            echo "Use MODEL=tuned_mmoe sbatch scripts/1km/train.sbatch ... or pass models=tuned_mmoe." >&2
            exit 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

MODEL_OVERRIDE=()
if [[ -n "$MODEL" ]]; then
    case "$MODEL" in
        stl|mlp|mtl|tuned_mmoe)
            MODEL_OVERRIDE=("models=$MODEL")
            ;;
        *)
            echo "Invalid MODEL='$MODEL'. Use one of: stl, mlp, mtl, tuned_mmoe." >&2
            exit 2
            ;;
    esac
fi

CHIPS_SRC=/cluster/work/igp_psr/plant-traits-earth-v2/data/1km/chips/patch128_stride64
CHIPS_DIR=$TMPDIR/chips
mkdir -p "$CHIPS_DIR"

echo "Copying zarr zips to local scratch ($TMPDIR)..."
cp "$CHIPS_SRC/train.zarr.zip" "$CHIPS_SRC/val.zarr.zip" "$CHIPS_DIR/"
echo "Done. $(du -sh "$CHIPS_DIR")"

# Run training
cd ~/plant-traits-earth-v2
python scripts/1km/train.py \
    +data.zarr_dir="$CHIPS_DIR" \
    "${MODEL_OVERRIDE[@]}" \
    train.use_tqdm=true \
    "${TRAIN_ARGS[@]}"
