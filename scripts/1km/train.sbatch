#!/bin/bash
#SBATCH --job-name=plant-traits-1km
#SBATCH --output=/cluster/home/dclara/plant-traits-earth-v2/logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Load modules and activate venv
module restore plant-traits
source /cluster/home/dclara/venvs/plant-traits/bin/activate

# Unzip chips to scratch if not already done
CHIPS_DIR=/cluster/work/igp_psr/plant-traits-earth-v2/data/1km/chips/patch128_stride64
SCRATCH=/cluster/scratch/dclara/plant-traits-1km/chips/patch128_stride64

mkdir -p $SCRATCH
for SPLIT in train val test; do
    if [ ! -d $SCRATCH/${SPLIT}.zarr ]; then
        echo "Unzipping ${SPLIT}.zip to scratch..."
        unzip -q $CHIPS_DIR/${SPLIT}.zip -d $SCRATCH/
    else
        echo "${SPLIT}.zarr already in scratch, skipping."
    fi
done

# Run training
cd /cluster/home/dclara/plant-traits-earth-v2
python scripts/1km/train.py \
    data.root_dir=$SCRATCH/ \
