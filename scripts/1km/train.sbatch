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
source ~/venvs/plant-traits/bin/activate

CHIPS_SRC=/cluster/work/igp_psr/plant-traits-earth-v2/data/1km/chips/patch128_stride64

# Option A (default): copy .h5 files to $TMPDIR for fastest local I/O.
# Option B: read directly from LUSTRE — skip the cp block and set CHIPS_DIR=$CHIPS_SRC.
#   h5py issues large sequential reads that LUSTRE handles well, so this is viable
#   when $TMPDIR is too small (~120 GB total for all three splits uncompressed).

CHIPS_DIR=$TMPDIR/chips/patch128_stride64
mkdir -p $CHIPS_DIR

for SPLIT in train val test; do
    echo "Copying ${SPLIT}.h5 to local scratch..."
    cp $CHIPS_SRC/${SPLIT}.h5 $CHIPS_DIR/
done

# Run training
cd ~/plant-traits-earth-v2
python scripts/1km/train.py \
    data.root_dir=$TMPDIR/
