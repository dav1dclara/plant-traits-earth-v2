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

# Copy zarr zips to local NVMe scratch for fast random access
CHIPS_DIR=/cluster/work/igp_psr/plant-traits-earth-v2/data/1km/chips/patch128_stride64
LOCAL=$TMPDIR/chips
mkdir -p $LOCAL

echo "Copying chips to local scratch ($TMPDIR)..."
cp $CHIPS_DIR/train.zarr.zip $CHIPS_DIR/val.zarr.zip $CHIPS_DIR/test.zarr.zip $LOCAL/
echo "Done. $(du -sh $LOCAL)"

# Run training
cd ~/plant-traits-earth-v2
python scripts/1km/train.py \
    +data.zarr_dir=$LOCAL \
    data_loaders.num_workers=4
