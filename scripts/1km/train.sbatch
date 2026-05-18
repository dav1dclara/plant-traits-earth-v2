#!/bin/bash
#SBATCH --job-name=plant-traits-1km-simple
#SBATCH --output=/cluster/home/ldominiak/plant-traits-earth-v2/logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --account=es_schin

# Load modules and activate venv
module restore plant-traits
source ~/venvs/plant-traits/bin/activate

CHIPS_SRC=/cluster/work/igp_psr/plant-traits-earth-v2/data/1km/chips/patch128_stride64
CHIPS_DIR=$TMPDIR/chips
mkdir -p $CHIPS_DIR

echo "Copying zarr zips to local scratch ($TMPDIR)..."
cp $CHIPS_SRC/train.zarr.zip $CHIPS_SRC/val.zarr.zip $CHIPS_DIR/
echo "Done. $(du -sh $CHIPS_DIR)"

# Run training
cd ~/plant-traits-earth-v2
python scripts/1km/train.py \
    +data.zarr_dir=$CHIPS_DIR \
    train.use_tqdm=true \
    "$@"
