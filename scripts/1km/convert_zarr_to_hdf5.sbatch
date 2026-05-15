#!/bin/bash
#SBATCH --job-name=zarr-to-hdf5
#SBATCH --output=/cluster/home/dclara/plant-traits-earth-v2/logs/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --account=es_schin

module restore plant-traits
source ~/venvs/plant-traits/bin/activate

cd ~/plant-traits-earth-v2
python scripts/1km/convert_zarr_to_hdf5.py \
    --chips-dir /cluster/work/igp_psr/plant-traits-earth-v2/data/1km/chips/patch128_stride64 \
    --no-compress
