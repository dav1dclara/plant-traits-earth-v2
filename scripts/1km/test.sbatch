#!/bin/bash
#SBATCH --job-name=plant-traits-1km-test
#SBATCH --output=/cluster/home/ldominiak/plant-traits-earth-v2/logs/%x_%j.log
#SBATCH --partition=gpuhe.24h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --time=12:00:00
#SBATCH --account=es_schin

module restore plant-traits
source ~/venvs/plant-traits/bin/activate

cd ~/plant-traits-earth-v2

python scripts/1km/test.py "$@"
