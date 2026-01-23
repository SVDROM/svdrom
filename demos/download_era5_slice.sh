#!/bin/sh
#SBATCH --qos turing
#SBATCH --account vjgo8416-dmd-ddwm
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name download_era5
#SBATCH --output download_era5.log

module purge
module load baskerville
module load Python/3.10.8-GCCcore-12.2.0

source ../.venv/bin/activate
echo $(which python)

python download_era5_slice.py
