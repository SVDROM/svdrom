#!/bin/sh
#SBATCH --qos turing
#SBATCH --account vjgo8416-dmd-ddwm
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name download_era5
#SBATCH --output download_era5.log

source ../load_python.sh

source ../.venv/bin/activate
echo $(which python)

python download_era5_slice.py
