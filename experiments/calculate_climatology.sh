#!/bin/sh
#SBATCH --qos turing
#SBATCH --account vjgo8416-dmd-ddwm
#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-gpu 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem 128G
#SBATCH --job-name calculate_climatology
#SBATCH --output calculate_climatology.log

source ../load_python.sh

source ../.venv/bin/activate
echo $(which python)
uv pip install "../.[extras]"

python calculate_climatology.py
