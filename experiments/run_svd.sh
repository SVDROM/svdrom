#!/bin/sh
#SBATCH --qos turing
#SBATCH --account vjgo8416-dmd-ddwm
#SBATCH --time 8:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-gpu 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem 256G
#SBATCH --job-name run_svd
#SBATCH --output run_svd.log

source ../load_python.sh

source ../.venv/bin/activate
echo $(which python)

python run_svd.py
