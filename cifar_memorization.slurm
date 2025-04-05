#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --time=100:00:00
#SBATCH --job-name=cifar_memorization
#SBATCH --account=behk-delta-gpu
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

module load python/miniforge3_pytorch
module list
echo "job is starting on `hostname`"

time srun \
  python cifar_memorization.py

exit