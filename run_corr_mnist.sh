#!/bin/bash -e
#SBATCH --job-name=corr_mnist
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/corr_mnist.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/corr_mnist.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/vinai/users/hainn14/envs/otdd
cd /lustre/scratch/client/vinai/users/hainn14/otdd

python3 correlation_mnist_experiment.py