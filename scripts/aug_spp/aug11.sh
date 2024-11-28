#!/bin/bash -e
#SBATCH --job-name=aug11
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/aug11.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/aug11.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/vinai/users/hainn14/envs/otdd
cd /lustre/scratch/client/vinai/users/hainn14/otdd

python3 augmentation_exp3.py --seed 11