#!/bin/bash -e
#SBATCH --job-name=aug
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/aug14.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/aug14.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

module purge
module load python/miniconda3/miniconda3

# Corrected line
eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/vinai/users/hainn14/envs/otdd
cd /lustre/scratch/client/vinai/users/hainn14/otdd

python3 augmentation_exp2.py --seed 14