#!/bin/bash -e
#SBATCH --job-name=demo
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/res.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/res.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io

python3 augmentation_exp.py