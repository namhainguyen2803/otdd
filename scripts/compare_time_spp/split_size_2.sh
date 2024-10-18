#!/bin/bash -e
#SBATCH --job-name=split_size2
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/split_size2.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/split_size2.err
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

parent_dir="saved_2"
exp_type="split_size"
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 200 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 500 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 1000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 2000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 8000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 10000 --num_projections 10000 --num_classes 100