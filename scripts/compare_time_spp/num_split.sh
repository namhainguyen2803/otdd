#!/bin/bash -e
#SBATCH --job-name=num_split
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/num_split.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/num_split.err
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


parent_dir="saved_ss2"
exp_type="num_split"
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 3 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 4 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 5 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 6 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 7 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 8 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 9 --split_size 5000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 10 --split_size 5000 --num_projections 10000 --num_classes 100