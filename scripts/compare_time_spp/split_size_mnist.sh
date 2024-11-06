#!/bin/bash -e
#SBATCH --job-name=split_size
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/split_size.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/split_size.err
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

parent_dir="saved_mnist"
exp_type="split_size"
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 100 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 200 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 300 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 400 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 500 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 600 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 700 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 800 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 900 --num_projections 10000 --num_classes 10
python split_mnist.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 1000 --num_projections 10000 --num_classes 10