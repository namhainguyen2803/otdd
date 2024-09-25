#!/bin/bash -e
#SBATCH --job-name=run12
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/res_run12.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/err_run12.err
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

dataset="YelpReviewPolarity"
epochs=2
python3 text_cls_transfer.py --dataset "$dataset" --num-epochs $epochs

dataset="YelpReviewFull"
epochs=2
python3 text_cls_transfer.py --dataset "$dataset" --num-epochs $epochs