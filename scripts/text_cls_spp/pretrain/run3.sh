#!/bin/bash -e
#SBATCH --job-name=run34
#SBATCH --output=/lustre/scratch/client/vinai/users/hainn14/otdd/res_run34.out
#SBATCH --error=/lustre/scratch/client/vinai/users/hainn14/otdd/err_run34.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.HaiNN14@vinai.io
module purge
module load python/miniconda3/miniconda3
eval “$(conda shell.bash hook)”
conda activate /lustre/scratch/client/vinai/users/hainn14/otdd
cd /lustre/scratch/client/vinai/users/hainn14/otdd



dataset="YelpReviewPolarity"
epochs=10
python3 text_cls.py --dataset "$dataset" --num-epochs $epochs

dataset="YelpReviewFull"
epochs=10
python3 text_cls.py --dataset "$dataset" --num-epochs $epochs