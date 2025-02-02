#!/bin/bash
#SBATCH --job-name=js25_lr_dp
#SBATCH --output=js25_lr_dp_%j.out 
#SBATCH --error=js25_lr_dp_%j.err
#SBATCH --partition=3090-gcondo
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --constraint=a5000

# Activate virtual environment
source ./js_kaggle_venv/bin/activate

# Extract the passed parameters
lr=$1
dropout=$2
job_name=$3


echo "Job $job_name submitted, lr $lr, dropout $dropout"
python js25_train.py --batch_size 8192 \
                     --lr $lr \
                     --dropout $dropout \
                     --hidden_layers 32 32 64 64 128 128 64 64 32 32 \
                     --emb_dims 8 \
                     --proj_dims 32 \
                     --total_iters 4000 \
                     --log_every 20 \
                     --eval_every 100 \
                     --save_every 500 \
                     --r2_loss True \
                     --tf32 True \
                     --mask_p 0.0

echo "Job $job_name done"

# proj_dims 128
# hidden_layers 128 128 256 256

# --hidden_layers 128 128 256 256 
# dp02 dp04

# 6, stopped some
# try nn architecture 1
# emb_dims = 32
# proj_dims = 64
# [64, 128, 256, 128, 64, 32]
# dp01 dp03

# 4
# try nn architecture 2
# emb_dims = 8
# proj_dims = 32
# 32 32 64 64 128 128 64 64 32 32 
# dp01 dp03

# 4
# try attention_nn architecture 1
# emb_dims = 8
# proj_dims = 32
# 32 32 64 64 128 128 64 64 32 32 
# dp01 dp03

# 4
# try attention_nn architecture 2
# emb_dims = 32
# proj_dims = 128
# 128, 128, 256, 256, 512, 512, 256, 256, 128, 128
# dp01 dp03

# 4 date_id>1100
# try nn architecture 2
# emb_dims = 8
# proj_dims = 32
# 32 32 64 64 128 128 64 64 32 32 
# dp01 dp03