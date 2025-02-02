#!/bin/bash
#SBATCH --job-name=js25_eval_lr_dp
#SBATCH --output=js25_eval_lr_dp_%j.out 
#SBATCH --error=js25_eval_lr_dp_%j.err
#SBATCH --partition=3090-gcondo
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --constraint=a5000


source ./js_kaggle_venv/bin/activate

# 20250111_022355_lr0.0007_log/mlp_model_step_2000.pth
# 20250111_022325_lr0.0002_log/mlp_model_step_2000.pth
# 20250111_022312_lr7e-05_log/mlp_model_step_2000.pth
# 20250111_022225_lr0.002_log/mlp_model_step_2000.pth
# 20250111_022155_lr0.0007_log/mlp_model_step_2000.pth
# 20250111_022133_lr0.0002_log/mlp_model_step_2000.pth
# 20250111_022103_lr7e-05_log/mlp_model_step_2000.pth

# MODEL_SAVE_PATH="20250111_022443_lr0.002_log/mlp_model_step_2000.pth"

VAL_DATA_PATH="./preprocessed_dataset/validation.parquet"
STATS_PATH="./preprocessed_dataset/stats.parquet"
FEAT_IN_DIMS=113
CAT_IN_DIMS=(23 10 32)
HIDDEN_LAYERS=(128 128 256 256)
EMB_DIMS=32
PROJ_DIMS=64
DROPOUT=0.2

echo $MODEL_SAVE_PATH

python js25_eval.py \
  --model-save-path "$MODEL_SAVE_PATH" \
  --val-data-path "$VAL_DATA_PATH" \
  --stats-path "$STATS_PATH" \
  --feat-in-dims $FEAT_IN_DIMS \
  --cat-in-dims "${CAT_IN_DIMS[@]}" \
  --hidden-layers "${HIDDEN_LAYERS[@]}" \
  --emb-dims $EMB_DIMS \
  --proj-dims $PROJ_DIMS \
  --dropout $DROPOUT
