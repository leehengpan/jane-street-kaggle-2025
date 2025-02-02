#!/bin/bash

# Array of all model paths
MODEL_PATHS=(
  "./20250111_022443_lr0.002_log/mlp_model_step_2000.pth"
  "./20250111_022355_lr0.0007_log/mlp_model_step_2000.pth"
  "./20250111_022325_lr0.0002_log/mlp_model_step_2000.pth"
  "./20250111_022312_lr7e-05_log/mlp_model_step_2000.pth"
  "./20250111_022225_lr0.002_log/mlp_model_step_2000.pth"
  "./20250111_022155_lr0.0007_log/mlp_model_step_2000.pth"
  "./20250111_022133_lr0.0002_log/mlp_model_step_2000.pth"
  "./20250111_022103_lr7e-05_log/mlp_model_step_2000.pth"
)

for MODEL_SAVE_PATH in "${MODEL_PATHS[@]}"; do
  echo "Submitting job for model: $MODEL_SAVE_PATH"
  sbatch --export=MODEL_SAVE_PATH="$MODEL_SAVE_PATH" launch_js25_eval.sh
done
