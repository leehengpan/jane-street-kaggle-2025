#!/bin/bash
#SBATCH --job-name=preprocess_dataset
#SBATCH --output=preprocess_dataset_%j.out 
#SBATCH --error=preprocess_dataset_%j.err
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=12:00:00

source ./js_kaggle_venv/bin/activate

python dataset_preprocessing.py

echo done