#!/bin/bash
#SBATCH --job-name=submit_jobs
#SBATCH --output=submit_jobs_%j.out 
#SBATCH --error=submit_jobs_%j.err
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00


# Define arrays for dropout and learning rate values
dropouts=(0.1 0.3)
lrs=(7e-4 1e-3)

# Loop through all combinations of dropout and learning rate
for dropout in "${dropouts[@]}"; do
    for lr in "${lrs[@]}"; do
        job_name="js25_train_lr${lr}_dp${dropout//.}"
        echo "Submitting job: $job_name"

        # Submit the job with the current combination of dropout and lr
        sbatch launch_js25_train.sh $lr $dropout $job_name

        # Wait for 20 seconds before submitting the next job
        echo "Waiting for 25 seconds before submitting the next job..."
        sleep 30
    done
done

# lrs=(7e-5 2e-4 7e-4)
# new lrs


# 1. High dropout (0.5) consistently performed better at validation than low dropout (0.3).
# 2. Small model has no significant difference vs large model, sometimes small model performed better, but the trend is not consistent.
# 3. Use bn or not use bn, has no significant difference in the beginning of training. 


# 8 experiments, (4 lr, 2 dp)
# 1. Use z score 
# 2. Use z score + bn

# No norm
