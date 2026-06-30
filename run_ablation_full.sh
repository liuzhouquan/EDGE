#!/bin/bash
#SBATCH --job-name=edge-abl-full
#SBATCH --time=15:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ablation_full-output.log
#SBATCH --error=logs/ablation_full-error.log
#SBATCH --gres=gpu:1

# Matched FULL baseline for the ablation table (CCL on, dropout 0.15/0.15).
# Self-contained reference so the ablation does not depend on exp9's unknown
# batch size / save interval. Identical protocol to the two ablation runs.
cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs
export WANDB_MODE=disabled

accelerate launch train.py \
  --exp_name ablation_full \
  --batch_size 128 \
  --epochs 1500 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.15 \
  --drop_prob_lead 0.15 \
  --save_latest_interval 10 \
  --save_interval 100
