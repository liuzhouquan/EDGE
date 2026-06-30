#!/bin/bash
#SBATCH --job-name=edge-smoke
#SBATCH --time=00:20:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/smoke-output.log
#SBATCH --error=logs/smoke-error.log
#SBATCH --gres=gpu:1

# Smoke test: verify (1) GPU guard passes / runs on cuda, (2) the val-loader
# drop_last=False fix removes the epoch-save ZeroDivisionError, (3) per-epoch
# speed is sane (~15s/epoch). 2 epochs, save every 2 → triggers val + save.
cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs
export WANDB_MODE=disabled

accelerate launch train.py \
  --exp_name smoke_test \
  --batch_size 128 \
  --epochs 2 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.15 \
  --drop_prob_lead 0.15 \
  --save_latest_interval 2 \
  --save_interval 2
