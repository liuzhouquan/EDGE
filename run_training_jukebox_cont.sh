#!/bin/bash
#SBATCH --job-name=edge-juke-cont
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/juke-train-output.log
#SBATCH --error=logs/juke-train-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

export WANDB_MODE=disabled

# ── Jukebox 续训（从 epoch 1000 继续到 epoch 2000）────────────────────────
#
# 接续 exp7 的训练结果，从 latest.pt 断点续训
# ─────────────────────────────────────────────────────────────────────────

accelerate launch train.py \
  --batch_size 64 \
  --epochs 2000 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.15 \
  --drop_prob_lead 0.15 \
  --save_latest_interval 10 \
  --save_interval 50 \
  --checkpoint runs/train/exp7/weights/latest.pt
