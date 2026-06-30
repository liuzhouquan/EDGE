#!/bin/bash
#SBATCH --job-name=edge-juke-train
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/juke-train-output.log
#SBATCH --error=logs/juke-train-error.log
#SBATCH --gres=gpu:1

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

export WANDB_MODE=disabled

# ── Jukebox 正式训练（双人模式，fine-tune from checkpoint.pt）────────────
#
# cond = cat([主舞动作(151), jukebox特征(4800)], dim=-1) = 4951维
# 每 50 epoch：保存 checkpoint + 生成 2 个视频 + 计算 LMA 得分
# 每 10 epoch：保存 latest.pt（断点续训用）
#
# 断点续训：将 --checkpoint 改为 runs/train/exp/weights/latest.pt
# ─────────────────────────────────────────────────────────────────────────

accelerate launch train.py \
  --batch_size 64 \
  --epochs 1000 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.15 \
  --drop_prob_lead 0.15 \
  --save_latest_interval 10 \
  --save_interval 50 \
  --checkpoint checkpoint.pt
