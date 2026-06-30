#!/bin/bash
#SBATCH --job-name=edge-abl-nodrop
#SBATCH --time=15:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ablation_no_dropout-output.log
#SBATCH --error=logs/ablation_no_dropout-error.log
#SBATCH --gres=gpu:1

# ── 消融 B：去掉输入级条件 dropout（drop_prob_lead = drop_prob_music = 0）────────
# 目的：验证两层条件 dropout 的必要性——没有输入级 dropout 时，模型见不到
# (lead, ∅) / (∅, music) 的偏 regime，预期 lead-only / music-only 推理会退化。
# 对照基准：exp9（drop=0.15/0.15）。仍保留 EDGE token 级 cond_drop_prob=0.25。
# 只改这一个变量，其余与 exp9 主训练一致。预计单卡 H200 ~7.5h。
# ──────────────────────────────────────────────────────────────────────────────

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs
export WANDB_MODE=disabled

accelerate launch train.py \
  --exp_name ablation_no_dropout \
  --batch_size 128 \
  --epochs 1500 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.0 \
  --drop_prob_lead 0.0 \
  --save_latest_interval 10 \
  --save_interval 100
