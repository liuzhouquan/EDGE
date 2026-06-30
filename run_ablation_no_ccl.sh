#!/bin/bash
#SBATCH --job-name=edge-abl-noccl
#SBATCH --time=15:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ablation_no_ccl-output.log
#SBATCH --error=logs/ablation_no_ccl-error.log
#SBATCH --gres=gpu:1

# ── 消融 A：去掉 Contact Consistency Loss (CCL / foot-skate loss) ───────────────
# 目的：复现 EDGE 论文 Tab.3 的结论——关闭 CCL 会让 PFC（物理合理性）变差。
# 对照基准：exp9（完整 CCL，2000 epoch）。本次只改一个变量 (--no_ccl)，其余
# 全部与 exp9 主训练一致，保证 apples-to-apples。
# 预计单卡 H200 ~7.5h（13.6 s/epoch × 2000）。
# ──────────────────────────────────────────────────────────────────────────────

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs
export WANDB_MODE=disabled

accelerate launch train.py \
  --exp_name ablation_no_ccl \
  --batch_size 128 \
  --epochs 1500 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --duet \
  --drop_prob_music 0.15 \
  --drop_prob_lead 0.15 \
  --no_ccl \
  --save_latest_interval 10 \
  --save_interval 100
