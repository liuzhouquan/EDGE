#!/bin/bash
#SBATCH --job-name=edge-train
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/train-output.log
#SBATCH --error=logs/train-error.log
#SBATCH --gres=gpu:1

cd /data/<your_username>/EDGE

source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge

mkdir -p logs

# ── 模式说明 ──────────────────────────────────────────────────────────────────
#
# 【单人模式】原始 EDGE 逻辑：输入=音乐特征(4800维)，输出=单人舞动作
#   条件维度: cond_feature_dim=4800
#   数据集:   AISTPPDataset（data/train/ 和 data/test/）
#   权重:     可从官方预训练 checkpoint.pt 继续训练
#
# 【双人模式】扩展逻辑：输入=主舞动作(151维)+音乐特征(4800维)，输出=伴舞动作
#   条件维度: cond_feature_dim=4951
#   数据集:   DuetDataset（同一 music_id 下不同舞者配对）
#   权重:     与单人模式不兼容，需要单独训练
#             可用 --checkpoint 指定原始权重做 fine-tune（cond_projection 层会重新初始化）
#
# 注意：两种模式的 checkpoint 不能互换加载，因为 cond_projection 层维度不同。
# ─────────────────────────────────────────────────────────────────────────────

# ── 断点续训说明 ──────────────────────────────────────────────────────────────
#
# 训练过程中会保存两类文件（均在 runs/train/exp/weights/ 下）：
#   latest.pt       — 每 10 epoch 覆盖一次，job 被中止时最多损失 10 个 epoch
#   train-100.pt    — 每 100 epoch 保存一次永久存档，用于回滚或分析
#
# 断点续训：只需在 --checkpoint 指向最近的 latest.pt 或 train-N.pt，
#           训练会自动从上次保存的 epoch+1 继续，无需手动指定 epoch。
#
# ─────────────────────────────────────────────────────────────────────────────

# ── 单人模式训练（默认，兼容官方预训练权重）────────────────────────────────
accelerate launch train.py \
  --batch_size 128 \
  --epochs 2000 \
  --feature_type jukebox \
  --learning_rate 0.0002 \
  --save_latest_interval 10 \
  --save_interval 100

# ── 单人模式断点续训（被中止后用这条命令重新提交）─────────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --save_latest_interval 10 \
#   --save_interval 100 \
#   --checkpoint runs/train/exp/weights/latest.pt

# ── 双人模式训练 ───────────────────────────────────────────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --duet \
#   --save_latest_interval 10 \
#   --save_interval 100

# ── 双人模式断点续训 ────────────────────────────────────────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --duet \
#   --save_latest_interval 10 \
#   --save_interval 100 \
#   --checkpoint runs/train/exp/weights/latest.pt

# ── 从单人权重 fine-tune 双人模式（推荐，比从头训练快）─────────────────────
# accelerate launch train.py \
#   --batch_size 128 \
#   --epochs 2000 \
#   --feature_type jukebox \
#   --learning_rate 0.0002 \
#   --duet \
#   --save_latest_interval 10 \
#   --save_interval 100 \
#   --checkpoint checkpoint.pt
