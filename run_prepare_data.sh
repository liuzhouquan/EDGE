#!/bin/bash
#SBATCH --job-name=edge-prep
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/prep-output.log
#SBATCH --error=logs/prep-output.log
#SBATCH --gres=gpu:0

# 数据预处理脚本（baseline 特征，CPU-only）
#
# 做了什么：
#   1. 按 duet 三路划分（train/val/test）复制 motions + wavs
#   2. 切片为 5s 窗口（stride=0.5s）
#   3. 提取 baseline 音频特征（35维，速度快，用于验证训练流程）
#
# 完成后再提交训练：
#   sbatch run_training.sh
#
# 如果后续要换 jukebox 特征（4800维），重新运行：
#   sbatch run_prepare_data_jukebox.sh

set -e

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

echo "=== 数据预处理开始: $(date) ==="

cd data

echo "[1/2] 切片 + 复制 train/val/test..."
python create_dataset.py \
    --dataset_folder edge_aistpp \
    --duet \
    --extract-baseline

echo "[2/2] 验证输出..."
for split in train val test; do
    n_motion=$(ls ${split}/motions_sliced/ 2>/dev/null | wc -l || echo 0)
    n_feat=$(ls ${split}/baseline_feats/ 2>/dev/null | wc -l || echo 0)
    echo "  ${split}: ${n_motion} motion slices, ${n_feat} baseline feat files"
done

echo "=== 数据预处理完成: $(date) ==="
echo "下一步: sbatch run_training.sh"
