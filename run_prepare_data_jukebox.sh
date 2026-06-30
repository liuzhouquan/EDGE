#!/bin/bash
#SBATCH --job-name=edge-juke
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/juke-output.log
#SBATCH --error=logs/juke-output.log
#SBATCH --gres=gpu:1

# Jukebox 特征提取（GPU 必须）
#
# 前提：run_prepare_data.sh 已完成（motions_sliced + wavs_sliced 存在）
# 输出：train/jukebox_feats/, val/jukebox_feats/, test/jukebox_feats/
#
# 完成后提交 jukebox 正式训练：
#   sbatch run_training_jukebox.sh

set -e

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

echo "=== Jukebox 特征提取开始: $(date) ==="

# ── Step 1: 预下载 Jukebox 模型（断点续传，避免 Python wget 超时中断）────
CACHE_DIR="${HOME}/.cache/jukemirlib"
REMOTE="https://openaipublic.azureedge.net/jukebox/models/5b"
mkdir -p "${CACHE_DIR}"

# 清理任何残留的临时文件
rm -f "${CACHE_DIR}"/*.tmp

echo "[1/3] 检查/下载 Jukebox 模型权重..."
for fname in vqvae.pth.tar prior_level_2.pth.tar; do
    fpath="${CACHE_DIR}/${fname}"
    if [ -f "${fpath}" ]; then
        echo "  ${fname}: 已存在 ($(du -h ${fpath} | cut -f1))"
    else
        echo "  ${fname}: 下载中（支持断点续传）..."
        wget --continue --tries=10 --waitretry=30 \
             --progress=dot:giga \
             -O "${fpath}" \
             "${REMOTE}/${fname}"
        echo "  ${fname}: 下载完成"
    fi
done

echo "[2/3] 切片 + 复制 train/val/test（如已完成则快速跳过）..."
cd data

python create_dataset.py \
    --dataset_folder edge_aistpp \
    --duet \
    --extract-jukebox

echo "[3/3] 验证输出..."
for split in train val test; do
    n_feat=$(ls ${split}/jukebox_feats/ 2>/dev/null | wc -l || echo 0)
    echo "  ${split}: ${n_feat} jukebox feat files"
done

echo "=== Jukebox 特征提取完成: $(date) ==="
echo "下一步: sbatch run_training_jukebox.sh"
