#!/bin/bash
#SBATCH --job-name=edge-render-cmp
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/render-cmp-output.log
#SBATCH --error=logs/render-cmp-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

CKPT=runs/train/exp9/weights/train-2000.pt

echo "=== 渲染对比开始: $(date) ==="

# ── full：音乐 + 主舞（质量上限对照）──────────────────────────────
echo ">>> [1/2] full (w_music=2, w_lead=2)"
python render.py \
    --checkpoint $CKPT \
    --n_pairs 3 \
    --out renders/duet_2000_full \
    --guidance_music 2.0 \
    --guidance_lead 2.0

# ── lead-only：只用主舞（重点）─────────────────────────────────────
echo ">>> [2/2] lead-only (w_music=0, w_lead=2)"
python render.py \
    --checkpoint $CKPT \
    --n_pairs 3 \
    --out renders/duet_2000_leadonly \
    --guidance_music 0.0 \
    --guidance_lead 2.0

echo "=== 渲染对比完成: $(date) ==="
