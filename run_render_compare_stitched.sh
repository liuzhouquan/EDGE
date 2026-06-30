#!/bin/bash
#SBATCH --job-name=edge-render-stitched
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/render-stitched-output.log
#SBATCH --error=logs/render-stitched-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

CKPT=runs/train/exp9/weights/train-2000.pt

echo "=== 改进拼接后的渲染对比开始: $(date) ==="
echo "改动：render.py 的 _stitch_chunks 替换为 _stitch_and_fk"
echo "      （drift correction + raised-cosine cross-fade + slerp）"

# ── full：音乐 + 主舞（质量上限对照）──────────────────────────────
echo ">>> [1/2] full (w_music=2, w_lead=2) → renders/duet_2000_full_stitched"
python render.py \
    --checkpoint $CKPT \
    --n_pairs 3 \
    --out renders/duet_2000_full_stitched \
    --guidance_music 2.0 \
    --guidance_lead 2.0

# ── lead-only：只用主舞（重点）─────────────────────────────────────
echo ">>> [2/2] lead-only (w_music=0, w_lead=2) → renders/duet_2000_leadonly_stitched"
python render.py \
    --checkpoint $CKPT \
    --n_pairs 3 \
    --out renders/duet_2000_leadonly_stitched \
    --guidance_music 0.0 \
    --guidance_lead 2.0

echo "=== 渲染对比完成: $(date) ==="
