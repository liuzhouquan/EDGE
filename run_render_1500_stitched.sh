#!/bin/bash
#SBATCH --job-name=edge-render-1500
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/render-1500-output.log
#SBATCH --error=logs/render-1500-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

CKPT=runs/train/exp9/weights/train-1500.pt

echo "=== epoch 1500 渲染对比开始: $(date) ==="
echo "改进拼接（drift correction + raised-cosine + slerp）+ 最优 epoch=1500"

echo ">>> [1/2] full (w_music=2, w_lead=2) → renders/duet_1500_full_stitched"
python render.py \
    --checkpoint $CKPT \
    --n_pairs 3 \
    --out renders/duet_1500_full_stitched \
    --guidance_music 2.0 \
    --guidance_lead 2.0

echo ">>> [2/2] lead-only (w_music=0, w_lead=2) → renders/duet_1500_leadonly_stitched"
python render.py \
    --checkpoint $CKPT \
    --n_pairs 3 \
    --out renders/duet_1500_leadonly_stitched \
    --guidance_music 0.0 \
    --guidance_lead 2.0

echo "=== 渲染完成: $(date) ==="
