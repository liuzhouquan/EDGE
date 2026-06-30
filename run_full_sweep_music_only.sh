#!/bin/bash
#SBATCH --job-name=edge-sweep-music
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/sweep-music-output.log
#SBATCH --error=logs/sweep-music-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== music_only sweep (PFC + LMA) 开始: $(date) ==="
echo "配置: guidance_music=2.0  guidance_lead=0.0   ← 只用音乐（忽略主舞）"
echo "策略: 40 个 epoch (50→2000) 全部重新推理（不能复用 full/lead 的 .pkl）"
echo "目的: 完成 3-config ablation，找 music_only 真正的 best checkpoint"
echo ""

python eval/run_full_sweep.py \
    --weights_dir   runs/train/exp9/weights \
    --pkl_dir       eval/pfc_val_music_only \
    --out_dir       eval/full_sweep_music_only \
    --guidance_music 2.0 \
    --guidance_lead  0.0

echo ""
echo "=== music_only sweep 完成: $(date) ==="
