#!/bin/bash
#SBATCH --job-name=edge-sweep-leadonly
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/sweep-leadonly-output.log
#SBATCH --error=logs/sweep-leadonly-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== lead_only sweep (PFC + LMA) 开始: $(date) ==="
echo "配置: guidance_music=0.0  guidance_lead=2.0   ← 只用主舞"
echo "策略: 40 个 epoch 全部重新推理（不能复用 full 的 .pkl）"
echo ""

python eval/run_full_sweep.py \
    --weights_dir   runs/train/exp9/weights \
    --pkl_dir       eval/pfc_val_leadonly \
    --out_dir       eval/full_sweep_leadonly \
    --guidance_music 0.0 \
    --guidance_lead  2.0

echo ""
echo "=== lead_only sweep 完成: $(date) ==="
