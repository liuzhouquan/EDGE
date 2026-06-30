#!/bin/bash
#SBATCH --job-name=edge-sweep
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/sweep-output.log
#SBATCH --error=logs/sweep-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Full sweep (PFC + LMA) 开始: $(date) ==="
echo "策略：epoch_0050..1000 复用现有 .pkl，1050..2000 走推理"
echo ""

python eval/run_full_sweep.py \
    --weights_dir   runs/train/exp9/weights \
    --pkl_dir       eval/pfc_val \
    --out_dir       eval/full_sweep \
    --guidance_music 2.0 \
    --guidance_lead  2.0

echo ""
echo "=== Full sweep 完成: $(date) ==="
