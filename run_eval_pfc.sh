#!/bin/bash
#SBATCH --job-name=edge-eval-pfc
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/eval-pfc-output.log
#SBATCH --error=logs/eval-pfc-error.log
#SBATCH --gres=gpu:1

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs eval/pfc_val

echo "=== PFC 评估开始: $(date) ==="

python eval/run_val_pfc.py \
    --weights_dir runs/train/exp7/weights \
    --data_dir data \
    --out_dir eval/pfc_val \
    --guidance_music 2.0 \
    --guidance_lead 2.0 \
    --feature_type jukebox

echo "=== PFC 评估完成: $(date) ==="
