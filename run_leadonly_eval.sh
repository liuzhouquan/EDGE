#!/bin/bash
#SBATCH --job-name=edge-leadonly
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/leadonly-eval-output.log
#SBATCH --error=logs/leadonly-eval-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

echo "=== lead-only 评估开始: $(date) ==="

python eval/run_leadonly_eval.py \
    --checkpoint runs/train/exp9/weights/train-2000.pt \
    --data_dir data \
    --out_dir eval/leadonly_2000 \
    --feature_type jukebox

echo "=== lead-only 评估完成: $(date) ==="
