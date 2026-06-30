#!/bin/bash
#SBATCH --job-name=edge-render
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/render-output.log
#SBATCH --error=logs/render-error.log
#SBATCH --gres=gpu:1

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge

mkdir -p logs

python render.py \
    --checkpoint runs/train/exp7/weights/train-950.pt \
    --n_pairs 3 \
    --out renders/duet_val_950 \
    --guidance_music 2.0 \
    --guidance_lead 2.0
