#!/bin/bash
#SBATCH --job-name=edge-train
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/train-output.log
#SBATCH --error=logs/train-error.log
#SBATCH --gres=gpu:1

cd /data/<your_username>/EDGE

source ~/miniconda3/etc/profile.d/conda.sh
conda activate edge

mkdir -p logs

accelerate launch train.py \
  --batch_size 128 \
  --epochs 2000 \
  --feature_type jukebox \
  --learning_rate 0.0002
