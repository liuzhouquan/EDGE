#!/bin/bash
#SBATCH --job-name=edge-abl-eval
#SBATCH --time=02:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/ablation_eval-output.log
#SBATCH --error=logs/ablation_eval-error.log
#SBATCH --gres=gpu:1

# Evaluate the finished ablation checkpoints (full baseline + no-dropout) on the
# held-out val pairs. Only the three w=1 configs are needed for the ablation
# tables (full / lead-only / music-only), and no video rendering (numerics only),
# so each checkpoint runs in a few minutes. Reuses the already-parametrized
# eval/run_ep1800_cfg_sweep.py (it takes --checkpoint / --eval_root / --cfg_filter).
cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs
export WANDB_MODE=disabled

CFGS="full_w1.0,lead_w1.0,music_w1.0"

echo "================ ablation_full (train-1500) ================"
python eval/run_ep1800_cfg_sweep.py \
  --checkpoint runs/train/ablation_full/weights/train-1500.pt \
  --eval_root  eval/ablation_full_eval \
  --cfg_filter "$CFGS" \
  --no_render

echo "================ ablation_no_dropout (train-1500) ================"
python eval/run_ep1800_cfg_sweep.py \
  --checkpoint runs/train/ablation_no_dropout/weights/train-1500.pt \
  --eval_root  eval/ablation_no_dropout_eval \
  --cfg_filter "$CFGS" \
  --no_render

echo "================ DONE ================"
