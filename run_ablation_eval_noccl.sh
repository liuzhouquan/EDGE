#!/bin/bash
#SBATCH --job-name=edge-abl-eval-noccl
#SBATCH --time=01:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=logs/ablation_eval_noccl-output.log
#SBATCH --error=logs/ablation_eval_noccl-error.log
#SBATCH --gres=gpu:1

# Evaluate the finished no-CCL ablation checkpoint on the held-out val pairs.
# Same protocol as run_ablation_eval.sh: only the three w=1 configs, no rendering
# (numerics only), so it runs in a few minutes. Fills the -CCL row of tab:abl-ccl.
cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs
export WANDB_MODE=disabled

CFGS="full_w1.0,lead_w1.0,music_w1.0"

echo "================ ablation_no_ccl (train-1500) ================"
python eval/run_ep1800_cfg_sweep.py \
  --checkpoint runs/train/ablation_no_ccl/weights/train-1500.pt \
  --eval_root  eval/ablation_no_ccl_eval \
  --cfg_filter "$CFGS" \
  --no_render

echo "================ DONE ================"
