#!/bin/bash
#SBATCH --job-name=edge-test-longmode
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/test-longmode-output.log
#SBATCH --error=logs/test-longmode-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Per-song long-mode test eval (EDGE-compatible) 开始: $(date) ==="
echo ""

python eval/run_test_eval_longmode.py \
    --weights_dir runs/train/exp9/weights \
    --out_dir eval/test_eval \
    --ckpts_lead_only 1700,1750,1800,1850,1900 \
    --ckpts_full      1750,1800,1850,1900,1950

echo ""
echo "=== 完成: $(date) ==="
