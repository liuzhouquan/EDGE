#!/bin/bash
#SBATCH --job-name=edge-test-eval
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/test-eval-output.log
#SBATCH --error=logs/test-eval-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Test 集 paper-ready 评估开始: $(date) ==="
echo "Split: AIST++ crossmodal_test.txt (ch02), 10 duet pairs, 93 slices"
echo "Lead_only ckpts: 1700,1750,1800,1850,1900"
echo "Full ckpts:      1750,1800,1850,1900,1950"
echo ""

python eval/run_test_eval.py \
    --weights_dir runs/train/exp9/weights \
    --out_dir eval/test_eval \
    --ckpts_lead_only 1700,1750,1800,1850,1900 \
    --ckpts_full      1750,1800,1850,1900,1950

echo ""
echo "=== Test 评估完成: $(date) ==="
