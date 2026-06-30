#!/bin/bash
#SBATCH --job-name=edge-render-v2
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/render-test-best-v2-output.log
#SBATCH --error=logs/render-test-best-v2-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Test-best 渲染 v2（修复 stride bug）开始: $(date) ==="
echo "Fix: render.py 现在挑 every-5th dense slice → 2.5s stride，匹配 long_ddim 拼接逻辑"
echo "Split: test (ch02), 3 pairs"
echo ""

echo ">>> [1/3] full @ 1750 (test long-mode 中 full 最佳 PFC=3.597)"
python render.py \
    --checkpoint runs/train/exp9/weights/train-1750.pt \
    --split test --n_pairs 3 \
    --out renders/test_full_1750 \
    --guidance_music 2.0 --guidance_lead 2.0

echo ""
echo ">>> [2/3] full @ 1950 (test long-mode 中 full 最佳 LMA=0.9271)"
python render.py \
    --checkpoint runs/train/exp9/weights/train-1950.pt \
    --split test --n_pairs 3 \
    --out renders/test_full_1950 \
    --guidance_music 2.0 --guidance_lead 2.0

echo ""
echo ">>> [3/3] lead_only @ 1900 (test long-mode 双指标 best: PFC=2.824, LMA=0.9262)"
python render.py \
    --checkpoint runs/train/exp9/weights/train-1900.pt \
    --split test --n_pairs 3 \
    --out renders/test_leadonly_1900 \
    --guidance_music 0.0 --guidance_lead 2.0

echo ""
echo "=== 渲染完成: $(date) ==="
