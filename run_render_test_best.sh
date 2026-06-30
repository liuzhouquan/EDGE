#!/bin/bash
#SBATCH --job-name=edge-render-test-best
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/render-test-best-output.log
#SBATCH --error=logs/render-test-best-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Test-best checkpoint 渲染开始: $(date) ==="
echo "Split: test (ch02), 3 pairs"
echo ""

echo ">>> [1/3] full @ 1750 (test best PFC=3.140) → renders/test_full_1750"
python render.py \
    --checkpoint runs/train/exp9/weights/train-1750.pt \
    --split test --n_pairs 3 \
    --out renders/test_full_1750 \
    --guidance_music 2.0 --guidance_lead 2.0

echo ""
echo ">>> [2/3] full @ 1950 (test best LMA=0.9305) → renders/test_full_1950"
python render.py \
    --checkpoint runs/train/exp9/weights/train-1950.pt \
    --split test --n_pairs 3 \
    --out renders/test_full_1950 \
    --guidance_music 2.0 --guidance_lead 2.0

echo ""
echo ">>> [3/3] lead_only @ 1900 (test best both: PFC=3.205, LMA=0.9286) → renders/test_leadonly_1900"
python render.py \
    --checkpoint runs/train/exp9/weights/train-1900.pt \
    --split test --n_pairs 3 \
    --out renders/test_leadonly_1900 \
    --guidance_music 0.0 --guidance_lead 2.0

echo ""
echo "=== 渲染完成: $(date) ==="
