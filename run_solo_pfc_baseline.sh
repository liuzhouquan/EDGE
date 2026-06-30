#!/bin/bash
#SBATCH --job-name=edge-solo-pfc
#SBATCH --time=00:15:00
#SBATCH --open-mode=append
#SBATCH --output=logs/solo-pfc-baseline-output.log
#SBATCH --error=logs/solo-pfc-baseline-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Solo PFC baseline reproduction 开始: $(date) ==="
echo "目的: 用 EDGE 预训练 checkpoint.pt (solo mode) 在 val 上跑 PFC"
echo "      如果能复现 EDGE 论文 ~1.5 的 PFC, 说明我们的 PFC 测量方式没问题"
echo ""

python eval/run_solo_pfc_baseline.py \
    --checkpoint checkpoint.pt \
    --split      val \
    --out_dir    eval/solo_pfc_baseline

echo ""
echo "=== Solo PFC baseline 完成: $(date) ==="
