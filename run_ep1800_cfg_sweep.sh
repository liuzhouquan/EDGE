#!/bin/bash
#SBATCH --job-name=edge-ep1800-cfg
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ep1800-cfg-sweep-output.log
#SBATCH --error=logs/ep1800-cfg-sweep-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== ep1800 CFG sweep 开始: $(date) ==="
echo ""
echo "目的: 在 ep1800 checkpoint 上对 15 个 (mode, weight) 推理配置做"
echo "      数字评估 + 视频渲染。报告统一英文，给导师审阅。"
echo ""
echo "配置:"
echo "  - 3 modes × 5 CFG points (1, 1.5, 2, 2.5, 3) = 15 configs"
echo "  - val split, 10 pairs (numerics 全, 视频 5 长 pair)"
echo "  - 总计 75 视频 + 15 个 per-config report + 1 master report"
echo ""

python eval/run_ep1800_cfg_sweep.py \
    --checkpoint runs/train/exp9/weights/train-1800.pt \
    --eval_root  eval/ep1800_render_eval \
    --render_root renders/ep1800_cfg_sweep

echo ""
echo "=== ep1800 CFG sweep 完成: $(date) ==="
