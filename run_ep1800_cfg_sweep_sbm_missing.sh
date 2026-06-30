#!/bin/bash
#SBATCH --job-name=edge-ep1800-sbm-missing
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ep1800-sbm-missing-output.log
#SBATCH --error=logs/ep1800-sbm-missing-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== ep1800 sBM 补齐渲染开始: $(date) ==="
echo ""
echo "目的: 在已有 sweep 基础上补齐另外 5 个 val pair 的视频 (idx 1,2,5,6,7),"
echo "      输出统一进 renders/ep1800_cfg_sweep/<cfg>/sBM/ 子目录。"
echo ""
echo "数据: dense per_slice pkl 已存在 -> 脚本会跳过推理, 只跑渲染。"
echo "渲染 pair: idx 1,2,5,6,7"
echo "  1 gHO_d21 (5 slices), 2 gJB_d09 (5), 5 gLH_d18 (7), 6 gLO_d15 (10), 7 gMH_d24 (8)"
echo "配置: 15 (full/lead/music × w∈{1.0, 1.5, 2.0, 2.5, 3.0})"
echo "合计 75 个新视频 -> 加上已移入子目录的 75 个 = 每配置 10 个 sBM"
echo ""

python eval/run_ep1800_cfg_sweep.py \
    --checkpoint runs/train/exp9/weights/train-1800.pt \
    --eval_root  eval/ep1800_render_eval \
    --render_root renders/ep1800_cfg_sweep \
    --render_subdir sBM \
    --render_indices 1,2,5,6,7

echo ""
echo "=== ep1800 sBM 补齐渲染完成: $(date) ==="
