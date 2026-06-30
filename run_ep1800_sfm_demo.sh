#!/bin/bash
#SBATCH --job-name=edge-ep1800-sfm-demo
#SBATCH --time=05:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ep1800-sfm-demo-output.log
#SBATCH --error=logs/ep1800-sfm-demo-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== ep1800 sFM demo 开始: $(date) ==="
echo ""
echo "目的: 10 个 sFM 自由编舞 demo (每 genre 最长 1 条),"
echo "      在 ep1800 checkpoint 上跑 15 配置 (full/lead/music × 5 w), 仅渲染。"
echo ""
echo "Step A: 数据预处理"
echo "  - 选 sFM 序列 (每 genre 最长), 转 raw->{pos,q,scale}, 复制 wav"
echo "  - slice 5s/0.5s-stride windows"
echo "  - 提 jukebox features (GPU)"
echo ""
echo "Step B: 渲染推理"
echo "  - 用 sFM 自身当作 lead 输入, 模型生成 follower"
echo "  - 视频里 lead = 原 sFM (灰), follower = 模型生成 (彩)"
echo "  - 跳过 PFC + LMA + 报告 (--skip_numerics, 单 sFM 无 GT follower)"
echo "  - 输出 -> renders/ep1800_cfg_sweep/<cfg>/sFM/"
echo "  - 总计 15 配置 × 10 序列 = 150 视频"
echo ""

# Step A: prepare data
echo "--- Step A: prepare sFM data ---"
python prepare_sfm_demo.py
if [ $? -ne 0 ]; then
    echo "FATAL: prepare_sfm_demo.py failed"
    exit 1
fi

# Step B: render
echo ""
echo "--- Step B: 15-config × 10-seq render ---"
python eval/run_ep1800_cfg_sweep.py \
    --checkpoint runs/train/exp9/weights/train-1800.pt \
    --eval_root  eval/ep1800_sfm_demo_eval \
    --render_root renders/ep1800_cfg_sweep \
    --render_subdir sFM \
    --split_subdir sFM_demo \
    --pairs_json data/splits/sFM_demo_pairs.json \
    --render_indices 0,1,2,3,4,5,6,7,8,9 \
    --skip_numerics

echo ""
echo "=== ep1800 sFM demo 完成: $(date) ==="
