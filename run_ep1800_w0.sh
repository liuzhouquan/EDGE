#!/bin/bash
#SBATCH --job-name=edge-ep1800-w0
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/ep1800-w0-output.log
#SBATCH --error=logs/ep1800-w0-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== ep1800 w=0 (pure unconditional) render 开始: $(date) ==="
echo ""
echo "目的: 在 ep1800 checkpoint 上渲染纯无条件 (w_music=0, w_lead=0) 视频。"
echo "      三路 CFG 塌缩成 eps = eps_unc，follower 从模型学到的无条件先验"
echo "      采样，不跟主舞协同、不跟音乐对拍 —— 作为对照组 (ablation control)。"
echo "      输出 -> renders/ep1800_cfg_sweep/w0/  (val 5 longest pairs)"
echo ""

python eval/run_ep1800_cfg_sweep.py \
    --checkpoint runs/train/exp9/weights/train-1800.pt \
    --render_root renders/ep1800_cfg_sweep \
    --include_uncond \
    --cfg_filter w0 \
    --skip_numerics

echo ""
echo "=== ep1800 w=0 render 完成: $(date) ==="
