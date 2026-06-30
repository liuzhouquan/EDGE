#!/bin/bash
#SBATCH --job-name=edge-contact-diag
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=logs/contact-diag-output.log
#SBATCH --error=logs/contact-diag-error.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /data/zliu753/EDGE
source /data/zliu753/init_env.sh
conda activate edge
mkdir -p logs

echo "=== Contact 诊断开始: $(date) ==="
echo ""
echo ">>> [1/2] lead_only @ 1700 (PFC 最低点)"
python eval/diagnose_contact.py \
    --checkpoint runs/train/exp9/weights/train-1700.pt \
    --guidance_music 0.0 --guidance_lead 2.0 \
    --out_dir eval/contact_diag/lead_1700

echo ""
echo ">>> [2/2] full @ 2000 (用了 2 个条件，对照)"
python eval/diagnose_contact.py \
    --checkpoint runs/train/exp9/weights/train-2000.pt \
    --guidance_music 2.0 --guidance_lead 2.0 \
    --out_dir eval/contact_diag/full_2000

echo ""
echo "=== Contact 诊断完成: $(date) ==="
