# EDGE 双人舞项目 — 服务器部署清单

> 照着这个清单在 HPC 上从头配好环境、跑通训练。
> 遇到问题直接用 Claude Code 提问。
>
> **当前状态**：代码已完成（单人模式 + 双人模式），待服务器验证。

---

## Step 1：克隆代码

```bash
git clone https://github.com/liuzhouquan/EDGE.git
cd EDGE
```

---

## Step 2：创建 conda 环境

```bash
conda env create -f environment-linux.yml
conda activate edge
```

然后手动安装 pytorch3d（预构建轮子，无需编译，约 5 分钟）：
```bash
pip install pytorch3d \
  --extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
```

验证安装：
```bash
python -c "import pytorch3d; print('pytorch3d OK')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Step 3：配置 accelerate

```bash
accelerate config
```

选择：单机单 GPU、fp16（half precision）。

---

## Step 4：下载预训练单人模型（原始 EDGE checkpoint）

```bash
bash download_model.sh
```

> 如果网络受限，手动把 `checkpoint.pt` 上传到项目根目录。

---

## Step 5：准备 AIST++ 数据集

数据文件不进 GitHub，需要单独上传到服务器。

### 5a. 把原始数据传到服务器

在**本机**执行，把外置硬盘上的 edge_aistpp 传过去：
```bash
rsync -avP --progress \
  /Volumes/gm7000/EDGE/data/edge_aistpp/ \
  <username>@<server>:~/EDGE/data/edge_aistpp/
```

> `edge_aistpp/` 需包含 `motions/`（1408 个 .pkl）和 `wavs/`（1408 个 .wav）两个子目录。

### 5b. 在服务器上一键预处理

```bash
bash prepare_data.sh
```

脚本会自动：
1. 检查 `data/edge_aistpp/` 是否就绪
2. 调用 `data/create_dataset.py` 做 train/test 切片 + baseline 音频特征提取
3. 输出验证结果

处理完后目录结构：
```
data/
  edge_aistpp/        ← 原始数据（rsync 过来）
  train/
    motions_sliced/
    baseline_feats/
    wavs_sliced/
  test/
    （同上）
```

> ⚠️ jukebox 特征需要 GPU 且耗时很长（数小时），按需手动开启：
> ```bash
> cd data && python create_dataset.py --dataset_folder edge_aistpp --extract-jukebox
> ```

---

## Step 6：验证单人模式能跑通（必做）

用 baseline 特征（不需要 jukebox GPU 特征）先跑 5 个 epoch 验证整个链路：

```bash
accelerate launch train.py \
  --batch_size 4 \
  --epochs 5 \
  --feature_type baseline
```

看到 `[MODEL SAVED at Epoch 5]` 或类似输出即表示成功。

---

## Step 7：正式训练单人模式（可选，跳到双人也行）

```bash
# 修改 run_training.sh 里的 <your_username>，然后：
mkdir -p logs
sbatch run_training.sh

# 查看 job 状态
squeue -u $USER
tail -f logs/train-output.log
```

---

## Step 8：准备双人模式数据

```bash
cd data

# 生成双人专用的 train/val/test 分割文件（如果还没有的话）
python create_duet_splits.py

# 切片 + 提取特征（三分区）
python create_dataset.py \
  --dataset_folder edge_aistpp \
  --duet \
  --extract-baseline     # 先用 baseline 验证，确认后换 --extract-jukebox

cd ..
```

---

## Step 9：验证双人模式能跑通

```bash
accelerate launch train.py \
  --batch_size 4 \
  --epochs 5 \
  --feature_type baseline \
  --duet
```

---

## Step 10：正式训练双人模式

在 `run_training.sh` 里取消注释双人模式那段命令，然后提交 job：

```bash
sbatch run_training.sh
```

**断点续训**（job 被中止后重新提交时用）：
```bash
# 在 run_training.sh 里改用断点续训命令，指向最近的 latest.pt：
# --checkpoint runs/train/exp/weights/latest.pt
sbatch run_training.sh
```

训练过程中会自动保存：
- `latest.pt` — 每 10 epoch 覆盖，最多损失 10 epoch
- `train-100.pt` / `train-200.pt` ... — 每 100 epoch 永久存档

---

## Step 11：评测

```bash
# LMA 基线（AIST++ 原始数据的参考值，已有报告：eval/aistpp_lma_baseline_report.txt）
python eval/aistpp_lma_baseline.py \
  --data_dir data/edge_aistpp/motions \
  --out eval/baseline_report.txt

# 生成双人舞推理结果
python test.py \
  --feature_type jukebox \
  --checkpoint runs/train/exp/weights/train-2000.pt \
  --duet \
  --lead_motion_dir path/to/lead_slices/ \
  --music_dir data/test_music/ \
  --save_motions

# 对生成结果跑 LMA 相似度评测
python eval/lma_similarity.py \
  --lead_dir eval/motions/lead/ \
  --follower_dir eval/motions/follower/

# 物理脚部接触指标
python eval/eval_pfc.py --motion_path eval/motions/
```

---

## 常用命令速查

```bash
squeue -u $USER                          # 查看 job 队列
scancel <job_id>                         # 取消 job
tail -f logs/train-output.log            # 实时查看训练日志
ls runs/train/exp/weights/               # 查看已保存的 checkpoint
```
