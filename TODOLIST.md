# 复现 EDGE — 上机操作清单

> 这是你 git clone 代码到 HPC 之后需要完成的步骤。
> 遇到问题随时用 Claude Code 提问，不需要提前把所有细节搞清楚。

---

## Step 1：克隆代码并创建 conda 环境

```bash
git clone https://github.com/liuzhouquan/EDGE.git
cd EDGE
conda env create -f environment-linux.yml
conda activate edge
```

然后手动安装 pytorch3d（预构建轮子，无需编译）：
```bash
pip install pytorch3d \
  --extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
```

---

## Step 2：配置 accelerate

```bash
accelerate config
```

选择：单机单 GPU，fp16。

---

## Step 3：下载预训练模型

```bash
bash download_model.sh
```

> 如果网络受限，手动把 `checkpoint.pt` 上传到服务器根目录。

---

## Step 4：准备 AIST++ 数据集

数据集结构需要如下（参考 `dataset/dance_dataset.py`）：
```
data/
  train/
    motions_sliced/   ← .pkl 文件
    jukebox_feats/    ← .npy 文件
    baseline_feats/   ← .npy 文件（baseline 模式用）
    wavs_sliced/      ← .wav 文件
  test/
    （同上结构）
```

> 数据准备是最耗时的步骤，遇到格式问题找 Claude Code 帮你看。

---

## Step 5：跑一次小规模验证（可选但推荐）

用 baseline 特征（无需 Jukebox GPU 特征提取）先跑通流程：
```bash
accelerate launch train.py \
  --batch_size 4 \
  --epochs 5 \
  --feature_type baseline
```

---

## Step 6：提交正式训练 Job

编辑 `run_training.sh`，把 `<your_username>` 改成你的用户名，然后：
```bash
mkdir -p logs
sbatch run_training.sh
squeue -u $USER   # 查看 job 状态
```

---

## 后续目标（代码改动，到时候再说）

- [ ] 修改 conditioning：主舞 motion + 音乐特征 → 从舞 motion
- [ ] 准备 duet 配对数据集
- [ ] 微调或重新训练
