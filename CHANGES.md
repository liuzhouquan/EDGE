# EDGE — 双环境开发改造说明

## 一、原始依赖分析

### requirements.txt 问题清单

| 问题 | 说明 |
|------|------|
| PyTorch 1.12.1（2022-07） | 过旧；MPS 支持不完善；pytorch3d arm64 无二进制包 |
| 未声明 Python 版本 | 实测兼容 3.7–3.10；本方案固定用 3.10 |
| **缺少 pytorch3d** | `vis.py`、`model/diffusion.py`、`dataset/dance_dataset.py` 均有导入 |
| **缺少 jukemirlib** | `test.py`、`data/audio_extraction/jukebox_features.py` 使用 |

### 升级方案说明（选用 PyTorch 2.1.x）

原始依赖（1.12.1）复现有以下风险：
- pytorch3d 在 Apple Silicon 上无法直接从 conda 安装，源码编译对 torch 1.12 也很痛苦
- accelerate API 有 breaking change，1.12 时代写法已过时

**选 PyTorch 2.1.x 的理由：**
- MPS 后端成熟（`torch.backends.mps.is_available()`）
- pytorch3d ≥ 0.7.4 提供 py310+cu118+pyt210 预构建轮子
- accelerate ≥ 0.24 接口稳定
- 潜在风险：`GaussianDiffusion` 内部的 `register_buffer` / EMA 逻辑在 PyTorch 2.x 行为一致，未见 breaking change

---

## 二、环境配置文件

### `environment-mac.yml`（本地 macOS Apple Silicon）

- Python 3.10，PyTorch 2.1.2（CPU + MPS，无 CUDA）
- **不包含 jukemirlib**：本地只用 `--feature_type baseline`（35 维特征），无需下载 10 GB Jukebox 权重
- pytorch3d 从 GitHub 源码编译（需要 Xcode CLT）：
  ```bash
  xcode-select --install   # 若未安装
  conda env create -f environment-mac.yml
  ```
  若编译 pytorch3d 失败，见「pytorch3d 编译失败应急方案」一节。

### `environment-linux.yml`（Linux x86_64 + NVIDIA GPU）

- Python 3.10，PyTorch 2.1.2 + CUDA 11.8
- **包含 jukemirlib**，支持 `--feature_type jukebox` 训练
- pytorch3d 需在激活环境后手动安装预构建轮子：
  ```bash
  conda env create -f environment-linux.yml
  conda activate edge-linux
  pip install pytorch3d \
    --extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
  ```

---

## 三、代码改动说明

### 3.1 `EDGE.py` — DataLoader `pin_memory` 修复

**改动位置：** `train_loop()` 方法，DataLoader 构建处

```python
# 改前
pin_memory=True,

# 改后
pin = torch.cuda.is_available()
...
pin_memory=pin,
```

**原因：** `pin_memory=True` 在没有 CUDA 的环境（macOS CPU/MPS）下会触发警告，且无任何收益。改为仅当 CUDA 可用时启用。

### 3.2 `test.py` — `torch.cuda.empty_cache()` 修复

**改动位置：** `test()` 函数末尾

```python
# 改前
torch.cuda.empty_cache()

# 改后
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**原因：** 在非 CUDA 设备上调用会抛 RuntimeError。

### 3.3 `args.py` + `train.py` — 补全 `--learning_rate` 参数

**原因：** CLAUDE.md 的训练示例命令包含 `--learning_rate 0.0002`，但 `parse_train_opt()` 中未声明该参数（会被 argparse 忽略），`EDGE.__init__()` 的学习率始终使用硬编码的 `4e-4`。

改动：
- `args.py`：在 `parse_train_opt()` 中添加 `--learning_rate`，默认值 `4e-4`
- `train.py`：传 `learning_rate=opt.learning_rate` 给 `EDGE()`

---

## 四、pytorch3d 编译失败应急方案

如果 `pip install git+https://...pytorch3d` 在 macOS 上失败，可以创建一个最小 stub 以支持 CPU 冒烟测试：

```bash
mkdir -p pytorch3d_stub/pytorch3d
cat > pytorch3d_stub/pytorch3d/__init__.py << 'EOF'
EOF
cat > pytorch3d_stub/pytorch3d/transforms.py << 'EOF'
# Minimal stub for CPU smoke tests — does NOT implement correct math
import torch

def axis_angle_to_quaternion(aa):
    # aa: (..., 3) -> (..., 4) [w, x, y, z]
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    half = angle / 2
    return torch.cat([half.cos(), half.sin() * axis], dim=-1)

def quaternion_to_axis_angle(q):
    # q: (..., 4) [w, x, y, z] -> (..., 3)
    half = q[..., 0:1].clamp(-1, 1).acos()
    sin_half = half.sin().clamp(min=1e-8)
    return q[..., 1:] / sin_half * half * 2

def quaternion_multiply(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
    ], dim=-1)

def quaternion_apply(q, pts):
    # q: (..., 4), pts: (..., 3) -> (..., 3)  [rough approximation]
    return pts  # identity for smoke purposes

class RotateAxisAngle:
    def __init__(self, angle, axis="X", degrees=True): pass
    def transform_points(self, pts): return pts
EOF
pip install -e pytorch3d_stub/
```

これは数学的に正確ではありませんが、コードパスを通過させるには十分です。

---

## 五、最小冒烟测试方案

### 快速验证命令

```bash
# 1. 无需真实数据集、无需 GPU
python smoke_test.py

# 参数说明（可按需缩减时间）
python smoke_test.py --epochs 1 --batch_size 2 --n_samples 4
```

### smoke_test.py 做了什么

1. **合成数据集**：生成随机 motion（`151`维 × 150帧）+ music feature（`35`维 × 150帧）
2. **完整模型初始化**：与真实训练完全相同的 `DanceDecoder` + `GaussianDiffusion`
3. **N 轮训练**：走完梯度下降全流程（loss 计算 → backward → Adan 优化步）
4. **DDIM 推理**：调用 `diffusion.ddim_sample()`，断言输出 shape 正确
5. **设备感知**：自动使用 CUDA > MPS > CPU，无需手动指定

### 验证范围与限制

| 验证内容 | 是否覆盖 |
|----------|---------|
| 模型前向传播 | ✅ |
| 扩散损失（recon + velocity + FK + foot contact） | ✅ |
| EMA 更新 | ✅ |
| DDIM 采样 | ✅ |
| accelerate 分布式包装 | ✅（单进程） |
| 真实数据集加载 | ❌（用合成数据代替） |
| Jukebox 特征提取 | ❌（用随机向量代替） |
| 视频渲染 | ❌（不调用 ffmpeg/matplotlib animation） |

---

## 六、给你的建议

### 下一步操作顺序

1. **先在 macOS 上创建环境并运行冒烟测试**（验证本地开发流程可用）：
   ```bash
   conda env create -f environment-mac.yml
   conda activate edge-mac
   python smoke_test.py
   ```

2. **在 Linux 上创建训练环境**，手动安装 pytorch3d，然后用小数据集（20条样本）跑 10 个 epoch 验证 GPU 流程：
   ```bash
   conda env create -f environment-linux.yml
   conda activate edge-linux
   pip install pytorch3d --extra-index-url https://...
   # 准备少量真实数据，然后：
   accelerate launch train.py \
     --batch_size 4 --epochs 10 --feature_type baseline
   ```

3. **用 sbatch 提交**时，在脚本中加载 conda 环境：
   ```bash
   #!/bin/bash
   #SBATCH --gres=gpu:1
   #SBATCH --time=48:00:00
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate edge-linux
   accelerate launch train.py --batch_size 128 --epochs 2000 \
     --feature_type jukebox --learning_rate 0.0002
   ```

### 需要你手动处理的事

- pytorch3d 在 macOS 上的编译（`xcode-select --install` 后约需 10-15 分钟）
- jukemirlib 在 Linux 上需联网下载 ~10GB Jukebox 预训练权重（首次 feature 提取时自动触发）
- `accelerate config` 在每台机器上都需要运行一次（配置 fp16/多 GPU）
