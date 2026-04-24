#!/usr/bin/env bash
# prepare_data.sh — 在新机器上一键准备好训练数据
#
# 用法（两种场景）：
#
# 场景A：已有 data/edge_aistpp/（本机或从其他地方 rsync 过来）
#   bash prepare_data.sh
#
# 场景B：有 AIST++ 原始数据文件夹（含 motions/ 和 wavs/）
#   bash prepare_data.sh --source /path/to/your_aistpp_folder
#
# 脚本会：
#   1. 把原始数据放到 data/edge_aistpp/（如果还没有）
#   2. 运行 create_dataset.py 切片 + 提取 baseline 音频特征
#      （jukebox 特征需要 GPU 且耗时很长，按需手动开启）
#
# 注意：data/ 目录在 .gitignore 里，不进 GitHub，需要单独传输。
# 推荐传输方式：
#   rsync -avP --progress data/edge_aistpp/ user@server:~/EDGE/data/edge_aistpp/

set -e

EDGE_DIR="$(cd "$(dirname "$0")" && pwd)"   # EDGE 项目根目录
DATA_DIR="$EDGE_DIR/data"
EDGE_AISTPP="$DATA_DIR/edge_aistpp"
SOURCE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=== EDGE 数据准备脚本 ==="
echo "项目目录: $EDGE_DIR"

# ── Step 1：建立 edge_aistpp 目录 ─────────────────────────────────────────
if [ -n "$SOURCE" ]; then
    echo ""
    echo "[1/3] 从 $SOURCE 建立 data/edge_aistpp/"
    mkdir -p "$EDGE_AISTPP/motions" "$EDGE_AISTPP/wavs"

    # 支持两种原始目录布局：
    #   布局A: source/motions/*.pkl + source/wavs/*.wav
    #   布局B: source/*.pkl + source/*.wav（扁平结构）
    if [ -d "$SOURCE/motions" ]; then
        echo "  检测到布局A（motions/ + wavs/ 子目录）"
        # 用软链接避免复制 300MB
        rm -rf "$EDGE_AISTPP/motions" "$EDGE_AISTPP/wavs"
        ln -sf "$SOURCE/motions" "$EDGE_AISTPP/motions"
        [ -d "$SOURCE/wavs" ] && ln -sf "$SOURCE/wavs" "$EDGE_AISTPP/wavs"
    else
        echo "  检测到布局B（扁平结构）"
        ln -sf "$SOURCE" "$EDGE_AISTPP/motions"
    fi
else
    echo ""
    echo "[1/3] 检查 data/edge_aistpp/ 是否已就绪..."
    MOTION_COUNT=$(ls "$EDGE_AISTPP/motions/" 2>/dev/null | wc -l || echo 0)
    WAV_COUNT=$(ls "$EDGE_AISTPP/wavs/" 2>/dev/null | wc -l || echo 0)
    echo "  motions: $MOTION_COUNT 个文件，wavs: $WAV_COUNT 个文件"

    if [ "$MOTION_COUNT" -eq 0 ]; then
        echo ""
        echo "❌ data/edge_aistpp/motions/ 为空。"
        echo "   请用以下任一方式准备数据："
        echo ""
        echo "   方式1（推荐）：从另一台机器 rsync："
        echo "     rsync -avP user@source:~/EDGE/data/edge_aistpp/ data/edge_aistpp/"
        echo ""
        echo "   方式2：下载 EDGE 官方打包数据（需要梯子）："
        echo "     cd data && bash download_dataset.sh && unzip edge_aistpp.zip && cd .."
        echo ""
        echo "   方式3：提供原始 AIST++ 数据路径："
        echo "     bash prepare_data.sh --source /path/to/aist_plusplus_with_wavs"
        echo ""
        echo "   ⚠️  注意：aist_plusplus_final/ 只有 motions 没有 wavs，无法单独使用。"
        exit 1
    fi

    if [ "$WAV_COUNT" -eq 0 ]; then
        echo "❌ data/edge_aistpp/wavs/ 为空，缺少音频文件，无法继续。"
        exit 1
    fi
fi

echo "  ✅ edge_aistpp 数据就绪"

# ── Step 2：切片（生成 train/ 和 test/ 的 motions_sliced + wavs_sliced）──
echo ""
echo "[2/3] 切片 motions 和 wavs（生成 data/train/ 和 data/test/）..."

# create_dataset.py 要在 data/ 目录下运行（它用相对路径读 splits/）
cd "$DATA_DIR"

TRAIN_SLICED="$DATA_DIR/train/motions_sliced"
if [ -d "$TRAIN_SLICED" ] && [ "$(ls "$TRAIN_SLICED" | wc -l)" -gt 0 ]; then
    echo "  检测到已有切片数据，跳过切片步骤（如需重切片请删除 data/train/ 和 data/test/）"
else
    python create_dataset.py \
        --dataset_folder edge_aistpp \
        --extract-baseline
    # 如果需要 jukebox 特征（GPU 必须），取消下面这行注释：
    # python create_dataset.py --dataset_folder edge_aistpp --extract-jukebox
fi

cd "$EDGE_DIR"

# ── Step 3：验证 ────────────────────────────────────────────────────────────
echo ""
echo "[3/3] 验证数据完整性..."
MOTION_SLICED=$(ls "$DATA_DIR/train/motions_sliced/" 2>/dev/null | wc -l || echo 0)
WAV_SLICED=$(ls "$DATA_DIR/train/wavs_sliced/" 2>/dev/null | wc -l || echo 0)
BASELINE=$(ls "$DATA_DIR/train/baseline_feats/" 2>/dev/null | wc -l || echo 0)
echo "  train/motions_sliced:  $MOTION_SLICED 个文件"
echo "  train/wavs_sliced:     $WAV_SLICED 个文件"
echo "  train/baseline_feats:  $BASELINE 个文件"

if [ "$MOTION_SLICED" -gt 0 ] && [ "$BASELINE" -gt 0 ]; then
    echo ""
    echo "✅ 数据准备完成！"
    echo ""
    echo "下一步——用 baseline 特征跑一次小规模验证："
    echo "  accelerate launch train.py --batch_size 4 --epochs 5 --feature_type baseline"
    echo ""
    echo "正式训练（jukebox 特征，需先确认 jukebox_feats/ 已生成）："
    echo "  sbatch run_training.sh"
else
    echo ""
    echo "⚠️  数据不完整，请检查上面的错误输出。"
    exit 1
fi
