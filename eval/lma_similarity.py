"""
主舞 vs 伴舞 LMA 相似度分析脚本。

用法示例
--------
# 比较两个 EDGE 输出的 motion pkl：
python eval/lma_similarity.py \
    --lead  eval/motions/lead.pkl \
    --follower eval/motions/follower.pkl \
    --plot

# 批量比较一个目录下所有配对（文件名前缀相同的视为一对）：
python eval/lma_similarity.py \
    --lead_dir  eval/motions/lead/ \
    --follower_dir eval/motions/follower/ \
    --out eval/lma_results.csv

输入格式
--------
每个 pkl 文件需包含 SMPL 关节世界坐标，有两种支持格式：

格式A（EDGE render_sample 输出的 fk_out pkl）：
  {"smpl_poses": [T,72], "smpl_trans": [T,3], "full_pose": [T,24,3]}
  — 直接使用 full_pose 字段。

格式B（原始 AIST++ 预处理切片，含 pos 和 q）：
  {"pos": [T,3], "q": [T,72]}
  — 需要运行 EDGE 的 SMPLSkeleton FK 得到 full_pose。

输出
----
每对动作输出：
  - 4 个 LMA 分量（Body/Effort/Shape/Space）的 Pearson r 序列
  - 加权综合相似度得分 (0~1)
  - （可选）热图可视化
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

# 确保能 import EDGE 项目的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.lma_features import extract_lma_features


# 4 个分量的权重（来自旧项目论文，可按需调整）
COMPONENT_WEIGHTS = {
    "body":   0.30,
    "effort": 0.30,
    "shape":  0.20,
    "space":  0.20,
}


def load_motion_as_joints(pkl_path: str) -> np.ndarray:
    """
    从 EDGE 输出的 pkl 加载 3D 关节位置。

    返回 np.ndarray [T, 24, 3]，世界坐标。
    """
    data = pickle.load(open(pkl_path, "rb"))

    # 格式A：EDGE fk_out pkl，直接有 full_pose
    if "full_pose" in data:
        joints = np.array(data["full_pose"])  # [T, 24, 3]
        if joints.ndim == 4:
            joints = joints[0]  # 去掉 batch 维
        return joints

    # 格式B：含 pos 和 q，需要跑 FK
    if "pos" in data and "q" in data:
        return _fk_from_smpl_params(data["pos"], data["q"])

    # 格式C：AIST++ 原始 pkl（smpl_poses/smpl_trans/smpl_scaling）
    if "smpl_poses" in data:
        pos = np.array(data["smpl_trans"])   # [T, 3]
        q   = np.array(data["smpl_poses"])   # [T, 72]
        if "smpl_scaling" in data:
            pos /= data["smpl_scaling"][0]
        return _fk_from_smpl_params(pos, q)

    raise ValueError(
        f"无法识别 pkl 格式，支持的 key 组合：\n"
        f"  A: full_pose\n  B: pos+q\n  C: smpl_poses+smpl_trans\n"
        f"  实际 keys: {list(data.keys())}"
    )


def _fk_from_smpl_params(pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    用 EDGE 的 SMPLSkeleton 做正向运动学。
    pos: [T,3] root position, q: [T,72] axis-angle rotations.
    返回 [T, 24, 3]。
    """
    import torch
    from vis import SMPLSkeleton
    from dataset.quaternion import ax_to_6v
    from pytorch3d.transforms import (
        axis_angle_to_quaternion, quaternion_multiply, quaternion_to_axis_angle
    )

    smpl = SMPLSkeleton()
    T = pos.shape[0]
    pos_t = torch.Tensor(pos)
    q_t   = torch.Tensor(q).reshape(T, -1, 3)  # [T, 24, 3]

    # AIST++ 是 y-up，EDGE 训练时旋转到 z-up
    root_q      = q_t[:, :1, :]
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation    = torch.Tensor([0.7071068, 0.7071068, 0, 0])
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    q_t[:, :1, :] = quaternion_to_axis_angle(root_q_quat)

    from pytorch3d.transforms import RotateAxisAngle
    pos_t = RotateAxisAngle(90, axis="X", degrees=True).transform_points(pos_t)

    # FK
    with torch.no_grad():
        joints = smpl.forward(q_t.unsqueeze(0), pos_t.unsqueeze(0))  # [1,T,24,3]
    return joints.squeeze(0).numpy()  # [T, 24, 3]


# ---------------------------------------------------------------------------
# 相似度计算
# ---------------------------------------------------------------------------

def pearson_per_window(feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
    """
    对两个特征矩阵 [W, F] 按窗口计算 Pearson r。
    窗口数不同时，取较小值对齐（时序截断）。
    返回 [W] 相关系数序列（NaN 窗口置 0）。
    """
    W = min(feat1.shape[0], feat2.shape[0])
    rs = np.zeros(W)
    for i in range(W):
        a, b = feat1[i], feat2[i]
        # 跳过全零或方差为零的窗口
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs[i] = 0.0
        else:
            r, _ = pearsonr(a, b)
            rs[i] = 0.0 if np.isnan(r) else r
    return rs


def compute_similarity(lead_joints: np.ndarray,
                       follower_joints: np.ndarray) -> dict:
    """
    计算主舞和伴舞动作之间的 LMA 相似度。

    返回 dict：
      "body_r", "effort_r", "shape_r", "space_r"  — Pearson r 序列 [W]
      "body_score", "effort_score", "shape_score", "space_score" — 各分量均分
      "total_score"  — 加权综合得分 (0~1，越高越相似)
    """
    feats1 = extract_lma_features(lead_joints)
    feats2 = extract_lma_features(follower_joints)

    results = {}
    total = 0.0

    for comp in ["body", "effort", "shape", "space"]:
        r_seq = pearson_per_window(feats1[comp], feats2[comp])
        # 将 Pearson r 映射到 [0,1]：(r+1)/2
        score = float(np.clip((r_seq.mean() + 1) / 2, 0, 1))
        results[f"{comp}_r"]     = r_seq
        results[f"{comp}_score"] = score
        total += COMPONENT_WEIGHTS[comp] * score

    results["total_score"] = total
    return results


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_similarity(results: dict, title: str = "LMA Similarity"):
    """绘制 4 个分量的 Pearson r 热图（和旧代码风格一致）。"""
    import matplotlib.pyplot as plt

    comps  = ["body", "effort", "shape", "space"]
    labels = [
        f"Body   (score={results['body_score']:.3f})",
        f"Effort (score={results['effort_score']:.3f})",
        f"Shape  (score={results['shape_score']:.3f})",
        f"Space  (score={results['space_score']:.3f})",
    ]

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 7), sharex=False)
    fig.suptitle(f"{title}\nTotal Score: {results['total_score']:.3f}", fontsize=13)

    for ax, comp, label in zip(axs, comps, labels):
        rs = results[f"{comp}_r"]
        cax = ax.imshow([rs], aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Window index")

    plt.colorbar(cax, ax=axs[-1], orientation="horizontal",
                 label="Pearson r (-1=opposite, +1=identical)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="计算主舞和伴舞动作之间的 LMA 相似度"
    )
    # 单对模式
    parser.add_argument("--lead",     type=str, default="",
                        help="主舞 motion pkl 路径")
    parser.add_argument("--follower", type=str, default="",
                        help="伴舞 motion pkl 路径")
    # 批量模式
    parser.add_argument("--lead_dir",     type=str, default="",
                        help="主舞 pkl 目录（与 --follower_dir 配合）")
    parser.add_argument("--follower_dir", type=str, default="",
                        help="伴舞 pkl 目录（按文件名排序配对）")
    # 输出
    parser.add_argument("--out",  type=str, default="",
                        help="CSV 输出路径（批量模式）")
    parser.add_argument("--plot", action="store_true",
                        help="显示热图（单对模式）")
    parser.add_argument("--save_plot", type=str, default="",
                        help="保存热图到此路径（不弹窗）")
    return parser.parse_args()


def single_pair(lead_path: str, follower_path: str,
                plot: bool = False, save_plot: str = "") -> dict:
    print(f"加载主舞:  {lead_path}")
    lead_joints = load_motion_as_joints(lead_path)
    print(f"加载伴舞:  {follower_path}")
    follower_joints = load_motion_as_joints(follower_path)

    print(f"主舞帧数: {lead_joints.shape[0]}, 伴舞帧数: {follower_joints.shape[0]}")

    results = compute_similarity(lead_joints, follower_joints)

    print("\n=== LMA 相似度报告 ===")
    for comp in ["body", "effort", "shape", "space"]:
        print(f"  {comp:6s} score = {results[f'{comp}_score']:.4f}")
    print(f"  综合得分  = {results['total_score']:.4f}")

    if plot or save_plot:
        import matplotlib.pyplot as plt
        name = os.path.basename(lead_path) + " vs " + os.path.basename(follower_path)
        fig = plot_similarity(results, title=name)
        if save_plot:
            fig.savefig(save_plot, dpi=150, bbox_inches="tight")
            print(f"热图已保存到 {save_plot}")
        if plot:
            plt.show()

    return results


def batch_pairs(lead_dir: str, follower_dir: str, out_csv: str = ""):
    import csv

    lead_files     = sorted(Path(lead_dir).glob("*.pkl"))
    follower_files = sorted(Path(follower_dir).glob("*.pkl"))

    if len(lead_files) != len(follower_files):
        print(f"警告：主舞 {len(lead_files)} 个，伴舞 {len(follower_files)} 个，按较小值截断")

    n = min(len(lead_files), len(follower_files))
    rows = []

    for i, (lp, fp) in enumerate(zip(lead_files[:n], follower_files[:n])):
        print(f"[{i+1}/{n}] {lp.name} vs {fp.name}")
        try:
            res = single_pair(str(lp), str(fp))
            rows.append({
                "lead":          lp.name,
                "follower":      fp.name,
                "body_score":    res["body_score"],
                "effort_score":  res["effort_score"],
                "shape_score":   res["shape_score"],
                "space_score":   res["space_score"],
                "total_score":   res["total_score"],
            })
        except Exception as e:
            print(f"  ❌ 跳过（{e}）")

    if rows:
        if out_csv:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"\n结果已保存到 {out_csv}")

        scores = [r["total_score"] for r in rows]
        print(f"\n批量汇总：共 {len(rows)} 对，综合得分均值={np.mean(scores):.4f}, "
              f"std={np.std(scores):.4f}")

    return rows


def main():
    args = parse_args()

    if args.lead and args.follower:
        single_pair(args.lead, args.follower,
                    plot=args.plot, save_plot=args.save_plot)
    elif args.lead_dir and args.follower_dir:
        batch_pairs(args.lead_dir, args.follower_dir, out_csv=args.out)
    else:
        print("请提供 --lead/--follower（单对）或 --lead_dir/--follower_dir（批量）")
        sys.exit(1)


if __name__ == "__main__":
    main()
