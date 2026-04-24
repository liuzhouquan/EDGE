"""
Laban Movement Analysis (LMA) 特征提取。

将旧项目 dance-ai-research-project/src/utils/laban_analysis/feature_utils.py
的逻辑用 numpy 重写，直接接受 EDGE FK 输出的 3D 关节位置。

输入
----
joints : np.ndarray, shape [T, 24, 3]
    SMPL 24 关节的世界坐标，由 EDGE 的 SMPLSkeleton.forward() 计算得到。

输出
----
dict，4 个分量各自的特征矩阵：
  "body"   : [W_bc, n_bc]   Body Component
  "effort" : [W_ec, n_ec]   Effort Component
  "shape"  : [W_sc, n_sc]   Shape Component
  "space"  : [W_pc, n_pc]   Space Component

其中 W 是滑动窗口数（stride=1），n 是该分量的特征维度。

LMA 四分量对应关系（继承自旧代码的特征编号）
  Body:   f1-f8   — 肢体延伸、身体重心
  Effort: f11,f13-f19 — 速度/加速度/抖动
  Shape:  f20-f29 — 身体包围盒体积、手位区域
  Space:  f30,f31 — 骨盆轨迹长度、轨迹包围面积
"""

import numpy as np

# SMPL 24 关节索引（和旧代码 JOINT_MAP 完全一致）
JOINT_IDX = {
    "Pelvis": 0, "RHip": 1, "LHip": 2, "spine1": 3,
    "RKnee": 4, "LKnee": 5, "spine2": 6, "RAnkle": 7,
    "LAnkle": 8, "spine3": 9, "RFoot": 10, "LFoot": 11,
    "Neck": 12, "RCollar": 13, "LCollar": 14, "Head": 15,
    "RShoulder": 16, "LShoulder": 17, "RElbow": 18, "LElbow": 19,
    "RWrist": 20, "LWrist": 21, "RHand": 22, "LHand": 23,
}

# 滑动窗口参数（和旧代码一致：BC/SC/PC 用 35 帧，EC 用 10 帧）
BC_SC_PC_WINDOW = 35   # ~1.17s @ 30fps
EC_WINDOW       = 10   # ~0.33s @ 30fps


def _j(joints: np.ndarray, name: str) -> np.ndarray:
    """取指定关节的坐标序列，返回 [T, 3]。"""
    return joints[:, JOINT_IDX[name], :]


def _norm(v: np.ndarray) -> np.ndarray:
    """逐行求向量模长，v: [..., 3] → [...] 标量。"""
    return np.linalg.norm(v, axis=-1)


def _sliding_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """
    对 arr [T, ...] 做步长为 1 的滑动窗口切分，
    返回 [num_windows, window, ...]。
    """
    T = arr.shape[0]
    n = T - window + 1
    if n <= 0:
        return arr[np.newaxis]  # 不够一个窗口时把整段当一个窗口
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]  # [n, window]
    return arr[idx]


def _agg(windows: np.ndarray) -> np.ndarray:
    """
    对窗口内的帧维度做 [min, max, mean, std] 聚合，
    windows: [W, window, F] → [W, F*4]。
    """
    mn  = windows.min(axis=1)    # [W, F]
    mx  = windows.max(axis=1)
    mu  = windows.mean(axis=1)
    std = windows.std(axis=1)
    return np.concatenate([mn, mx, mu, std], axis=-1)  # [W, F*4]


# ---------------------------------------------------------------------------
# Body Component  (f1~f8)
# ---------------------------------------------------------------------------

def compute_body_component(joints: np.ndarray) -> np.ndarray:
    """
    对每帧计算 8 个体态特征，然后在 BC_SC_PC_WINDOW 帧窗口内做统计聚合。

    f1  — 腿部延伸：mean(|LFoot-LHip|, |RFoot-RHip|)
    f2  — 手臂延伸：mean(|LHand-LShoulder|, |RHand-RShoulder|)
    f3  — 双手张开宽度：|RHand-LHand|
    f4  — 手距头部：mean(|LHand-Head|, |RHand-Head|)
    f5  — 手距髋部：mean(|LHand-LHip|, |RHand-RHip|)
    f6  — 骨盆高度（y 分量）
    f7  — 腿部延伸 - 平均髋高（重心弯曲程度）
    f8  — 双脚间距：|LFoot-RFoot|

    返回 [W, 8*4] 的聚合特征矩阵。
    """
    T = joints.shape[0]

    f1 = (
        _norm(_j(joints, "LFoot") - _j(joints, "LHip")) +
        _norm(_j(joints, "RFoot") - _j(joints, "RHip"))
    ) / 2  # [T]

    f2 = (
        _norm(_j(joints, "LHand") - _j(joints, "LShoulder")) +
        _norm(_j(joints, "RHand") - _j(joints, "RShoulder"))
    ) / 2

    f3 = _norm(_j(joints, "RHand") - _j(joints, "LHand"))

    f4 = (
        _norm(_j(joints, "LHand") - _j(joints, "Head")) +
        _norm(_j(joints, "RHand") - _j(joints, "Head"))
    ) / 2

    f5 = (
        _norm(_j(joints, "LHand") - _j(joints, "LHip")) +
        _norm(_j(joints, "RHand") - _j(joints, "RHip"))
    ) / 2

    f6 = _j(joints, "Pelvis")[:, 1]  # y 分量

    hip_height = (
        _j(joints, "LHip")[:, 1] + _j(joints, "RHip")[:, 1]
    ) / 2
    f7 = f1 - hip_height

    f8 = _norm(_j(joints, "LFoot") - _j(joints, "RFoot"))

    # [T, 8]
    frame_feats = np.stack([f1, f2, f3, f4, f5, f6, f7, f8], axis=-1)

    # 滑动窗口 + 聚合 → [W, 8*4=32]
    wins = _sliding_windows(frame_feats, BC_SC_PC_WINDOW)  # [W, 35, 8]
    return _agg(wins)


# ---------------------------------------------------------------------------
# Effort Component  (f11,f13-f19)
# ---------------------------------------------------------------------------

def compute_effort_component(joints: np.ndarray) -> np.ndarray:
    """
    计算关键关节的速度、加速度和抖动，在 EC_WINDOW 帧窗口内聚合。

    f11 — 骨盆速度（帧间位移模长）
    f13 — 双手平均速度
    f14 — 双脚平均速度
    f15 — 骨盆加速度（f11 的差分）
    f17 — 双手平均加速度
    f18 — 双脚平均加速度
    f19 — 骨盆抖动（f15 的差分）

    返回 [W_ec, 7*4] 的聚合特征矩阵。
    """
    def speed(seq):
        """[T,3] → [T] 帧间速度（第 0 帧补 0）。"""
        v = np.zeros(seq.shape[0])
        v[1:] = _norm(np.diff(seq, axis=0))
        return v

    pelvis_v = speed(_j(joints, "Pelvis"))
    lhand_v  = speed(_j(joints, "LHand"))
    rhand_v  = speed(_j(joints, "RHand"))
    lfoot_v  = speed(_j(joints, "LFoot"))
    rfoot_v  = speed(_j(joints, "RFoot"))

    f11 = pelvis_v
    f13 = (lhand_v + rhand_v) / 2
    f14 = (lfoot_v + rfoot_v) / 2

    # 加速度（速度的差分）
    f15 = np.zeros_like(f11); f15[1:] = np.diff(f11)
    f17 = np.zeros_like(f13); f17[1:] = np.diff(f13)
    f18 = np.zeros_like(f14); f18[1:] = np.diff(f14)

    # 抖动（加速度的差分）
    f19 = np.zeros_like(f15); f19[1:] = np.diff(f15)

    frame_feats = np.stack([f11, f13, f14, f15, f17, f18, f19], axis=-1)  # [T, 7]

    wins = _sliding_windows(frame_feats, EC_WINDOW)  # [W, 10, 7]
    return _agg(wins)  # [W, 28]


# ---------------------------------------------------------------------------
# Shape Component  (f20-f29)
# ---------------------------------------------------------------------------

def compute_shape_component(joints: np.ndarray) -> np.ndarray:
    """
    计算身体各部位在 3D 包围盒中的体积及手部高度区域，
    在 BC_SC_PC_WINDOW 帧窗口内聚合。

    f20 — 四肢端点包围盒体积（Head/LHand/RHand/LFoot/RFoot）
    f21 — 全身包围盒体积
    f22 — 上半身包围盒体积
    f23 — 下半身包围盒体积
    f24 — 右侧体节 + 中轴包围盒体积
    f25 — 左侧体节 + 中轴包围盒体积
    f26 — 脊柱向量长度（Head-Pelvis 距离）
    f27 — 手部高于头部（0/1）
    f28 — 手部低于骨盆（0/1）
    f29 — 手部在头骨盆之间（0/1）

    返回 [W, 10*4] 的聚合特征矩阵。
    """
    def bbox_vol(pts_list: list) -> np.ndarray:
        """pts_list: list of [T,3] → [T] 包围盒体积。"""
        stacked = np.stack(pts_list, axis=1)  # [T, N, 3]
        span = stacked.max(axis=1) - stacked.min(axis=1)  # [T, 3]
        return span[:, 0] * span[:, 1] * span[:, 2]

    upper = ["Head", "Neck", "LShoulder", "RShoulder", "LCollar", "RCollar",
             "LElbow", "RElbow", "LWrist", "RWrist", "LHand", "RHand",
             "spine3", "spine2", "spine1"]
    lower = ["Pelvis", "LHip", "RHip", "LKnee", "RKnee",
             "LAnkle", "RAnkle", "LFoot", "RFoot"]
    right_mid = ["RHip", "RKnee", "RAnkle", "RFoot",
                 "RShoulder", "RCollar", "RElbow", "RWrist", "RHand",
                 "Head", "spine3", "spine2", "spine1", "Pelvis"]
    left_mid  = ["LHip", "LKnee", "LAnkle", "LFoot",
                 "LShoulder", "LCollar", "LElbow", "LWrist", "LHand",
                 "Head", "spine3", "spine2", "spine1", "Pelvis"]
    all_joints = list(JOINT_IDX.keys())

    f20 = bbox_vol([_j(joints, n) for n in ["Head", "LHand", "RHand", "LFoot", "RFoot"]])
    f21 = bbox_vol([_j(joints, n) for n in all_joints])
    f22 = bbox_vol([_j(joints, n) for n in upper])
    f23 = bbox_vol([_j(joints, n) for n in lower])
    f24 = bbox_vol([_j(joints, n) for n in right_mid])
    f25 = bbox_vol([_j(joints, n) for n in left_mid])
    f26 = _norm(_j(joints, "Head") - _j(joints, "Pelvis"))

    avg_hand_y = (_j(joints, "LHand")[:, 1] + _j(joints, "RHand")[:, 1]) / 2
    head_y     = _j(joints, "Head")[:, 1]
    pelvis_y   = _j(joints, "Pelvis")[:, 1]

    f27 = (avg_hand_y > head_y).astype(float)
    f28 = (avg_hand_y < pelvis_y).astype(float)
    f29 = ((avg_hand_y <= head_y) & (avg_hand_y >= pelvis_y)).astype(float)

    frame_feats = np.stack(
        [f20, f21, f22, f23, f24, f25, f26, f27, f28, f29], axis=-1
    )  # [T, 10]

    wins = _sliding_windows(frame_feats, BC_SC_PC_WINDOW)  # [W, 35, 10]
    return _agg(wins)  # [W, 40]


# ---------------------------------------------------------------------------
# Space Component  (f30, f31)
# ---------------------------------------------------------------------------

def compute_space_component(joints: np.ndarray) -> np.ndarray:
    """
    分析骨盆在水平面（XZ 平面）上的运动轨迹，
    在 BC_SC_PC_WINDOW 帧窗口内聚合。

    f30 — 窗口内骨盆 XZ 帧间位移的均值（行进速度）
    f31 — 骨盆 XZ 轨迹的凸包面积（使用 shoelace 公式近似）

    返回 [W, 2*4] 的聚合特征矩阵（对单窗口内多子窗口值做统计）。

    注意：由于 f30/f31 本身已经是窗口级别的统计量，
    这里改为只输出整体统计，形状 [1, 2*4] 后 tile 到 [W, 2*4]
    以保持和其他分量窗口数一致（实际按时序使用时应注意这一点）。
    """
    pelvis_xz = _j(joints, "Pelvis")[:, [0, 2]]  # [T, 2]，取 X 和 Z

    # 帧间位移
    disp = np.zeros(len(pelvis_xz))
    disp[1:] = _norm(np.diff(pelvis_xz, axis=0))

    def shoelace_area(pts: np.ndarray) -> float:
        """pts: [N, 2]，用 shoelace 公式计算多边形面积（近似凸包）。"""
        if len(pts) < 3:
            return 0.0
        x, z = pts[:, 0], pts[:, 1]
        return 0.5 * abs(
            np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))
        )

    # 逐窗口计算
    T = len(joints)
    n_wins = max(T - BC_SC_PC_WINDOW + 1, 1)
    f30_wins = np.zeros(n_wins)
    f31_wins = np.zeros(n_wins)
    for i in range(n_wins):
        sl = slice(i, i + BC_SC_PC_WINDOW)
        f30_wins[i] = disp[sl].mean()
        f31_wins[i] = shoelace_area(pelvis_xz[sl])

    # [W, 2] → agg [W, 2*4]
    # 这里不再做滑窗聚合，f30/f31 本身已是窗口统计，直接横向堆叠
    feat = np.stack([f30_wins, f31_wins], axis=-1)  # [W, 2]
    # 为和其他分量接口统一，返回 [W, 2]（调用方自行决定是否再聚合）
    return feat


# ---------------------------------------------------------------------------
# 主入口：提取全部 LMA 特征
# ---------------------------------------------------------------------------

def extract_lma_features(joints: np.ndarray) -> dict:
    """
    从 3D 关节位置序列提取完整 LMA 特征。

    参数
    ----
    joints : np.ndarray [T, 24, 3]
        SMPL 24 关节在世界坐标系下的位置（由 EDGE FK 计算得到）。
        坐标系：z-up（EDGE 经过旋转后的标准坐标系）。

    返回
    ----
    dict with keys:
        "body"   : np.ndarray [W_bc, 32]
        "effort" : np.ndarray [W_ec, 28]
        "shape"  : np.ndarray [W_sc, 40]
        "space"  : np.ndarray [W_pc,  2]
    """
    return {
        "body":   compute_body_component(joints),
        "effort": compute_effort_component(joints),
        "shape":  compute_shape_component(joints),
        "space":  compute_space_component(joints),
    }
