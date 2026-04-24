import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from torch.utils.data import Dataset

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton


class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(
            f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}"
        )

        # process data, convert to 6dof etc
        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {
            "pose": pose_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return self.data["pose"][idx], feature, filename_, self.data["wavs"][idx]

    def load_aistpp(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- motion_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_features
        #   |    |- jukebox_features
        #   |    |- motions
        #   |    |- wavs

        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the motions and features together
        all_pos = []
        all_q = []
        all_names = []
        all_wavs = []
        assert len(motions) == len(features)
        for motion, feature, wav in zip(motions, features, wavs):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert m_name == f_name == w_name, str((motion, feature, wav))
            # load motion
            data = pickle.load(open(motion, "rb"))
            pos = data["pos"]
            q = data["q"]
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)

        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(all_pos.shape)
        all_pos = all_pos[:, :: self.data_stride, :]
        all_q = all_q[:, :: self.data_stride, :]
        data = {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input


def _parse_filename_parts(filename):
    """从特征文件路径中解析 music_id 和 dancer_id。

    文件名格式（去掉扩展名）：gBR_sBM_cAll_d04_mBR1_slice0
    返回 (music_id, dancer_id)，例如 ("mBR1", "d04")。
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_")
    # parts: ["gBR", "sBM", "cAll", "d04", "mBR1", "slice0"]
    dancer_id = parts[3]  # e.g. "d04"
    music_id = parts[4]   # e.g. "mBR1"
    return music_id, dancer_id


def preprocess_motion_to_tensor(pos_np, q_np, normalizer, data_stride=2):
    """把单条原始运动数据预处理成归一化的 151 维张量。

    用于推理时把主舞动作转成和训练数据相同格式的条件向量。

    Args:
        pos_np:      根节点位置 [T_raw, 3]，numpy，原始帧率（60fps）
        q_np:        关节旋转 [T_raw, 72]，axis-angle，原始帧率
        normalizer:  从训练集 checkpoint 里读出的 Normalizer 对象
        data_stride: 降采样倍数，默认 2（60fps→30fps）

    Returns:
        归一化后的 float tensor [T, 151]，
        格式：[foot_contacts(4), root_pos(3), rotations_6d(144)]
        与 AISTPPDataset 输出格式完全一致。
    """
    # 降采样
    pos_np = pos_np[::data_stride]
    q_np = q_np[::data_stride]
    T = pos_np.shape[0]

    pos = torch.Tensor(pos_np)                        # [T, 3]
    local_q = torch.Tensor(q_np).reshape(T, -1, 3)   # [T, 24, 3]

    # AIST++ 是 y-up 坐标系，旋转到 z-up（和 AISTPPDataset.process_dataset 完全一致）
    root_q = local_q[:, :1, :]
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor([0.7071068, 0.7071068, 0, 0])  # 绕 X 轴旋转 90°
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    local_q[:, :1, :] = quaternion_to_axis_angle(root_q_quat)

    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    pos = pos_rotation.transform_points(pos)

    # 正向运动学，计算脚部接触标签
    smpl = SMPLSkeleton()
    positions = smpl.forward(local_q.unsqueeze(0), pos.unsqueeze(0))  # [1, T, 24, 3]
    feet = positions[0, :, (7, 8, 10, 11)]                            # [T, 4, 3]
    feetv = torch.zeros(T, 4)
    feetv[:-1] = (feet[1:] - feet[:-1]).norm(dim=-1)
    contacts = (feetv < 0.01).to(local_q.dtype)                       # [T, 4]

    # 旋转转换到 6D 表示
    local_q = ax_to_6v(local_q.unsqueeze(0)).squeeze(0)  # [T, 24, 6]
    local_q_flat = local_q.reshape(T, -1)                 # [T, 144]

    # 拼接成 151 维向量：[contacts(4), root_pos(3), rotations(144)]
    motion_vec = torch.cat([contacts, pos, local_q_flat], dim=-1).float()  # [T, 151]

    # 用训练集的归一化器做归一化
    motion_vec = normalizer.normalize(motion_vec.unsqueeze(0)).squeeze(0)  # [T, 151]

    return motion_vec


class DuetDataset(AISTPPDataset):
    """在 AISTPPDataset 基础上支持"主舞+伴舞"配对。

    核心思路
    --------
    AIST++ 里同一个 music_id（如 mBR1）的序列使用相同的 BGM，但由不同
    dancer（如 d04、d05）独立录制。我们把同 music_id、不同 dancer 的序列
    视为一组，每次 __getitem__ 随机从组内选一条作为主舞，当前序列作为伴舞。

    返回格式
    --------
    (follower_pose, cond, filename, wavname)
      follower_pose : [150, 151]   — 伴舞动作（训练目标）
      cond          : [150, 4951]  — cat([主舞动作(151), 音乐特征(4800)], dim=-1)
      filename      : 伴舞对应的特征文件路径（字符串）
      wavname       : 伴舞对应的音频文件路径（字符串）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_music_groups()

    def _build_music_groups(self):
        """按 (music_id, dancer_id) 把序列分组，为每条序列记录可用的主舞候选。

        只选"同 music_id、不同 dancer_id"的序列作为主舞，
        保证主舞和伴舞来自不同的人。
        """
        from collections import defaultdict

        # 第一遍：按 music_id 聚合，记录 (数据索引, dancer_id)
        groups = defaultdict(list)  # music_id -> [(idx, dancer_id), ...]
        for idx, filename in enumerate(self.data["filenames"]):
            music_id, dancer_id = _parse_filename_parts(filename)
            groups[music_id].append((idx, dancer_id))

        # 第二遍：为每个序列找出"不同 dancer"的候选主舞索引列表
        # idx -> [lead_idx, ...]（同 music_id、不同 dancer_id 的所有序列）
        self.idx_to_leads = {}
        for music_id, entries in groups.items():
            for (idx, dancer_id) in entries:
                leads = [i for (i, d) in entries if d != dancer_id]
                if leads:
                    self.idx_to_leads[idx] = leads

        # 只保留有合法主舞候选的序列（理论上 AIST++ 每组 ≥2 个 dancer，全部有效）
        self.valid_indices = sorted(self.idx_to_leads.keys())
        self.length = len(self.valid_indices)

        print(
            f"DuetDataset: {self.length} valid follower sequences "
            f"across {len(set(groups.keys()))} music groups"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 把压缩后的 idx 映射回原始数据索引
        follower_idx = self.valid_indices[idx]

        # 从同 music_id、不同 dancer 里随机选一条作为主舞
        lead_idx = random.choice(self.idx_to_leads[follower_idx])

        # 伴舞动作（已归一化）
        follower_pose = self.data["pose"][follower_idx]  # [150, 151]
        # 主舞动作（同一归一化器，格式完全相同）
        lead_pose = self.data["pose"][lead_idx]          # [150, 151]

        # 伴舞对应的音乐特征（同 music_id 下音乐特征相同，用伴舞侧文件即可）
        music_feat = torch.from_numpy(
            np.load(self.data["filenames"][follower_idx])
        )  # [150, feature_dim]，jukebox=4800 或 baseline=35

        # 拼接主舞动作 + 音乐特征作为条件向量
        # 模型 cond_feature_dim 需设为 feature_dim + 151
        cond = torch.cat([lead_pose, music_feat], dim=-1)  # [150, 4951 or 186]

        return (
            follower_pose,
            cond,
            self.data["filenames"][follower_idx],
            self.data["wavs"][follower_idx],
        )


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "baseline",
        data_name: str = "aist",
    ):
        self.data_path = data_path
        self.data_fps = 30
        self.feature_type = feature_type
        self.test_list = set(
            [
                "mLH4",
                "mKR2",
                "mBR0",
                "mLO2",
                "mJB5",
                "mWA0",
                "mJS3",
                "mMH3",
                "mHO5",
                "mPO1",
            ]
        )
        self.train = train

        # if not aist, then set train to true to ignore test split logic
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True

        self.data = self.load_music()  # Call this last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            max_start = len(seq) - batch_size
            start = random.randint(0, max_start)
            seq_slice = seq[start : start + batch_size]

        # now we have a batch of filenames
        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        # get the features
        features = np.array([np.load(x) for x in filenames])

        return torch.Tensor(features), seq_slice

    def load_music(self):
        # open data path
        split_data_path = os.path.join(self.data_path)
        music_path = os.path.join(
            split_data_path,
            f"{self.data_name}_baseline_feats"
            if self.feature_type == "baseline"
            else f"{self.data_name}_juke_feats/juke_66",
        )
        self.music_path = music_path
        # get the music filenames strided, with each subsequent item 5 slices (2.5 seconds) apart
        all_names = []

        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
            aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
            ka, kb = key_func(a), key_func(b)
            if aa < bb:
                return -1
            if aa > bb:
                return 1
            if ka < kb:
                return -1
            if ka > kb:
                return 1
            return 0

        for features in glob.glob(os.path.join(music_path, "*.npy")):
            fname = os.path.splitext(os.path.basename(features))[0]
            all_names.append(fname)
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))
        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or (
                (not self.train) and k not in self.test_list
            ):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict
