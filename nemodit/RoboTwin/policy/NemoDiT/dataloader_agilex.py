"""
AgileX (Songling) robot data loader for NemoDiT.

与 ``dataloader.py`` (RoboTwin) 的主要差异
----------------------------------------
AgileX/Mobile-Aloha 使用 cobot_magic 采集脚本，HDF5 结构为扁平的 14-D
qpos/action + 3 个相机视角：

.. code-block:: text

    episode_X.hdf5
    ├── attrs
    │   ├── sim        (bool)      always False for real data
    │   └── compress   (bool)      True -> JPEG-encoded images
    ├── /observations
    │   ├── qpos       (T, 14)     slave arm 关节状态 (left 7 + right 7)
    │   ├── qvel       (T, 14)
    │   ├── effort     (T, 14)
    │   └── images/{cam_high, cam_left_wrist, cam_right_wrist}
    │                  (T, 480, 640, 3) uint8
    ├── /action        (T, 14)     master arm 目标命令 (主臂示教)
    └── /base_action   (T, 2)      [linear.x, angular.z]

每只手臂的 7-D 布局为 ``[joint0..joint5, gripper]``。因此扁平 14-D 向量已经
等价于 NemoDiT 的 ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``
布局，可以直接作为 14-D action 输入使用，无需重排序。

与 RoboTwin 版 dataloader 的另一个差异：state 来源。RoboTwin 里 state 取
``action[0]``（仿真下 master/slave 一致），但 AgileX 采集中 master（/action）与
slave（qpos）并不恒等，所以使用当前时刻的 ``qpos[t]`` 作为 state 更符合部署时
真实机器人的观测。
"""

import os
import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


AGILEX_CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
AGILEX_STATE_DIM = 14  # left(7) + right(7), each = 6 joints + 1 gripper


def _decode_image(raw: np.ndarray, compressed: bool) -> np.ndarray:
    """Decode a single-frame RGB image from HDF5.

    AgileX ``collect_data.py`` 默认写入未压缩 ``uint8`` (H, W, 3) 图像，但支持
    通过 ``compress`` 属性启用 JPEG 压缩。这里兼容两种情况。
    """
    if compressed:
        # `raw` is 1-D bytes array (encoded JPEG/PNG).
        img = cv2.imdecode(np.asarray(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        # OpenCV decodes as BGR; collect_data stores BGR via cv_bridge 'passthrough',
        # so we keep BGR->RGB conversion outside to match training/inference flow.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return np.asarray(raw)


def _list_episode_files(
    data_path: str,
    episode_ids: Optional[List[int]] = None,
) -> List[str]:
    """Return HDF5 episode file paths, optionally filtered by numeric id."""
    pattern = os.path.join(data_path, "episode_*.hdf5")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise ValueError(f"No episode files found in {data_path}")
    if episode_ids is None:
        return all_files
    wanted = set(int(i) for i in episode_ids)
    out: List[str] = []
    for path in all_files:
        stem = Path(path).stem  # "episode_N"
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        if idx in wanted:
            out.append(path)
    return out


def compute_agilex_norm_stats(
    data_path: str,
    num_episodes: Optional[int] = None,
    use_robot_base: bool = False,
    episode_ids: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-dimension mean/std statistics for qpos + action.

    Matches the normalization approach used in ``agx_robot/aloha-devel/act/utils.py``:
    std 下限 clip 到 1e-2 以避免除零。

    Args:
        data_path: Directory containing ``episode_*.hdf5`` files.
        num_episodes: If given (and ``episode_ids`` is None), the first ``N`` episodes
            sorted by filename are used.
        use_robot_base: Concatenate ``/base_action`` (2-D) onto qpos/action for
            mobile-base tasks. Default ``False``.
        episode_ids: Explicit list of numeric episode ids to include. Takes
            precedence over ``num_episodes`` when both are set.

    Returns:
        Dict with keys ``qpos_mean``, ``qpos_std``, ``action_mean``,
        ``action_std``.  Each array has shape ``(state_dim,)``.
    """
    if episode_ids is not None:
        files = _list_episode_files(data_path, episode_ids)
    else:
        files = _list_episode_files(data_path, None)
        if num_episodes is not None:
            files = files[:num_episodes]
    if not files:
        raise ValueError(f"No episode files found in {data_path}")

    all_qpos: List[np.ndarray] = []
    all_action: List[np.ndarray] = []
    for f_path in files:
        with h5py.File(f_path, "r") as root:
            qpos = root["/observations/qpos"][()].astype(np.float32)
            action = root["/action"][()].astype(np.float32)
            if use_robot_base:
                base = root["/base_action"][()].astype(np.float32)
                qpos = np.concatenate([qpos, base], axis=1)
                action = np.concatenate([action, base], axis=1)
        all_qpos.append(qpos)
        all_action.append(action)

    qpos_cat = np.concatenate(all_qpos, axis=0)
    action_cat = np.concatenate(all_action, axis=0)

    stats = {
        "qpos_mean": qpos_cat.mean(axis=0).astype(np.float32),
        "qpos_std": np.clip(qpos_cat.std(axis=0), 1e-2, None).astype(np.float32),
        "action_mean": action_cat.mean(axis=0).astype(np.float32),
        "action_std": np.clip(action_cat.std(axis=0), 1e-2, None).astype(np.float32),
    }
    return stats


class AgileXDataset(Dataset):
    """AgileX/Songling robot dataset for NemoDiT.

    每个样本包含:
        - ``images`` : (n_obs_steps, num_cameras, 3, H, W) - 多帧观测
        - ``state``  : (action_dim,) - 当前时刻 qpos，已归一化
        - ``actions``: (future_action_window - 1, action_dim) - 归一化后的未来动作
                        (与 RoboTwin 版本对齐：state 占据第 0 帧，之后是要预测的动作)
        - ``episode_idx`` / ``timestep``

    Args:
        data_path: Directory containing ``episode_*.hdf5`` files.
        norm_stats: Pre-computed normalization stats (see ``compute_agilex_norm_stats``).
            Must be reused between train / eval / deployment.
        future_action_window: Total action window (state + predictions). The
            first frame will be the state, the remaining ``future_action_window - 1``
            frames are the prediction target.
        past_action_window: Kept for API parity with RoboTwin dataloader.
            Must be ``0`` (the DiT implementation ignores history).
        transform: Optional per-frame torchvision transform (defaults to simple
            ``uint8 (H,W,C) -> float (C,H,W) / 255``).
        camera_names: List of camera names; defaults to all 3 AgileX cameras.
        num_cameras: Number of cameras to actually read (must be ``<= len(camera_names)``).
        n_obs_steps: Number of historical frames fed to the vision encoder.
        use_robot_base: If True, concatenates ``/base_action`` (2-D) to qpos/action,
            yielding 16-D state/action. Stats must be computed with the same flag.
        arm_delay_time: Forward shift for the action target in frames, matching
            ``agx_robot/aloha-devel/act/utils.py``. Default 0 (no shift).
    """

    def __init__(
        self,
        data_path: str,
        norm_stats: Dict[str, np.ndarray],
        future_action_window: int = 13,
        past_action_window: int = 0,
        transform: Optional[Callable] = None,
        camera_names: Optional[List[str]] = None,
        num_cameras: int = 3,
        n_obs_steps: int = 2,
        use_robot_base: bool = False,
        arm_delay_time: int = 0,
        episode_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        assert past_action_window == 0, "past_action_window must be 0 (DiT ignores history)."
        assert n_obs_steps >= 1, f"n_obs_steps must be >= 1, got {n_obs_steps}"

        self.data_path = data_path
        self.future_action_window = future_action_window
        self.transform = transform
        self.n_obs_steps = n_obs_steps
        self.use_robot_base = use_robot_base
        self.arm_delay_time = arm_delay_time
        self.norm_stats = {k: np.asarray(v, dtype=np.float32) for k, v in norm_stats.items()}

        if camera_names is None:
            camera_names = AGILEX_CAMERA_NAMES
        assert num_cameras <= len(camera_names), (
            f"num_cameras={num_cameras} > len(camera_names)={len(camera_names)}"
        )
        self.camera_names = list(camera_names[:num_cameras])

        self.episode_files = self._load_episode_files(episode_ids)
        print(f"[AgileXDataset] Found {len(self.episode_files)} episode files in {data_path}")
        self.indices = self._build_indices()
        print(f"[AgileXDataset] Total samples: {len(self.indices)}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_episode_files(self, episode_ids: Optional[List[int]]) -> List[str]:
        return _list_episode_files(self.data_path, episode_ids)

    def _build_indices(self) -> List[Tuple[int, int]]:
        """Every valid timestep is a sample; edges are padded in __getitem__."""
        indices: List[Tuple[int, int]] = []
        for ep_idx, ep_file in enumerate(self.episode_files):
            with h5py.File(ep_file, "r") as f:
                ep_len = f["/observations/qpos"].shape[0]
            for t in range(ep_len):
                indices.append((ep_idx, t))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------ #
    # Sample construction
    # ------------------------------------------------------------------ #

    def _load_actions(self, f: h5py.File, start_idx: int) -> np.ndarray:
        """Return ``(future_action_window, action_dim)`` action window.

        Index 0 always corresponds to ``start_idx`` (= the current observation
        timestep). The last observation frame in the returned window is used as
        the model's ``state`` conditioning; the rest is the prediction target.
        """
        actions = f["/action"][()].astype(np.float32)  # (T, 14)
        if self.use_robot_base:
            base = f["/base_action"][()].astype(np.float32)
            actions = np.concatenate([actions, base], axis=1)  # (T, 16)

        T_total = actions.shape[0]
        # Delay shift mirrors the agx_robot ACT dataloader: when the master arm
        # lags the slave by arm_delay_time frames the label is taken earlier.
        idx = max(0, start_idx - self.arm_delay_time)
        window = actions[idx : idx + self.future_action_window]

        if window.shape[0] < self.future_action_window:
            pad = np.repeat(window[-1:], self.future_action_window - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        return window  # (future_action_window, action_dim)

    def _load_qpos(self, f: h5py.File, timestep: int) -> np.ndarray:
        qpos = f["/observations/qpos"][timestep].astype(np.float32)  # (14,)
        if self.use_robot_base:
            base = f["/base_action"][timestep].astype(np.float32)  # (2,)
            qpos = np.concatenate([qpos, base], axis=0)
        return qpos

    def _load_frame_images(
        self, f: h5py.File, timestep: int, compressed: bool
    ) -> List[np.ndarray]:
        frame: List[np.ndarray] = []
        for cam in self.camera_names:
            raw = f[f"/observations/images/{cam}"][timestep]
            img = _decode_image(raw, compressed)
            frame.append(img)
        return frame

    def _load_obs_images(
        self, f: h5py.File, anchor_t: int, compressed: bool
    ) -> List[List[np.ndarray]]:
        """Load ``n_obs_steps`` frames whose last frame is ``anchor_t``."""
        start = anchor_t - self.n_obs_steps + 1
        frames: List[List[np.ndarray]] = []
        for t in range(start, anchor_t + 1):
            t_clamped = max(0, t)  # left-pad with frame 0
            frames.append(self._load_frame_images(f, t_clamped, compressed))
        return frames

    # ------------------------------------------------------------------ #
    # Torch Dataset API
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, timestep = self.indices[idx]
        episode_file = self.episode_files[episode_idx]

        with h5py.File(episode_file, "r") as f:
            compressed = bool(f.attrs.get("compress", False))
            qpos = self._load_qpos(f, timestep)  # (action_dim,)
            actions = self._load_actions(f, timestep)  # (future_action_window, action_dim)
            obs_frames = self._load_obs_images(f, timestep, compressed)

        # Normalize qpos / actions.
        state = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        actions_norm = (actions - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # Align with RoboTwin convention: state corresponds to action[0];
        # prediction target is action[1:].
        # In AgileX data, action[0] is the master arm command at `timestep`, which
        # corresponds to the slave qpos only after calibration. Using observed qpos
        # as state (instead of action[0]) better matches deployment conditions.
        state_tensor = torch.from_numpy(state).float()
        actions_to_predict = torch.from_numpy(actions_norm[1:]).float()

        # Build image tensor: (n_obs_steps, num_cameras, 3, H, W)
        per_frame_tensors: List[torch.Tensor] = []
        for frame in obs_frames:
            cam_tensors: List[torch.Tensor] = []
            for img in frame:
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                if self.transform is not None:
                    cam_tensors.append(self.transform(img))
                else:
                    cam_tensors.append(
                        torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    )
            per_frame_tensors.append(torch.stack(cam_tensors, dim=0))
        images_tensor = torch.stack(per_frame_tensors, dim=0)

        return {
            "images": images_tensor,
            "state": state_tensor,
            "actions": actions_to_predict,
            "episode_idx": episode_idx,
            "timestep": timestep,
        }


# ---------------------------------------------------------------------- #
# CLI self-test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the AgileX dataloader.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    args = parser.parse_args()

    stats = compute_agilex_norm_stats(args.data_path, args.num_episodes)
    print("qpos_mean:", stats["qpos_mean"])
    print("qpos_std :", stats["qpos_std"])

    dataset = AgileXDataset(
        data_path=args.data_path,
        norm_stats=stats,
        future_action_window=13,
        num_cameras=3,
        n_obs_steps=2,
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
