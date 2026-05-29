"""Real-robot inference for NemoDiT on AgileX (松灵) / Mobile-Aloha hardware.

This script mirrors the topology of ``agx_robot/aloha-devel/act/inference.py``
(ROS-based master/puppet arms + 3 RealSense cameras) but replaces the ACT
policy with a trained ``ActionModel`` (Flow-Matching DiT). It reads a
checkpoint produced by ``train_agilex.py`` — which embeds both the training
hyper-parameters and the qpos normalization stats — so that the same file is
sufficient for deployment.

Convention (matches ``dataloader_agilex.py``):
  * Input  : qpos[t] (observed slave joints) -> state token; image frames -> vision token.
  * Output : qpos[t+1 .. t+n_action_steps] (future joint positions).
  * Predictions are sent as joint commands to /master/joint_left|right; the
    master arm controller drives the slave to these target qpos.
  * /action from the HDF5 was never read during training; we never produce
    "master commands" directly, only desired qpos.

Topology of a control tick:
  1. ROS callbacks buffer camera frames + joint states.
  2. ``get_frame()`` returns time-synchronized sensor data.
  3. The observation is pushed into a fixed-length ``deque`` of length
     ``n_obs_steps``; on the first tick the deque is filled by duplication.
  4. The current slave qpos is normalized and used as the DiT ``state`` token.
  5. ``model.sample(...)`` returns ``(1, n_action_steps, action_dim)`` normalized
     actions; we denormalize and publish them one per tick at ``publish_rate``.
  6. When the execution queue drains, the cycle repeats with fresh observations.

Usage::

    python inference_agilex.py \
        --checkpoint checkpoints/agilex-run/final.pt \
        --publish_rate 40

Checkpoint requirements:
    ``torch.load(checkpoint)`` must yield a dict that contains
    ``model_state_dict``, ``args`` (from ``train_agilex.py``), and either
    ``norm_stats`` or a sibling ``dataset_stats.pkl`` file.
"""

from __future__ import annotations

import argparse
import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

# ROS imports are kept optional so the file can be imported on a workstation
# without a ROS install (e.g. just for static analysis / tests).
# NOTE: we deliberately do NOT use cv_bridge.imgmsg_to_cv2 because its
# cv_bridge.boost.cv_bridge_boost C++ extension transitively pulls libgdal +
# libtiff and breaks on many Ubuntu installs (undefined symbol
# TIFFReadRGBATileExt). For raw color frames we just np.frombuffer the bytes.
try:  # pragma: no cover - only available on the robot PC
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Header
    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ROS_AVAILABLE = False

from dataloader_agilex import (
    AGILEX_CAMERA_NAMES,
    AGILEX_STATE_DIM,
    agilex_action_dim,
)
from model.action_model.action_model import ActionModel


# ------------------------------------------------------------------ #
# Policy wrapper
# ------------------------------------------------------------------ #


class AgileXPolicy:
    """Thin wrapper around ``ActionModel`` that handles normalization + queueing."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        num_inference_steps: Optional[int] = None,
        ode_solver: str = "midpoint",
        cfg_scale: float = 1.0,
        stats_path: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"[AgileXPolicy] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        train_args = ckpt.get("args", {})
        if not train_args:
            raise ValueError(
                "Checkpoint does not contain training args; cannot rebuild the model."
            )

        self.n_obs_steps = int(train_args.get("n_obs_steps", 2))
        self.n_action_steps = int(train_args.get("n_action_steps", 8))
        self.num_cameras = int(train_args.get("num_cameras", 3))
        self.camera_names = train_args.get("camera_names") or AGILEX_CAMERA_NAMES
        self.camera_names = list(self.camera_names[: self.num_cameras])
        self.use_robot_base = bool(train_args.get("use_robot_base", False))
        expected_dim = agilex_action_dim(self.use_robot_base)
        self.action_dim = int(train_args.get("action_dim", expected_dim))
        assert self.action_dim == expected_dim, (
            f"Checkpoint action_dim={self.action_dim} disagrees with "
            f"use_robot_base={self.use_robot_base} (expected {expected_dim})"
        )
        self.cfg_scale = cfg_scale
        self.ode_solver = ode_solver
        self.num_inference_steps = (
            int(num_inference_steps)
            if num_inference_steps is not None
            else int(train_args.get("num_inference_steps", 10))
        )

        # Build model with the *exact* training configuration.
        self.model = ActionModel(
            token_size=int(train_args.get("token_size", 2048)),
            model_type=train_args.get("model_type", "DiT-B"),
            in_channels=self.action_dim,
            future_action_window_size=int(train_args.get("future_action_window", 13)),
            past_action_window_size=int(train_args.get("past_action_window", 0)),
            time_sampling=train_args.get("time_sampling", "logit_normal"),
            logit_normal_loc=float(train_args.get("logit_normal_loc", 0.0)),
            logit_normal_scale=float(train_args.get("logit_normal_scale", 1.0)),
            beta_alpha=float(train_args.get("beta_alpha", 1.5)),
            beta_beta=float(train_args.get("beta_beta", 1.0)),
            num_timestep_buckets=int(train_args.get("num_timestep_buckets", 1000)),
            use_vision_condition=True,
            vision_backbone_type=train_args.get("vision_backbone", "resnet50"),
            vision_pretrained=False,
            num_cameras=self.num_cameras,
            freeze_vision_backbone=False,
            adapter_type=train_args.get("adapter_type", "mlp"),
            class_dropout_prob=0.0,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            temporal_agg=train_args.get("temporal_agg", "concat"),
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.to(self.device).eval()

        # Normalization stats. Prefer what the checkpoint stored; else fall back
        # to a sibling pickle file.
        stats = ckpt.get("norm_stats")
        if stats is None:
            default_path = stats_path or os.path.join(
                os.path.dirname(os.path.abspath(checkpoint_path)),
                train_args.get("stats_name", "dataset_stats.pkl"),
            )
            with open(default_path, "rb") as f:
                stats = pickle.load(f)
        # AgileX dataloader trains state and target from the same qpos stream,
        # so qpos_mean/std and action_mean/std are equal by construction. Reading
        # them under both names lets us keep the symmetry: normalize the
        # observed qpos with `qpos_*`, denormalize the predicted future qpos
        # with `action_*`. They are numerically identical.
        self.qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)

        # Observation cache (images are pre-normalized float32 (K, 3, H, W) arrays).
        self._image_cache: Deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self._latest_qpos: Optional[np.ndarray] = None

        # Rolling queue of actions that have already been sampled.
        self.action_queue: List[np.ndarray] = []

        self._imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(
            f"[AgileXPolicy] Ready — model={train_args.get('model_type')} "
            f"action_dim={self.action_dim} n_obs_steps={self.n_obs_steps} "
            f"n_action_steps={self.n_action_steps} num_cameras={self.num_cameras}"
        )

    # -------------------------------------------------------------- #
    # Observation processing
    # -------------------------------------------------------------- #

    def preprocess_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """Convert a list of per-camera BGR uint8 frames to (K, 3, H, W) float32.

        Matches ``transforms.ToTensor + Normalize`` with ImageNet stats, sans
        resize (training default on AgileX data).
        """
        if len(images_bgr) != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} camera frames, got {len(images_bgr)}"
            )
        per_cam: List[np.ndarray] = []
        for img in images_bgr:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Camera frame must be HxWx3, got shape {img.shape}")
            arr = img.astype(np.float32) / 255.0
            arr = (arr - self._imagenet_mean) / self._imagenet_std
            arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
            per_cam.append(arr)
        return np.stack(per_cam, axis=0)  # (K, 3, H, W)

    def reset(self) -> None:
        """Clear cached observations and queued actions (call at episode start)."""
        self._image_cache.clear()
        self._latest_qpos = None
        self.action_queue = []

    def update_obs(self, images_bgr: List[np.ndarray], qpos: np.ndarray) -> None:
        frame = self.preprocess_images(images_bgr)
        if not self._image_cache:
            for _ in range(self.n_obs_steps):
                self._image_cache.append(frame)
        else:
            self._image_cache.append(frame)
        self._latest_qpos = np.asarray(qpos, dtype=np.float32)

    # -------------------------------------------------------------- #
    # Inference
    # -------------------------------------------------------------- #

    @torch.no_grad()
    def predict(self) -> np.ndarray:
        """Run one ODE sampling pass and return the denormalized action chunk.

        Returns:
            actions: ``(n_action_steps, action_dim)`` numpy array in radians.
        """
        if self._latest_qpos is None or not self._image_cache:
            raise RuntimeError("update_obs() must be called before predict().")

        images = np.stack(list(self._image_cache), axis=0)  # (T, K, 3, H, W)
        images = np.expand_dims(images, axis=0)  # (1, T, K, 3, H, W)
        images_t = torch.from_numpy(images).float().to(self.device)

        qpos_norm = (self._latest_qpos - self.qpos_mean) / self.qpos_std
        state_t = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)

        out = self.model.sample(
            images_t,
            state=state_t,
            num_steps=self.num_inference_steps,
            ode_solver=self.ode_solver,
            cfg_scale=self.cfg_scale,
            return_all=False,
        )  # (1, n_action_steps, action_dim)
        out_np = out.cpu().numpy()[0]
        return out_np * self.action_std + self.action_mean

    def get_action(self, images_bgr: List[np.ndarray], qpos: np.ndarray) -> np.ndarray:
        """Return the next action, re-planning when the execution queue is empty."""
        self.update_obs(images_bgr, qpos)
        if not self.action_queue:
            chunk = self.predict()  # (n_action_steps, action_dim)
            self.action_queue = [chunk[i] for i in range(chunk.shape[0])]
        return self.action_queue.pop(0)


# ------------------------------------------------------------------ #
# ROS I/O layer
# ------------------------------------------------------------------ #


def _ros_image_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to a HxWxC numpy array without cv_bridge.

    Handles the encodings produced by RealSense color streams (``bgr8``,
    ``rgb8``) and the common AgileX collect_data passthrough case. Returns a
    writable copy so the downstream preprocessing pipeline can mutate it.
    """
    encoding = (msg.encoding or "").lower()
    raw = msg.data
    height, width = msg.height, msg.width

    if encoding in ("bgr8", "rgb8"):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        return arr.copy()
    if encoding in ("bgra8", "rgba8"):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        return arr[..., :3].copy()
    if encoding in ("mono8", "8uc1"):
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width)).copy()
    if encoding in ("mono16", "16uc1"):
        return np.frombuffer(raw, dtype=np.uint16).reshape((height, width)).copy()

    # Generic fallback: infer channel count from `step` (bytes per row).
    step = getattr(msg, "step", 0)
    channels = max(1, step // max(1, width)) if step else 3
    arr = np.frombuffer(raw, dtype=np.uint8)
    if channels == 1:
        return arr.reshape((height, width)).copy()
    return arr.reshape((height, width, channels)).copy()


class AgileXRosBridge:
    """Subscribe / publish wrapper (same topics as agx_robot ACT inference)."""

    def __init__(self, args: argparse.Namespace):
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rospy / cv_bridge are not available. This script must be run on "
                "the AgileX machine inside a ROS environment."
            )
        self.args = args
        # No CvBridge — see module docstring; we decode raw bytes ourselves.

        self.img_front_deque: Deque = deque(maxlen=2000)
        self.img_left_deque: Deque = deque(maxlen=2000)
        self.img_right_deque: Deque = deque(maxlen=2000)
        self.puppet_arm_left_deque: Deque = deque(maxlen=2000)
        self.puppet_arm_right_deque: Deque = deque(maxlen=2000)
        self.robot_base_deque: Deque = deque(maxlen=2000)

        self._init_ros()

    # ---- Subscribers ----
    def _init_ros(self) -> None:
        rospy.init_node("nemodit_agilex_inference", anonymous=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self._img_front_cb,
                         queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self._img_left_cb,
                         queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self._img_right_cb,
                         queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState,
                         self._puppet_left_cb, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState,
                         self._puppet_right_cb, queue_size=1000, tcp_nodelay=True)
        if self.args.use_robot_base:
            rospy.Subscriber(self.args.robot_base_topic, Odometry,
                             self._robot_base_cb, queue_size=1000, tcp_nodelay=True)

        self.pub_left = rospy.Publisher(self.args.puppet_arm_left_cmd_topic,
                                        JointState, queue_size=10)
        self.pub_right = rospy.Publisher(self.args.puppet_arm_right_cmd_topic,
                                         JointState, queue_size=10)
        if self.args.use_robot_base:
            self.pub_base = rospy.Publisher(self.args.robot_base_cmd_topic,
                                            Twist, queue_size=10)
        else:
            self.pub_base = None

    # ---- Callbacks ----
    def _img_front_cb(self, msg): self.img_front_deque.append(msg)
    def _img_left_cb(self, msg): self.img_left_deque.append(msg)
    def _img_right_cb(self, msg): self.img_right_deque.append(msg)
    def _puppet_left_cb(self, msg): self.puppet_arm_left_deque.append(msg)
    def _puppet_right_cb(self, msg): self.puppet_arm_right_deque.append(msg)
    def _robot_base_cb(self, msg): self.robot_base_deque.append(msg)

    # ---- Diagnostics ----
    def topic_status(self) -> Dict[str, int]:
        """Return per-topic message counts in the buffers (for debugging)."""
        return {
            "img_front": len(self.img_front_deque),
            "img_left": len(self.img_left_deque),
            "img_right": len(self.img_right_deque),
            "puppet_left": len(self.puppet_arm_left_deque),
            "puppet_right": len(self.puppet_arm_right_deque),
            "robot_base": len(self.robot_base_deque),
        }

    def frame_diagnose(self) -> str:
        """Describe why get_frame() would currently fail (for waiting state)."""
        s = self.topic_status()
        empty = [k for k, v in s.items() if v == 0
                 and (k != "robot_base" or self.args.use_robot_base)]
        if empty:
            return f"empty deques: {empty}"
        # All non-empty: check stamp sync.
        try:
            frame_time = min(
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_left_deque[-1].header.stamp.to_sec(),
                self.img_right_deque[-1].header.stamp.to_sec(),
            )
        except Exception as exc:
            return f"could not compute frame_time: {exc}"
        lag = {}
        for k, dq in (("img_front", self.img_front_deque),
                      ("img_left", self.img_left_deque),
                      ("img_right", self.img_right_deque),
                      ("puppet_left", self.puppet_arm_left_deque),
                      ("puppet_right", self.puppet_arm_right_deque)):
            if dq:
                lag[k] = round(dq[-1].header.stamp.to_sec() - frame_time, 3)
        return f"all topics have data; latest-stamp vs frame_time={lag}"

    # ---- Sync ----
    def get_frame(self) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """Return (images_bgr, qpos) for the most recent synchronized timestep.

        Images are ordered to match ``AGILEX_CAMERA_NAMES`` =
        ``[cam_high(front), cam_left_wrist(left), cam_right_wrist(right)]``.
        """
        if not (self.img_front_deque and self.img_left_deque and self.img_right_deque
                and self.puppet_arm_left_deque and self.puppet_arm_right_deque):
            return None
        if self.args.use_robot_base and not self.robot_base_deque:
            return None

        frame_time = min(
            self.img_front_deque[-1].header.stamp.to_sec(),
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
        )

        def _wait(deque_: Deque) -> bool:
            return deque_[-1].header.stamp.to_sec() >= frame_time

        if not all(_wait(d) for d in (self.img_front_deque, self.img_left_deque,
                                      self.img_right_deque,
                                      self.puppet_arm_left_deque,
                                      self.puppet_arm_right_deque)):
            return None
        if self.args.use_robot_base and not _wait(self.robot_base_deque):
            return None

        def _drain(deque_: Deque):
            # Exhaust all messages older than `frame_time`, then return the
            # first one with stamp >= frame_time (matches ACT's drain loop).
            while len(deque_) > 1 and deque_[0].header.stamp.to_sec() < frame_time:
                deque_.popleft()
            return deque_.popleft()

        img_front = _ros_image_to_numpy(_drain(self.img_front_deque))
        img_left = _ros_image_to_numpy(_drain(self.img_left_deque))
        img_right = _ros_image_to_numpy(_drain(self.img_right_deque))
        pup_left = _drain(self.puppet_arm_left_deque)
        pup_right = _drain(self.puppet_arm_right_deque)

        qpos = np.concatenate(
            [np.asarray(pup_left.position, dtype=np.float32),
             np.asarray(pup_right.position, dtype=np.float32)],
            axis=0,
        )
        if self.args.use_robot_base:
            base = _drain(self.robot_base_deque)
            qpos = np.concatenate(
                [qpos,
                 np.asarray([base.twist.twist.linear.x,
                             base.twist.twist.angular.z], dtype=np.float32)],
                axis=0,
            )

        # cv_bridge passthrough preserves the camera encoding; for AgileX RealSense
        # color streams this is typically BGR8, matching the training pipeline.
        return [img_front, img_left, img_right], qpos

    # ---- Publishing ----
    def _build_joint_state(self, positions: np.ndarray) -> "JointState":
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f"joint{i}" for i in range(len(positions))]
        msg.position = positions.tolist()
        return msg

    def publish_arms(self, left_cmd: np.ndarray, right_cmd: np.ndarray) -> None:
        self.pub_left.publish(self._build_joint_state(left_cmd))
        self.pub_right.publish(self._build_joint_state(right_cmd))

    def publish_base(self, linear_x: float, angular_z: float) -> None:
        if self.pub_base is None:
            return
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.pub_base.publish(twist)

    # ---- Soft-start ----
    def soft_start_to_pose(
        self,
        target_left: np.ndarray,
        target_right: np.ndarray,
        step_size: float = 0.01,
        rate_hz: int = 200,
        wait_timeout: float = 10.0,
    ) -> None:
        """Slowly move both master arms from their current position to a target pose.

        Mirrors ``agx_robot/aloha-devel/act/inference.puppet_arm_publish_continuous``:
        each tick advances every joint by at most ``step_size`` toward the
        target until all joints have arrived. Prevents the slave from snapping
        when the policy's first prediction is far from the current pose.

        Args:
            target_left, target_right: 7-D home poses for left / right arm.
            step_size: Max per-tick joint delta (rad).
            rate_hz: Tick rate during the ramp.
            wait_timeout: Seconds to wait for the first /puppet/joint_* message
                before aborting with a clear error.
        """
        # Wait for at least one observation from each arm so we know where we are.
        rate = rospy.Rate(rate_hz)
        start = rospy.Time.now()
        last_print = start
        print("[AgileX] Waiting for first /puppet/joint_left and /puppet/joint_right messages...")
        while (not rospy.is_shutdown()) and (
            not self.puppet_arm_left_deque or not self.puppet_arm_right_deque
        ):
            now = rospy.Time.now()
            elapsed = (now - start).to_sec()
            if elapsed > wait_timeout:
                raise RuntimeError(
                    f"[AgileX] Timed out after {wait_timeout:.1f}s waiting for "
                    f"/puppet/joint_left or /puppet/joint_right. "
                    f"left_msgs={len(self.puppet_arm_left_deque)} "
                    f"right_msgs={len(self.puppet_arm_right_deque)}. "
                    f"Is the puppet arm driver running? Try: "
                    f"`rostopic hz /puppet/joint_left` in another terminal."
                )
            if (now - last_print).to_sec() >= 1.0:
                print(f"  ...still waiting ({elapsed:.1f}s)  "
                      f"left={len(self.puppet_arm_left_deque)} "
                      f"right={len(self.puppet_arm_right_deque)}")
                last_print = now
            rate.sleep()
        if rospy.is_shutdown():
            return

        left_cur = np.asarray(self.puppet_arm_left_deque[-1].position, dtype=np.float32)
        right_cur = np.asarray(self.puppet_arm_right_deque[-1].position, dtype=np.float32)
        target_left = np.asarray(target_left, dtype=np.float32)
        target_right = np.asarray(target_right, dtype=np.float32)
        print(f"[AgileX] Soft-start starting from "
              f"L={left_cur.round(3).tolist()}  R={right_cur.round(3).tolist()}")
        print(f"[AgileX] Target home pose       "
              f"L={target_left.round(3).tolist()}  R={target_right.round(3).tolist()}")

        while not rospy.is_shutdown():
            left_diff = target_left - left_cur
            right_diff = target_right - right_cur
            left_cur = left_cur + np.clip(left_diff, -step_size, step_size)
            right_cur = right_cur + np.clip(right_diff, -step_size, step_size)
            self.publish_arms(left_cur, right_cur)
            if (np.all(np.abs(target_left - left_cur) <= 1e-6)
                    and np.all(np.abs(target_right - right_cur) <= 1e-6)):
                break
            rate.sleep()


# ------------------------------------------------------------------ #
# Control loop
# ------------------------------------------------------------------ #


def _parse_pose_arg(s: Optional[str], expected_dim: int = 7) -> Optional[np.ndarray]:
    """Parse comma-separated floats into a numpy array, or return None."""
    if not s:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} comma-separated values, got {len(parts)}: {s!r}"
        )
    return np.asarray(parts, dtype=np.float32)


def run_inference(args: argparse.Namespace) -> None:
    policy = AgileXPolicy(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        ode_solver=args.ode_solver,
        cfg_scale=args.cfg_scale,
        stats_path=args.stats_path,
    )
    bridge = AgileXRosBridge(args)

    # Print the topics we're touching so it's obvious if a default is wrong.
    print("[AgileX] Subscribed topics:")
    print(f"   front cam : {args.img_front_topic}")
    print(f"   left  cam : {args.img_left_topic}")
    print(f"   right cam : {args.img_right_topic}")
    print(f"   puppet L  : {args.puppet_arm_left_topic}")
    print(f"   puppet R  : {args.puppet_arm_right_topic}")
    if args.use_robot_base:
        print(f"   base odom : {args.robot_base_topic}")
    print("[AgileX] Publishing topics:")
    print(f"   master L  : {args.puppet_arm_left_cmd_topic}")
    print(f"   master R  : {args.puppet_arm_right_cmd_topic}")
    if args.use_robot_base:
        print(f"   cmd_vel   : {args.robot_base_cmd_topic}")

    # 1. Soft-start: ramp master to a home pose so the first policy prediction
    #    can't yank the slave across joint space. Defaults match ACT inference's
    #    `left0 / right0` constants; users can override via CLI.
    if not args.no_soft_start:
        left0 = _parse_pose_arg(args.soft_start_left, 7) or np.array(
            [-0.00133514, 0.00209808, 0.01583099, -0.03261662,
             -0.00286102, 0.00095367, 3.55783081], dtype=np.float32
        )
        right0 = _parse_pose_arg(args.soft_start_right, 7) or np.array(
            [-0.00133514, 0.00438690, 0.03452396, -0.05359745,
             -0.00476837, -0.00209808, 3.55783081], dtype=np.float32
        )
        print("[AgileX] Soft-starting to home pose...")
        bridge.soft_start_to_pose(left0, right0, step_size=args.soft_start_step)
        if args.soft_start_pause:
            try:
                input("[AgileX] Soft-start done. Press <Enter> to begin policy inference...")
            except EOFError:
                pass

    # 1b. Obs cache warm-up: the very first predict() call would otherwise see
    #     a duplicated first frame in the n_obs_steps deque (update_obs() fills
    #     by repetition on the first call). That input is OOD compared to the
    #     30Hz consecutive frames the model trained on. By collecting
    #     `warmup_obs_frames` real frames here, the first published prediction
    #     is made on a fully real temporal window -- effectively the equivalent
    #     of "discard the first chunk".
    n_warmup = (args.warmup_obs_frames
                if args.warmup_obs_frames is not None
                else policy.n_obs_steps)
    if n_warmup > 0:
        print(f"[AgileX] Warming up obs cache: collecting {n_warmup} real frames...")
        collected = 0
        warmup_rate = rospy.Rate(args.publish_rate)
        while collected < n_warmup and not rospy.is_shutdown():
            frame = bridge.get_frame()
            if frame is None:
                rospy.sleep(0.005)
                continue
            images_bgr, qpos = frame
            policy.update_obs(images_bgr, qpos)
            collected += 1
            print(f"  obs frame {collected}/{n_warmup} collected  "
                  f"qpos[14]={qpos.round(3).tolist()}")
            warmup_rate.sleep()  # pace at publish_rate so frames differ
        print("[AgileX] Obs cache ready.")

    # 2. Threaded inference: model.sample() can take 50-150ms; if we ran it in
    #    the publish loop we'd miss ticks at publish_rate=40Hz. Inference thread
    #    pulls a fresh frame on demand, fills the queue, signals "ready".
    inference_lock = threading.Lock()
    state = {
        "shutdown": False,
        "ready": False,                       # True once at least one prediction is queued
        "wants_replan": True,                 # True when the queue should be refilled
    }

    def _inference_worker():
        chunk_count = 0
        none_count = 0
        last_status_print = rospy.Time.now()
        first_frame_seen = False
        try:
            while not rospy.is_shutdown() and not state["shutdown"]:
                with inference_lock:
                    wants = state["wants_replan"]
                if not wants:
                    rospy.sleep(0.001)
                    continue
                frame = bridge.get_frame()
                if frame is None:
                    none_count += 1
                    now = rospy.Time.now()
                    if (now - last_status_print).to_sec() >= 1.0:
                        print(f"[infer] no synced frame yet (try #{none_count})  "
                              f"-> {bridge.frame_diagnose()}")
                        last_status_print = now
                    rospy.sleep(0.005)
                    continue
                images_bgr, qpos = frame
                if not first_frame_seen:
                    first_frame_seen = True
                    shapes = [im.shape for im in images_bgr]
                    print(f"[infer] first synced frame OK: image shapes={shapes}  "
                          f"qpos[14]={qpos.round(3).tolist()}")
                # First call also seeds the n_obs_steps deque.
                t0 = rospy.Time.now()
                with torch.inference_mode():
                    policy.update_obs(images_bgr, qpos)
                    chunk = policy.predict()  # (n_action_steps, action_dim)
                dt_ms = (rospy.Time.now() - t0).to_sec() * 1000.0
                with inference_lock:
                    policy.action_queue = [chunk[i] for i in range(chunk.shape[0])]
                    state["ready"] = True
                    state["wants_replan"] = False
                chunk_count += 1
                if args.verbose or chunk_count <= 3 or chunk_count % 20 == 0:
                    pred0 = chunk[0]
                    diff = pred0 - qpos
                    print(f"[infer #{chunk_count}] {dt_ms:6.1f}ms  "
                          f"qpos_now={qpos.round(3).tolist()}\n"
                          f"           pred[0]={pred0.round(3).tolist()}\n"
                          f"           diff   ={diff.round(3).tolist()}  "
                          f"(max |Δ| = {np.abs(diff).max():.4f})")
        except Exception as exc:
            # Make sure exceptions in this thread are visible -- by default
            # uncaught exceptions in threads only print to stderr at process
            # exit, which can hide the real cause of "nothing moves" issues.
            import traceback
            print("[infer ERROR] worker thread crashed:")
            traceback.print_exc()
            state["shutdown"] = True
            raise

    inference_thread = threading.Thread(target=_inference_worker, daemon=True)
    inference_thread.start()

    # 3. Publish loop at fixed publish_rate; replans when the queue drains.
    rate = rospy.Rate(args.publish_rate)
    print(f"[AgileX] Publishing at {args.publish_rate} Hz (Ctrl+C to stop)...")
    last_action: Optional[np.ndarray] = None
    publish_tick = 0
    waiting_ticks = 0
    try:
        while not rospy.is_shutdown():
            if args.max_steps is not None and args.max_steps <= 0:
                break

            with inference_lock:
                if not state["ready"]:
                    queued = []
                else:
                    queued = policy.action_queue

            if queued:
                action = queued.pop(0)
                with inference_lock:
                    if not policy.action_queue:
                        state["wants_replan"] = True  # ask worker to fetch next chunk
                last_action = action
            elif last_action is not None:
                # Queue is empty and replan in flight: hold the last command so
                # the master doesn't go limp during the inference gap.
                action = last_action
            else:
                # No prediction yet at all — wait.
                waiting_ticks += 1
                if waiting_ticks % args.publish_rate == 0:
                    print(f"[publish] waiting for first prediction ... "
                          f"({waiting_ticks / args.publish_rate:.1f}s)  "
                          f"state.ready={state['ready']}  "
                          f"thread_alive={inference_thread.is_alive()}")
                rate.sleep()
                continue

            bridge.publish_arms(action[:7], action[7:14])
            if policy.use_robot_base and action.shape[0] >= 16:
                bridge.publish_base(action[14], action[15])

            publish_tick += 1
            if args.verbose or publish_tick <= 3 or publish_tick % (args.publish_rate * 2) == 0:
                print(f"[publish #{publish_tick:5d}] "
                      f"L={action[:7].round(3).tolist()}  R={action[7:14].round(3).tolist()}")

            if args.max_steps is not None:
                args.max_steps -= 1
            rate.sleep()
    finally:
        # Graceful shutdown: stop the inference thread.
        state["shutdown"] = True
        inference_thread.join(timeout=2.0)
        print("[AgileX] Inference stopped.")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-robot NemoDiT inference for AgileX.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pt checkpoint produced by train_agilex.py")
    parser.add_argument("--stats_path", type=str, default=None,
                        help="Optional path to dataset_stats.pkl if not embedded in checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Euler/midpoint ODE steps (defaults to value from checkpoint)")
    parser.add_argument("--ode_solver", type=str, default="midpoint",
                        choices=["euler", "midpoint"])
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--publish_rate", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional cap on total publish ticks (for scripted tests)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Print every inference + publish event (very chatty).")

    # Soft-start (ramp master to a home pose before policy takes over).
    parser.add_argument("--no_soft_start", action="store_true", default=False,
                        help="Skip the ramp-to-home-pose step at startup.")
    parser.add_argument("--soft_start_left", type=str, default=None,
                        help="Comma-separated 7-D left arm home pose; overrides default.")
    parser.add_argument("--soft_start_right", type=str, default=None,
                        help="Comma-separated 7-D right arm home pose; overrides default.")
    parser.add_argument("--soft_start_step", type=float, default=0.01,
                        help="Max joint delta per tick during soft-start (rad/tick).")
    parser.add_argument("--soft_start_pause", action="store_true", default=False,
                        help="Pause for an Enter keypress after soft-start (like ACT).")

    # Obs cache warm-up
    parser.add_argument("--warmup_obs_frames", type=int, default=None,
                        help="Collect N real frames before the first inference so "
                             "the n_obs_steps deque is fully real (not padded "
                             "duplicates of the very first frame). Default = "
                             "checkpoint's n_obs_steps. Set to 0 to disable.")

    # ROS topics (defaults mirror agx_robot/aloha-devel/act/inference.py).
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--puppet_arm_left_cmd_topic", type=str, default="/master/joint_left")
    parser.add_argument("--puppet_arm_right_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--robot_base_topic", type=str, default="/odom_raw")
    parser.add_argument("--robot_base_cmd_topic", type=str, default="/cmd_vel")
    parser.add_argument("--use_robot_base", action="store_true", default=False,
                        help="Enable if the checkpoint was trained with --use_robot_base")

    return parser.parse_args()


if __name__ == "__main__":
    if not _ROS_AVAILABLE:
        raise SystemExit(
            "rospy is not available. Source your ROS setup (e.g. "
            "`source /opt/ros/noetic/setup.bash`) before running this script."
        )
    run_inference(parse_args())
