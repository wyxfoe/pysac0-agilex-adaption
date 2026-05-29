# NemoDiT 适配松灵 AgileX / Mobile-Aloha 真机

本仓库已经裁剪为**仅真机管线** —— 原 RoboTwin 仿真器、`code_gen` / `description` /
`envs` / `assets` 等仿真侧目录全部移除。剩下的代码全部围绕 cobot_magic 采集栈 +
`agx_robot` ACT pipeline 设计：

| 文件 | 作用 |
|------|------|
| `dataloader_agilex.py` | 读取 `collect_data.py` 保存的扁平 14-D AgileX HDF5 |
| `train_agilex.py` / `train_agilex.sh` | 在 AgileX 数据上训练 Flow-Matching DiT |
| `inference_agilex.py` / `deploy_agilex.sh` | ROS1 真机推理 (camera → 模型 → puppet arm cmd) |
| `inspect_hdf5.py` | HDF5 数据审查工具 |
| `model/` / `utils/` | DiT 主干 + ResNet/ViT 视觉 backbone + Flow Matching + EMA |

## 1. 数据格式

`agx_robot/collect_data/collect_data.py` 保存的 AgileX HDF5 结构：

```
episode_X.hdf5
├── attrs
│   ├── sim        (bool)      总为 False (真机)
│   └── compress   (bool)      True -> JPEG 压缩
├── /observations
│   ├── qpos       (T, 14)     左臂(7) + 右臂(7), 每臂 = 6 关节 + 1 夹爪
│   ├── qvel       (T, 14)
│   ├── effort     (T, 14)
│   └── images/{cam_high, cam_left_wrist, cam_right_wrist}
│                  (T, 480, 640, 3) uint8
├── /action        (T, 14)     master arm 命令 — 训练时**完全不使用**
└── /base_action   (T, 2)      [linear.x, angular.z]
```

关键点：

1. **14-D 布局恰好等价于 NemoDiT 的 `joint` 模式**
   (`[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`)，无需重排。
2. **state 与 target 都来自 `/observations/qpos`**：
   - `state  = qpos[t]`
   - `target = qpos[t+1 .. t+W-1]`
   - `/action` 完全弃用 → 模型变成 next-state regressor。
   - 归一化只用一套 qpos 统计（`action_mean/std` 在 dict 里仍存在，是
     `qpos_mean/std` 的副本，方便老代码读 `action_*` 不报错）。
3. **3 路相机**：`cam_high, cam_left_wrist, cam_right_wrist`（不再是 4 路）。
4. **可选 `compress`** —— dataloader 会自动 `cv2.imdecode` 并 BGR→RGB。
5. **可选机载底盘 (`/base_action`)** —— 训练 / 推理打开 `--use_robot_base` 后
   qpos / target 维度变为 16。

## 2. 训练

### 2.1 数据目录结构

`train_agilex.py --data_path` 指向的目录需含若干 `episode_N.hdf5`：

```
~/data/pick_place/
├── episode_0.hdf5
├── episode_1.hdf5
├── ...
```

### 2.2 直接调用 Python 脚本

```bash
cd NemoDiT

python train_agilex.py \
    --data_path ~/data/pick_place \
    --num_episodes 50 \
    --num_cameras 3 \
    --model_type DiT-B \
    --vision_backbone resnet50 \
    --adapter_type mlp \
    --n_obs_steps 2 \
    --n_action_steps 8 \
    --future_action_window 13 \
    --temporal_agg concat \
    --epochs 500 \
    --batch_size 32 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/pick_place-50-seed0
```

会在 `checkpoints/pick_place-50-seed0/` 下产出：

- `{epoch}.pt` / `final.pt` / `latest.pt` — checkpoint，`args` 和 `norm_stats` 嵌入其中
- `dataset_stats.pkl` — 与 ACT 兼容的归一化统计 (qpos_mean/std, action_mean/std)

### 2.3 Shell 脚本

```bash
bash train_agilex.sh pick_place 50 0 0 ~/data
# args: <task_name> <expert_data_num> <seed> <gpu_id> [data_root]
```

### 2.4 关键超参

| 参数 | 说明 | 默认 |
|------|------|------|
| `--num_cameras` | 相机数 (最多 3) | `3` |
| `--future_action_window` | state+预测总帧数；实际预测 `window-1` 帧 | **`10`** (= 9 预测帧 ≈ 300ms @ 30Hz) |
| `--n_obs_steps` | 视觉历史帧数 | `2` |
| `--n_action_steps` | 推理时一次执行的帧数 (receding horizon) | **`4`** (闭环重规划 7.5Hz @ 30Hz) |
| `--temporal_agg` | 多帧特征聚合 (`last/mean/concat`) | `concat` |
| `--use_robot_base` | 把 `/base_action` 拼接到 qpos/action | `False` |
| `--arm_delay_time` | 前移 action 目标帧数 (同 ACT dataloader) | `0` |
| `--exclude_terminal_padding` | 丢掉 episode 末端需要 padding 的样本 | `False` |
| `--val_ratio` / `--val_every` | 验证集比例 / 验证频率 | `0.1 / 10` |
| `--use_amp` | AMP fp16 | `False` |
| `--use_ema` | EMA 权重 | `False` |
| `--dropout_prob` | CFG 训练时条件丢弃概率（推理 `--cfg_scale > 1` 才生效） | **`0.0`** (不用 CFG) |
| `--num_inference_steps` | Flow Matching ODE 积分步数 | `10` |

> **约束**：`n_obs_steps + n_action_steps <= future_action_window`。

### 2.4.1 单臂任务约定（无单/双臂分支）

任何 AgileX 数据都按 **14-D 双臂** 训练。即使采集时只动一只手臂（另一只
qpos 几乎恒定），也保留 14 维：

- 静止那一侧会被归一化成接近常数的目标，模型几乎无代价学到。
- 推理时仍向 `/master/joint_left` 和 `/master/joint_right` 各发一组命令；
  静止侧会回到训练数据中那个常量姿态（一般就是采集时的"静止位"），可以当作
  自动的安全姿势锁定。

如果你担心静止维度被 `std` clip (1e-2) 后放大噪声，可以：
- 用 `inspect_hdf5.py --only /observations/qpos` 检查每维真实 std；
- 或不动它（推荐）—— 静止数据的归一化噪声本身也微小。

### 2.4.2 验证集 / 早停

```bash
python train_agilex.py \
    --data_path ~/data/pick_place \
    --num_episodes 100 \
    --val_ratio 0.1 --val_every 10 --val_seed 0 \
    --checkpoint_dir checkpoints/pick-100
```

- `val_ratio` 按 episode 切分（同一个 episode 不会同时进 train / val）。
- 归一化统计**只用训练集**计算。
- 每 `val_every` 个 epoch 跑一次验证，val_loss 创新低时落盘 `best.pt`。

### 2.4.3 Episode 末端 padding

采集脚本写出的 episode 长度固定（例如 200 帧），dataloader 默认对最后
`future_action_window` 帧用末帧重复填充。这会教模型"快结束时就别动"。
若不想要这种偏置：

```bash
python train_agilex.py ... --exclude_terminal_padding
```

会丢掉约 `future_action_window / episode_length ≈ 6%` 的样本，但训练目标更干净。

### 2.5 Dataloader 自检

```bash
python dataloader_agilex.py --data_path ~/data/pick_place --num_episodes 3
```

会打印前若干个 episode 的归一化统计及首条样本的各字段形状。

## 3. 真机推理

`inference_agilex.py` 复用 `agx_robot/aloha-devel/act/inference.py` 的 ROS 话题：

```
Subscribe:
  /camera_f/color/image_raw   (cam_high)
  /camera_l/color/image_raw   (cam_left_wrist)
  /camera_r/color/image_raw   (cam_right_wrist)
  /puppet/joint_left          (从臂当前关节)
  /puppet/joint_right
  /odom_raw                   (可选, --use_robot_base)

Publish:
  /master/joint_left          (命令主臂)
  /master/joint_right
  /cmd_vel                    (可选, --use_robot_base)
```

### 3.1 启动

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash    # 包含 puppet_arm_publisher 依赖

python inference_agilex.py \
    --checkpoint checkpoints/pick_place-50-seed0/final.pt \
    --publish_rate 40 \
    --num_inference_steps 10 \
    --ode_solver midpoint
```

checkpoint 中已嵌入 `args` 与 `norm_stats`，无需额外传递。若仅有裸模型权重，
用 `--stats_path /path/to/dataset_stats.pkl` 指定。

### 3.2 控制循环（两线程模型）

```
┌────────────────────────────┐         ┌──────────────────────────┐
│  Inference thread          │         │  Publish thread (主)      │
│                            │         │                          │
│  1. get_frame() 同步        │         │  rospy.Rate(publish_rate)│
│  2. update_obs(deque)       │ queue   │  pop action from queue   │
│  3. policy.predict()        │ ─────►  │  split [:7]/[7:14] →     │
│     (ODE 10步, ~50-150ms)   │         │  publish_arms()          │
│  4. fill action_queue       │         │  当队列空: 重新触发       │
│  5. wait for replan signal  │         │     wants_replan=True    │
└────────────────────────────┘         └──────────────────────────┘
```

- 推理线程独立于 publish loop，避免 ODE 采样卡住 40Hz 节拍。
- 队列空 + 新推理在路上时，**持续发布上一帧动作** (`last_action` hold)，
  保证主臂不会"卡 0"。
- Ctrl+C 触发 `finally` 通知推理线程退出。

**首启动 soft-start**：默认从当前从臂位置以 `--soft_start_step` (默认 0.01 rad/tick) 的速度
线性插值到 home pose（见代码内置 `left0/right0` 常量），完成后才允许策略接管。
`--soft_start_pause` 会在 ramp 结束后等待 Enter（首次运行新任务推荐打开）。

跳过 soft-start：`--no_soft_start`。
自定义 home pose：`--soft_start_left "0,0,...,0" --soft_start_right "0,0,...,0"`（各 7 维）。

**Obs cache warm-up**：soft-start 完成后，正式推理前会**先收集 `--warmup_obs_frames`
帧真实观测**（默认 = checkpoint 中的 `n_obs_steps`，即 2 帧）填满 `obs deque`，
避免第一次 `predict()` 把 "两帧同一图像" 喂进 ResNet（这种输入分布外，模型
没在训练时见过）。等价于"丢弃第一个 chunk 不用"，但**不浪费 GPU 推理**。

- 强制关闭：`--warmup_obs_frames 0`
- 多收几帧：`--warmup_obs_frames 4`

### 3.3 数据流细节

1. ROS callbacks 缓冲相机帧 + 从臂 JointState。
2. `get_frame()` 取最新同步帧（exhaust-and-popleft）；缺失则等。
3. 观测进入长度 `n_obs_steps` 的 deque (首轮复制填充)。
4. `qpos` 按训练时的 mean/std 归一化，作为 DiT `state` token。
5. `ActionModel.sample(num_steps, ode_solver, cfg_scale)` 生成
   `(1, n_action_steps, action_dim)` 动作；反归一化后入队。
6. 每个 tick 从队列弹出一帧，14-D 切两半发布：
   - `action[:7]`  → `/master/joint_left`
   - `action[7:14]` → `/master/joint_right`
   若训练时开了 base (`use_robot_base`)，再解包 `action[14:16]` 发到 `/cmd_vel`。

### 3.2.1 启用 Classifier-Free Guidance（可选）

训练时默认 `--dropout_prob 0.1`，意味着 10% 的样本会用无条件 token 替换视觉
条件。推理时只要把 `--cfg_scale > 1` 调高，就能用 guidance：

```bash
python inference_agilex.py --checkpoint final.pt --cfg_scale 1.5
```

`cfg_scale=1.0`（默认）等价于无 guidance，相当于训练时的 dropout 白浪费。
真机上可以从 `1.0 → 1.3 → 1.5` 渐增观察控制是否更稳。

### 3.3 与 ACT 推理的差异

| 项 | ACT inference.py | inference_agilex.py |
|----|------------------|---------------------|
| 观测帧 | 单帧 | `n_obs_steps` 多帧 (deque) |
| 条件 | qpos + images | qpos (state token) + images (vision token) |
| 策略 | Transformer 预测整段 chunk | Flow-Matching DiT ODE 采样 |
| temporal_agg | 指数加权 (all_time_actions) | receding horizon (执行队列) |
| 动作插值 | 可选 `--use_actions_interpolation` | 由 DiT 本身保证轨迹连贯 |

若需要软着陆启动 / 安全姿态序列，可复用 ACT inference 中的
`puppet_arm_publish_continuous` / `puppet_arm_publish_linear` 逻辑放在
`run_inference` 的最开始。

## 4. 环境依赖

### 训练机
```
pip install torch torchvision h5py numpy tqdm timm opencv-python wandb
```

### 真机 (AgileX, ROS Noetic = ROS1)

代码层面 `inference_agilex.py` 已经是纯 ROS1 (`rospy`/`cv_bridge`/`sensor_msgs`)，
但要让 `torch` 和 `rospy` 在**同一个 Python 解释器**里能 import 是真机部署的最大坑：

1. **方案 A — 复用 aloha conda env (Python 3.8)**（推荐）

   ```bash
   conda env create -f agx_robot/aloha.yml      # 已包含 rospy, cv_bridge, torch
   conda activate aloha
   pip install timm tqdm h5py                    # NemoDiT 额外依赖
   source /opt/ros/noetic/setup.bash             # 引入系统 ROS 路径
   source ~/catkin_ws/devel/setup.bash           # 你的 puppet_arm 工作空间
   ```

2. **方案 B — 独立 conda env (Python 3.9+, 想用更新的 torch)**

   `cv_bridge` 默认链接 Python 3.8，conda 3.9+ 直接 import 会失败。两种修法：
   ```bash
   # B1: 在 conda env 里 pip 重装与 ROS Python 版本无关的纯 Python cv_bridge
   pip install cv_bridge3                        # 第三方包，纯 Python，仅 RGB/depth 够用
   # B2: 在 conda env 里从源码编译 cv_bridge
   git clone https://github.com/ros-perception/vision_opencv.git
   cd vision_opencv/cv_bridge && pip install -e .
   ```

3. **常见检查命令**
   ```bash
   python -c "import rospy; print(rospy.__file__)"   # 应当指向 /opt/ros/noetic/...
   python -c "from cv_bridge import CvBridge"
   python -c "import torch; print(torch.cuda.is_available())"
   rostopic list | grep -E '(camera|puppet)'         # 确认采集端在发数据
   ```

4. **额外 ROS 依赖**
   - `puppet_arm_publish_*` 节点 (见 `agx_robot/follow_control/`)
   - RealSense 驱动 (`/camera_f /camera_l /camera_r/color/image_raw`)
   - Master / puppet 主从映射节点（采集时用的同一套）

## 5. 常见坑

1. **图像色彩空间**
   - `agx_robot/collect_data.py` 使用 `cv_bridge.imgmsg_to_cv2(msg, 'passthrough')`，
     RealSense `rgb8` → RGB，`bgr8` → BGR。
   - `dataloader_agilex._decode_image` 只在 `compress=True` 时做 BGR→RGB。
   - 推理时 `get_frame()` 同样 passthrough。
   - 如果你的数据采集和推理使用的相机 encoding 不一致，训练和部署会对不上。
2. **关节顺序**
   - AgileX `puppet_arm_left.position` 为 7-D 向量；末位是夹爪。和 NemoDiT
     `[left_arm(6), left_gripper(1), ...]` 字节对齐，不需重排。
   - **统一 14-D**：训练 / 推理都用 14 维，不管数据里某一侧是否在动。
     静止侧会被模型学成常量姿态，部署时同样发布到对应 `/master/joint_*`。
3. **归一化统计**
   - 推理时必须使用训练集相同的 `qpos_mean/std` & `action_mean/std`。
     checkpoint 已内嵌，优先从 checkpoint 读；失败才回落到 `dataset_stats.pkl`。
4. **`num_episodes` 一致性**
   - `train_agilex.py` 用数字 id `list(range(N))` 选择 episode，避免
     `episode_10` 字典序小于 `episode_2` 带来的混乱。
5. **`publish_rate` vs `n_action_steps`**
   - 每次 `predict()` 得到 `n_action_steps` 帧，按 `publish_rate` 逐帧发布。
     过大的 `n_action_steps` + 过低的 `publish_rate` 会让闭环变慢；建议
     `n_action_steps` 匹配控制器 200~400ms 的 horizon。
