# NemoDiT 适配松灵 AgileX / Mobile-Aloha 真机

本目录下除了原 RoboTwin 管线之外，新增了针对松灵 AgileX 双臂机器人 (基于
cobot_magic 采集栈 + `agx_robot` ACT pipeline) 的三个文件：

| 文件 | 作用 |
|------|------|
| `dataloader_agilex.py` | 读取 `collect_data.py` 保存的扁平 14-D AgileX HDF5 |
| `train_agilex.py` / `train_agilex.sh` | 在 AgileX 数据上训练 Flow-Matching DiT |
| `inference_agilex.py` | 真机 ROS 推理 (camera → 模型 → puppet arm cmd) |

原 `dataloader.py` / `train.py` / `eval.py` / `deploy_policy.py` 仍保留用于
RoboTwin 仿真。

## 1. 数据格式差异

`agx_robot/collect_data/collect_data.py` 保存的数据与 RoboTwin 版不同：

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
├── /action        (T, 14)     master arm 命令 (主臂示教)
└── /base_action   (T, 2)      [linear.x, angular.z]
```

关键点：

1. **14-D 布局恰好等价于 NemoDiT 的 `joint` 模式**
   (`[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`)，无需重排。
2. **采用 `/action` 作为目标，`/observations/qpos` 作为 state**。RoboTwin 里
   `state = action[0]` 依赖 master/slave 一致；真机中两者会有滞后，因此
   `inference_agilex.py` 和 `dataloader_agilex.py` 均使用当前帧 qpos。
3. **3 路相机**：`cam_high, cam_left_wrist, cam_right_wrist`（不再是 4 路）。
4. **可选 `compress`** —— dataloader 会自动 `cv2.imdecode` 并 BGR→RGB。
5. **可选机载底盘 (`/base_action`)** —— 训练 / 推理打开 `--use_robot_base` 后
   qpos / action 维度变为 16。

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
cd nemodit/RoboTwin/policy/NemoDiT

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
| `--future_action_window` | state+预测总帧数；实际预测 `window-1` 帧 | `13` |
| `--n_obs_steps` | 视觉历史帧数 | `2` |
| `--n_action_steps` | 推理时一次执行的帧数 (receding horizon) | `8` |
| `--temporal_agg` | 多帧特征聚合 (`last/mean/concat`) | `concat` |
| `--use_robot_base` | 把 `/base_action` 拼接到 qpos/action | `False` |
| `--arm_delay_time` | 前移 action 目标帧数 (同 ACT dataloader) | `0` |
| `--use_amp` | AMP fp16 | `False` |
| `--use_ema` | EMA 权重 | `False` |

> **约束**：`n_obs_steps + n_action_steps <= future_action_window`。

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

### 3.2 控制循环

1. ROS callbacks 缓冲相机帧 + 从臂 JointState。
2. `get_frame()` 取最新同步帧；缺失则 `rate.sleep()`。
3. 观测进入长度 `n_obs_steps` 的 deque (首轮复制填充)。
4. `qpos` 按训练时的 mean/std 归一化，作为 DiT `state` token。
5. `ActionModel.sample(num_steps, ode_solver, cfg_scale)` 生成
   `(1, n_action_steps, action_dim)` 动作；反归一化后入队。
6. 每个 tick 从队列弹出一帧，split 成 `left[:7] / right[7:14]`
   (若开启 base 再解包 `action[14:16]`)，发布到 `/master/joint_*`。

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

### 真机 (AgileX)
- ROS Noetic (aloha 环境见 `agx_robot/aloha.yml`)
- `rospy, cv_bridge, sensor_msgs, geometry_msgs, nav_msgs, std_msgs`
- `puppet_arm_publish_*` 依赖 (见 `agx_robot/follow_control/`)

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
