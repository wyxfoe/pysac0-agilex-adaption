# NemoDiT (AgileX real-robot fork)

Flow-Matching Diffusion Transformer 适配松灵 AgileX / Mobile-Aloha 真机的版本。
本目录只保留真机管线，原 RoboTwin 仿真已从仓库中移除。

详细使用方法见 **[AGILEX_README.md](AGILEX_README.md)**。

## 快速入口

```bash
# 训练 (在 NemoDiT/ 目录下)
python train_agilex.py --data_path /path/to/agilex/episodes --num_episodes 100 \
    --val_ratio 0.1 --checkpoint_dir checkpoints/agilex-100

# 真机推理 (需要 ROS Noetic + 主从臂节点 + 3 路相机就绪)
bash deploy_agilex.sh checkpoints/agilex-100/best.pt 30

# 数据审查 (训练前先看下数据是否健康)
python inspect_hdf5.py /path/to/agilex/episodes --num_episodes 3
```

## 模型概览

- **架构**: Flow-Matching DiT (Rectified Flow, 不是 DDPM/DDIM)
- **视觉**: ResNet50 / ViT-B 多帧多相机融合 → MLP adapter → DiT cross-attn
- **观测**: `qpos[t]` (14-D 双臂关节 + 夹爪) + `n_obs_steps` 帧图像
- **预测**: `qpos[t+1 .. t+12]` (12 帧未来关节位置)，receding-horizon 执行
- **采样**: midpoint ODE, 默认 10 步
- **训练目标**: MSE on velocity field `v = x_1 - noise`

## 上游参考

- DiT: Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
- Rectified Flow: Liu et al., "Flow Straight and Fast", ICLR 2023
- 真机管线: NVIDIA Isaac-GR00T, thu-ml/RDT-2, Mobile-Aloha ACT
