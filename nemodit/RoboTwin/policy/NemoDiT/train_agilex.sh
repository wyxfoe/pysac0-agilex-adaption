#!/bin/bash
# ============================================================================
# NemoDiT training on AgileX / Songling real-robot HDF5 data
# ============================================================================
#
# Usage:
#   bash train_agilex.sh <task_name> <expert_data_num> <seed> <gpu_id> [data_root]
#
# Example:
#   bash train_agilex.sh pick_place 50 0 0 ~/data
#   # expects HDF5 episodes at ~/data/pick_place/episode_*.hdf5
#
# Notes:
#   - Assumes AgileX/Mobile-Aloha HDF5 layout produced by collect_data.py
#     (see agx_robot/collect_data/collect_data.py).
#   - Saves checkpoints + dataset_stats.pkl under checkpoints/<run_name>/.
#   - Default uses 3 wrist+top cameras (cam_high, cam_left_wrist, cam_right_wrist),
#     joint action, 14-D qpos/action.

set -euo pipefail

task_name=${1:?"task_name required"}
expert_data_num=${2:?"expert_data_num required"}
seed=${3:?"seed required"}
gpu_id=${4:?"gpu_id required"}
data_root=${5:-"$HOME/data"}

# ---------------- Training hyper-parameters ----------------
model_type="DiT-B"
vision_backbone="resnet50"
adapter_type="mlp"

n_obs_steps=2
n_action_steps=8
future_action_window=13
temporal_agg="concat"

num_cameras=3                # cam_high, cam_left_wrist, cam_right_wrist
use_robot_base=""            # set to "--use_robot_base" for mobile base tasks
arm_delay_time=0

epochs=500
batch_size=32
lr=1e-4
weight_decay=0.01
grad_clip=1.0
num_workers=4
save_every=50
num_inference_steps=10

# ---------------- WandB ----------------
use_wandb=""                 # set to "--use_wandb" to enable
wandb_project="nemodit_agilex"

# ---------------- Paths ----------------
data_path="${data_root}/${task_name}"
run_name="${task_name}-${expert_data_num}-seed${seed}"
checkpoint_dir="checkpoints/${run_name}"

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo "============================================"
echo "[NemoDiT/AgileX] Task       : ${task_name}"
echo "[NemoDiT/AgileX] Episodes   : ${expert_data_num}"
echo "[NemoDiT/AgileX] Seed       : ${seed}"
echo "[NemoDiT/AgileX] GPU        : ${gpu_id}"
echo "[NemoDiT/AgileX] Data path  : ${data_path}"
echo "[NemoDiT/AgileX] Checkpoint : ${checkpoint_dir}"
echo "============================================"

python train_agilex.py \
    --data_path "${data_path}" \
    --num_episodes "${expert_data_num}" \
    --num_cameras "${num_cameras}" \
    ${use_robot_base} \
    --arm_delay_time "${arm_delay_time}" \
    --model_type "${model_type}" \
    --vision_backbone "${vision_backbone}" \
    --vision_pretrained \
    --adapter_type "${adapter_type}" \
    --n_obs_steps "${n_obs_steps}" \
    --n_action_steps "${n_action_steps}" \
    --future_action_window "${future_action_window}" \
    --temporal_agg "${temporal_agg}" \
    --num_inference_steps "${num_inference_steps}" \
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --grad_clip "${grad_clip}" \
    --num_workers "${num_workers}" \
    --save_every "${save_every}" \
    --checkpoint_dir "${checkpoint_dir}" \
    ${use_wandb} \
    --wandb_project "${wandb_project}" \
    --device cuda:0

echo "Training finished. Checkpoints in ${checkpoint_dir}"
