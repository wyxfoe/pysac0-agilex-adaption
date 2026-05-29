#!/bin/bash
# ============================================================================
# NemoDiT real-robot deployment on AgileX / Songling (ROS Noetic / ROS1)
# ============================================================================
#
# Usage:
#   bash deploy_agilex.sh <checkpoint_path> [publish_rate] [extra_args...]
#
# Example:
#   bash deploy_agilex.sh checkpoints/agilex-100/best.pt 40
#   bash deploy_agilex.sh checkpoints/agilex-100/best.pt 40 --no_soft_start
#
# Prerequisites:
#   * ROS Noetic running, camera + puppet arm topics publishing.
#   * A conda env (or system python) where `rospy` AND `torch` both import.
#     See AGILEX_README.md §5 for environment setup hints.
#   * Checkpoint produced by train_agilex.py (embeds args + norm_stats).
#
# Safety:
#   * Defaults perform a soft-start ramp to a home pose before policy execution.
#   * --soft_start_pause additionally waits for an Enter keypress (recommended
#     for first runs on a new task).

set -euo pipefail

ckpt=${1:?"checkpoint path required"}
rate=${2:-30}    # default matches cobot_magic/collect_data.py capture rate
shift 2 || true

# Source ROS env if not already loaded.
if [[ -z "${ROS_DISTRO:-}" ]]; then
    if [[ -f /opt/ros/noetic/setup.bash ]]; then
        # shellcheck source=/dev/null
        source /opt/ros/noetic/setup.bash
    fi
fi

# Source workspace if present (provides puppet_arm_publisher etc.).
if [[ -f "$HOME/catkin_ws/devel/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/catkin_ws/devel/setup.bash"
fi

echo "============================================"
echo "[NemoDiT/AgileX] ROS_DISTRO  : ${ROS_DISTRO:-not set}"
echo "[NemoDiT/AgileX] checkpoint  : ${ckpt}"
echo "[NemoDiT/AgileX] publish_rate: ${rate} Hz"
echo "[NemoDiT/AgileX] extra args  : $*"
echo "============================================"

python inference_agilex.py \
    --checkpoint "${ckpt}" \
    --publish_rate "${rate}" \
    --soft_start_pause \
    "$@"
