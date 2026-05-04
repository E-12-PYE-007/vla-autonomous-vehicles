#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_split_ros.sh CLOUD_IP [ROBOCLAW_DRY_RUN] [GOAL_MODE] [GOAL_VALUE]"
  echo "Example text:  bash run_split_ros.sh 203.0.113.10 true text 'go to the red cup'"
  echo "Example image: bash run_split_ros.sh 203.0.113.10 true image /home/jetson/goal.png"
  exit 1
fi

CLOUD_IP="$1"
ROBOCLAW_DRY_RUN="${2:-true}"
GOAL_MODE="${3:-text}"
GOAL_VALUE="${4:-}"
WORKSPACE="${WORKSPACE:-$HOME/cap_ros2_jazzy_ws/vla-autonomous-vehicles}"

source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"

if [[ "$GOAL_MODE" == "image" ]]; then
  ros2 launch asclinic_vla asclinic_vla_split.launch.py \
    zenoh_connect_endpoint:="tcp/$CLOUD_IP:7447" \
    use_goal_publisher:=true \
    goal_mode:=image \
    goal_image_path:="$GOAL_VALUE" \
    roboclaw_dry_run:="$ROBOCLAW_DRY_RUN"
else
  ros2 launch asclinic_vla asclinic_vla_split.launch.py \
    zenoh_connect_endpoint:="tcp/$CLOUD_IP:7447" \
    use_goal_publisher:=true \
    goal_mode:=text \
    goal_text:="$GOAL_VALUE" \
    roboclaw_dry_run:="$ROBOCLAW_DRY_RUN"
fi
