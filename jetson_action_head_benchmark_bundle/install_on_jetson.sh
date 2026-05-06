#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/cap_ros2_jazzy_ws/vla-autonomous-vehicles}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[install] Bundle: $BUNDLE_DIR"
echo "[install] Workspace: $WORKSPACE"

mkdir -p "$WORKSPACE/src"

echo "[install] Copying ROS packages into workspace src/"
rm -rf "$WORKSPACE/src/asclinic_vla" "$WORKSPACE/src/asclinic_vla_interfaces"
cp -a "$BUNDLE_DIR/ros2_src/asclinic_vla" "$WORKSPACE/src/"
cp -a "$BUNDLE_DIR/ros2_src/asclinic_vla_interfaces" "$WORKSPACE/src/"

echo "[install] Building ROS packages"
cd "$WORKSPACE"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select asclinic_vla_interfaces asclinic_vla

echo "[install] Done. Source with:"
echo "source $WORKSPACE/install/setup.bash"
