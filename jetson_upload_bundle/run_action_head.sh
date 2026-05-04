#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_action_head.sh CLOUD_IP"
  echo "Example: bash run_action_head.sh 203.0.113.10"
  exit 1
fi

CLOUD_IP="$1"
WORKSPACE="${WORKSPACE:-$HOME/cap_ros2_jazzy_ws/vla-autonomous-vehicles}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASYNCVLA_SOURCE="$BUNDLE_DIR/external_runtime/AsyncVLA"
VLA_RELEASE="$BUNDLE_DIR/external_runtime/AsyncVLA_release"
DATASET_CONFIG="$ASYNCVLA_SOURCE/config_nav/dataset_config.yaml"
CHECKPOINT="$VLA_RELEASE/shead--750000_checkpoint.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[error] Missing checkpoint: $CHECKPOINT"
  echo "Put shead--750000_checkpoint.pt into external_runtime/AsyncVLA_release/ first."
  exit 1
fi

source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"

ros2 run asclinic_vla split_action_head -- \
  --connect "tcp/$CLOUD_IP:7447" \
  --asyncvla-source "$ASYNCVLA_SOURCE" \
  --vla-path "$VLA_RELEASE" \
  --dataset-config "$DATASET_CONFIG"
