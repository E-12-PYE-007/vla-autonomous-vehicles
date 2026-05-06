#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/cap_ros2_jazzy_ws/vla-autonomous-vehicles}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASYNCVLA_SOURCE="${ASYNCVLA_SOURCE:-$BUNDLE_DIR/external_runtime/AsyncVLA}"
DATASET_CONFIG="${DATASET_CONFIG:-$ASYNCVLA_SOURCE/config_nav/dataset_config.yaml}"
CHECKPOINT="${CHECKPOINT:-$BUNDLE_DIR/external_runtime/AsyncVLA_release/shead--750000_checkpoint.pt}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
DTYPE="${DTYPE:-bfloat16}"

source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"

CHECKPOINT_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
else
  echo "[warn] Checkpoint not found: $CHECKPOINT"
  echo "[warn] Benchmarking randomly initialized small-head weights."
fi

ros2 run asclinic_vla benchmark_action_head -- \
  --asyncvla-source "$ASYNCVLA_SOURCE" \
  --dataset-config "$DATASET_CONFIG" \
  "${CHECKPOINT_ARGS[@]}" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
