#!/usr/bin/env bash
set -euo pipefail

# First-Time Setup — TIC-VLA
#
# Prerequisite: TIC-VLA repo already cloned at $BASE_DIR/TIC-VLA
#
# BASE_DIR defaults to ~/capstone/code/ticvla. Override it for machines that keep the
# models elsewhere — on rcp the home filesystem is too small, so use /vla_storage:
#   TICVLA_BASE_DIR=/vla_storage/capstone/code/ticvla bash tools/setup_ticvla_desktop.bash

CONDA_DIR="$HOME/miniconda3"
BASE_DIR="${TICVLA_BASE_DIR:-$HOME/capstone/code/ticvla}"
TICVLA_DIR="$BASE_DIR/TIC-VLA"
MODEL_DIR="$BASE_DIR/InternVL3-1B"
CHECKPOINT_DIR="$BASE_DIR/TIC-VLA-model.ckpt"

echo "Installing TIC-VLA under: $BASE_DIR"


# --- Step 1: Initialise git submodules ---
echo "[1/6] Initialising git submodules..."
cd "$TICVLA_DIR"
git submodule update --init --recursive


# --- Step 2: Install Miniconda ---
echo "[2/6] Installing Miniconda..."
if [[ -d "$CONDA_DIR" ]] || command -v conda &>/dev/null; then
    echo "  Conda already installed — skipping."
else
    wget -q --show-progress "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O "$HOME/Miniconda3-latest-Linux-x86_64.sh"
    chmod +x "$HOME/Miniconda3-latest-Linux-x86_64.sh"
    bash "$HOME/Miniconda3-latest-Linux-x86_64.sh" -b -p "$CONDA_DIR"
    rm "$HOME/Miniconda3-latest-Linux-x86_64.sh"
    "$CONDA_DIR/bin/conda" init bash
fi


# --- Step 3: Accept conda Terms of Service ---
echo "[3/6] Accepting conda Terms of Service..."
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# --- Step 4: Create ticvla conda environment ---
# NOTE: upstream tic-vla.yaml specifies Python 3.11, which cannot be used here — ROS 2
# Humble's rclpy C extension is built for Python 3.10 only, so `import rclpy` fails with
# "No module named 'rclpy._rclpy_pybind11'". Create the env at 3.10 and install the
# upstream requirements into it instead.
echo "[4/6] Creating tic-vla conda environment (Python 3.10)..."
if conda env list | grep -q "^tic-vla "; then
    echo "  Environment already exists — skipping."
else
    conda create -n tic-vla python=3.10 -y
fi


# --- Step 5: Install TIC-VLA package ---
echo "[5/6] Installing TIC-VLA Python package..."
conda activate tic-vla
cd "$TICVLA_DIR"
pip install -r requirements-train.txt
pip install -e .

# cv_bridge needs cv2. Pin below 5 — newer opencv requires numpy>=2, which breaks
# TIC-VLA's own "numpy>=1.26,<2.0" requirement.
pip install numpy==1.26.4 opencv-python-headless==4.11.0.86


# --- Step 6: Download model weights ---
echo "[6/6] Downloading model weights from HuggingFace..."
pip install huggingface_hub

# Both downloads are resumable — if interrupted, re-run this script rather than deleting
# the target directory, or the partial blobs are discarded and it starts over.
if [[ -d "$MODEL_DIR" ]] && [[ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
    echo "  InternVL3-1B base model already present — skipping."
else
    huggingface-cli download OpenGVLab/InternVL3-1B --local-dir "$MODEL_DIR"
fi

if [[ -f "$CHECKPOINT_DIR" ]]; then
    echo "  TIC-VLA checkpoint already present — skipping."
else
    # Note: the checkpoint lives in a *dataset* repo, not a model repo.
    huggingface-cli download handsomeYun/TIC-VLA --repo-type dataset \
        --include "TIC-VLA-model.ckpt" --local-dir "$BASE_DIR"
fi


# --- Done ---
echo ""
echo "Setup complete. To run TIC-VLA inference nodes:"
echo "  conda activate tic-vla"
echo "  source /opt/ros/humble/setup.bash"
echo "  source ~/capstone/vla-autonomous-vehicles/install/setup.bash"
echo "  ros2 launch tic_vla asc_tic_rcp.launch.py   # or asc_tic_dsk.launch.py"
echo ""
echo "Model paths are set by the launch file — update the path constants at the top of"
echo "the launch file if you installed somewhere other than $BASE_DIR."
