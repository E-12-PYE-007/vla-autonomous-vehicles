#!/usr/bin/env bash
set -euo pipefail

# Edge Setup — AsyncVLA sys1 (small head only)
#
# For robots or machines that run sys1 only. Installs the conda env and the
# prismatic Python package but skips the 15 GB model weights — sys1 only needs
# the shead checkpoint, which lives in the AsyncVLA_release dir you already have.
#
# Prerequisite: AsyncVLA repo already cloned at $BASE_DIR/AsyncVLA
#
# BASE_DIR defaults to ~/asyncvla-test. Override if your layout differs:
#   ASYNCVLA_BASE_DIR=/some/other/path bash tools/setup_asyncvla_edge.bash

CONDA_DIR="$HOME/miniconda3"
BASE_DIR="${ASYNCVLA_BASE_DIR:-$HOME/asyncvla-test}"
ASYNCVLA_DIR="$BASE_DIR/AsyncVLA"

echo "Setting up AsyncVLA edge (sys1) under: $BASE_DIR"


# --- Step 1: Initialise git submodules ---
echo "[1/5] Initialising git submodules..."
cd "$ASYNCVLA_DIR"
git submodule update --init --recursive


# --- Step 2: Install Miniconda ---
echo "[2/5] Installing Miniconda..."
if [[ -d "$CONDA_DIR" ]] || command -v conda &>/dev/null; then
    echo "  Conda already installed — skipping."
else
    wget -q --show-progress "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" \
        -O "$HOME/Miniconda3-latest-Linux-aarch64.sh"
    chmod +x "$HOME/Miniconda3-latest-Linux-aarch64.sh"
    bash "$HOME/Miniconda3-latest-Linux-aarch64.sh" -b -p "$CONDA_DIR"
    rm "$HOME/Miniconda3-latest-Linux-aarch64.sh"
    "$CONDA_DIR/bin/conda" init bash
fi


# --- Step 3: Accept conda Terms of Service ---
echo "[3/5] Accepting conda Terms of Service..."
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# --- Step 4: Create asyncvla conda environment ---
echo "[4/5] Creating asyncvla conda environment..."
if conda env list | grep -q "^asyncvla "; then
    echo "  Environment already exists — skipping."
else
    conda create -n asyncvla python=3.10 -y
fi


# --- Step 5: Install Python dependencies ---
echo "[5/5] Installing Python dependencies..."
conda activate asyncvla

pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

pip install opencv-python-headless==4.11.0.86

cd "$ASYNCVLA_DIR"
pip install -e .

cd "$ASYNCVLA_DIR/visualnav-transformer/train"
pip install -e .


# --- Done ---
echo ""
echo "Setup complete. To run sys1 on this machine:"
echo "  conda activate asyncvla"
echo "  source /opt/ros/humble/setup.bash"
echo "  source <path-to-workspace>/install/setup.bash"
echo "  ros2 launch async_vla asc_with_sys1.launch.py"
