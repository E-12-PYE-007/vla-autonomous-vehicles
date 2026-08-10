#!/usr/bin/env bash
set -euo pipefail

# Edge Setup — AsyncVLA sys1 (aarch64 / Jetson)
#
# Installs the same packages as setup_asyncvla_desktop.bash with two differences:
#   1. Uses the aarch64 Miniconda installer.
#   2. Installs prismatic with --no-deps to skip tensorflow-graphics, which
#      declares tensorflow-addons as a dependency that has no aarch64 wheels.
#      All other packages from requirements.txt that are in the runtime import
#      chain are installed explicitly below.
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
echo "[1/6] Initialising git submodules..."
cd "$ASYNCVLA_DIR"
git submodule update --init --recursive


# --- Step 2: Install Miniconda ---
echo "[2/6] Installing Miniconda..."
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
echo "[3/6] Accepting conda Terms of Service..."
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# --- Step 4: Create asyncvla conda environment ---
echo "[4/6] Creating asyncvla conda environment..."
if conda env list | grep -q "^asyncvla "; then
    echo "  Environment already exists — skipping."
else
    conda create -n asyncvla python=3.10 -y
fi


# --- Step 5: Install dependencies ---
echo "[5/6] Installing dependencies..."
conda activate asyncvla

pip install \
    numpy==1.26.4 \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    pillow==12.2.0 \
    efficientnet_pytorch==0.7.1 \
    timm==0.9.10 \
    opencv-python-headless==4.11.0.86

pip install \
    huggingface_hub==0.29.1 \
    tokenizers==0.19.1 \
    sentencepiece==0.1.99 \
    accelerate==1.13.0 \
    draccus==0.8.0 \
    rich==15.0.0 \
    peft==0.11.1

# Pinned transformers fork required by AsyncVLA for bidirectional attention
pip install git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab

# TensorFlow stack — imported at module level in prismatic's dataset pipeline
pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3

# AsyncVLA dataset loading library
pip install git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46


# --- Step 6: Install prismatic and vint_train ---
# --no-deps skips tensorflow-graphics (and its tensorflow-addons dependency)
# which has no aarch64 wheels. All runtime deps are already installed above.
echo "[6/6] Installing prismatic and vint_train..."
cd "$ASYNCVLA_DIR"
pip install --no-deps -e .

cd "$ASYNCVLA_DIR/visualnav-transformer/train"
pip install --no-deps -e .


# --- Done ---
echo ""
echo "Setup complete. To run sys1 on this machine:"
echo "  conda activate asyncvla"
echo "  source /opt/ros/humble/setup.bash"
echo "  source <path-to-workspace>/install/setup.bash"
echo "  ros2 launch async_vla asc_with_sys1.launch.py"
