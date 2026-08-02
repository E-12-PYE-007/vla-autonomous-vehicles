#!/usr/bin/env bash
set -euo pipefail

# First-Time Setup — AsyncVLA
#
# Prerequisite: AsyncVLA repo already cloned at $BASE_DIR/AsyncVLA
#
# BASE_DIR defaults to ~/capstone/code/asyncvla. Override it for machines that keep the
# models elsewhere — on rcp the home filesystem is too small, so use /vla_storage:
#   ASYNCVLA_BASE_DIR=/vla_storage/capstone/code/asyncvla bash tools/setup_asyncvla_desktop.bash

CONDA_DIR="$HOME/miniconda3"
BASE_DIR="${ASYNCVLA_BASE_DIR:-$HOME/capstone/code/asyncvla}"
ASYNCVLA_DIR="$BASE_DIR/AsyncVLA"
RELEASE_DIR="$ASYNCVLA_DIR/AsyncVLA_release"

echo "Installing AsyncVLA under: $BASE_DIR"


# --- Step 1: Initialise git submodules ---
echo "[1/6] Initialising git submodules..."
cd "$ASYNCVLA_DIR"
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


# --- Step 4: Create asyncvla conda environment ---
echo "[4/6] Creating asyncvla conda environment..."
if conda env list | grep -q "^asyncvla "; then
    echo "  Environment already exists — skipping."
else
    conda create -n asyncvla python=3.10 -y
fi


# --- Step 5: Install Python dependencies ---
echo "[5/6] Installing Python dependencies..."
conda activate asyncvla

pip install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# cv_bridge needs cv2. Pin below 5 — newer opencv requires numpy>=2, which breaks
# AsyncVLA's numpy==1.26.4 pin.
pip install opencv-python-headless==4.11.0.86

cd "$ASYNCVLA_DIR"
pip install -e .

cd "$ASYNCVLA_DIR/visualnav-transformer/train"
pip install -e .


# --- Step 6: Download model weights ---
echo "[6/6] Downloading model weights from HuggingFace..."
if [[ -d "$RELEASE_DIR" ]] && [[ -n "$(ls -A "$RELEASE_DIR" 2>/dev/null)" ]]; then
    echo "  Weights already present — skipping."
else
    pip install huggingface_hub
    cd "$ASYNCVLA_DIR"
    huggingface-cli download NHirose/AsyncVLA_release --local-dir ./AsyncVLA_release
fi

# The download is resumable — if it was interrupted, re-run this script rather than
# deleting the directory, or the partial blobs are discarded and it starts over.
if find "$RELEASE_DIR" -name '*.incomplete' 2>/dev/null | grep -q .; then
    echo ""
    echo "  WARNING: incomplete downloads remain in $RELEASE_DIR."
    echo "  Re-run this script to resume. A truncated download shows up later as"
    echo "  'KeyError: OpenVLAConfig' when the nodes start."
fi


# --- Done ---
echo ""
echo "Setup complete. To run AsyncVLA inference nodes:"
echo "  conda activate asyncvla"
echo "  source /opt/ros/humble/setup.bash"
echo "  source ~/capstone/vla-autonomous-vehicles/install/setup.bash"
echo "  ros2 launch async_vla asc_async_rcp.launch.py   # or asc_async_dsk.launch.py"
echo ""
echo "Model paths are set by the launch file — update the path constants at the top of"
echo "the launch file if you installed somewhere other than $BASE_DIR."
