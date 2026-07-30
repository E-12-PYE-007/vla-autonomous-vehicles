#!/usr/bin/env bash
set -euo pipefail

# Desktop First-Time Setup — TIC-VLA
# Run from ~/capstone/code/ticvla on the desktop machine.
# Prerequisite: TIC-VLA repo already cloned at ~/capstone/code/ticvla/TIC-VLA

CONDA_DIR="$HOME/miniconda3"
BASE_DIR="$HOME/capstone/code/ticvla"
TICVLA_DIR="$BASE_DIR/TIC-VLA"
MODEL_DIR="$BASE_DIR/InternVL3-1B"
CHECKPOINT_DIR="$BASE_DIR/TIC-VLA-model.ckpt"


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
echo "[4/6] Creating ticvla conda environment from tic-vla.yaml..."
if conda env list | grep -q "^tic-vla "; then
    echo "  Environment already exists — skipping."
else
    cd "$TICVLA_DIR"
    conda env create -f tic-vla.yaml
fi


# --- Step 5: Install TIC-VLA package ---
echo "[5/6] Installing TIC-VLA Python package..."
conda activate tic-vla
cd "$TICVLA_DIR"
pip install -e .


# --- Step 6: Download model weights ---
echo "[6/6] Downloading model weights from HuggingFace..."
pip install huggingface_hub

if [[ -d "$MODEL_DIR" ]] && [[ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
    echo "  InternVL3-1B base model already present — skipping."
else
    python - <<'EOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="OpenGVLab/InternVL3-1B",
    repo_type="model",
    local_dir=os.path.expanduser("~/capstone/code/ticvla/InternVL3-1B"),
)
print("InternVL3-1B downloaded.")
EOF
fi

if [[ -f "$CHECKPOINT_DIR" ]]; then
    echo "  TIC-VLA checkpoint already present — skipping."
else
    python - <<'EOF'
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download(
    repo_id="handsomeYun/TIC-VLA",
    filename="TIC-VLA-model.ckpt",
    repo_type="dataset",
)
dest = os.path.expanduser("~/capstone/code/ticvla/TIC-VLA-model.ckpt")
shutil.copy(path, dest)
print(f"TIC-VLA checkpoint saved to {dest}")
EOF
fi


# --- Done ---
echo ""
echo "Setup complete. To run TIC-VLA inference nodes:"
echo "  conda activate tic-vla"
echo "  source /opt/ros/humble/setup.bash"
echo "  export PYTHONPATH=\$CONDA_PREFIX/lib/python3.10/site-packages:\$PYTHONPATH"
echo "  source ~/capstone/code/vla-autonomous-vehicles/install/setup.bash"
echo "  ros2 launch tic_vla asc_tic.launch.py"
