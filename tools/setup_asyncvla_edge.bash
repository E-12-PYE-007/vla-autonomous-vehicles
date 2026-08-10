#!/usr/bin/env bash
set -euo pipefail

# Edge venv setup — AsyncVLA sys1 (aarch64 / Jetson Orin Nano)
#
# Creates a Python venv for running the sys1 ROS node on the Jetson.
#
# Uses --system-site-packages so the venv inherits JetPack's CUDA-enabled
# torch build. Do NOT pip install torch here — it would replace the JetPack
# wheel with a CPU-only aarch64 build.
#
# tensorflow and dlimp are intentionally omitted. sys1.py loads Edge_adapter
# via importlib (bypassing prismatic's package init) so the RLDS dataset
# pipeline — and its tensorflow dependency — is never imported.
#
# Prerequisites:
#   - JetPack installed (system torch with CUDA)
#   - AsyncVLA repo cloned at ASYNCVLA_DIR with submodules initialised
#
# Override defaults if your layout differs:
#   ASYNCVLA_DIR=/path/to/AsyncVLA VENV_DIR=~/venvs/my_venv bash tools/setup_asyncvla_edge.bash

ASYNCVLA_DIR="${ASYNCVLA_DIR:-$HOME/AsyncVLA}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/asclinic_vla_action_head}"


# --- Step 1: Init submodules ---
echo "[1/4] Initialising git submodules..."
cd "$ASYNCVLA_DIR"
git submodule update --init --recursive


# --- Step 2: Create venv ---
echo "[2/4] Creating venv at $VENV_DIR..."
if [[ -d "$VENV_DIR" ]]; then
    echo "  Already exists — skipping."
else
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel


# --- Step 3: Install Python inference dependencies ---
echo "[3/4] Installing inference dependencies..."
python -m pip install \
    "huggingface_hub[cli]" \
    pillow \
    "numpy==1.26.4" \
    efficientnet_pytorch \
    einops \
    "timm==0.9.10" \
    "transformers==4.40.1" \
    accelerate \
    sentencepiece \
    safetensors


# --- Step 4: Install vint_train ---
# small_head.py imports MultiLayerDecoder_trans from vint_train.
# --no-deps avoids vint_train's tensorflow-based training requirements.
echo "[4/4] Installing vint_train (--no-deps)..."
cd "$ASYNCVLA_DIR/visualnav-transformer/train"
pip install --no-deps -e .


echo ""
echo "Done. Verify CUDA is available through the venv:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -c \"import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())\""
