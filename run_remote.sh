#!/usr/bin/env bash
# run_h100.sh — one script, run it every time (first setup or later restarts).
# Setup steps are skipped if already done, so reruns are fast and don't
# re-download checkpoints or re-clone the repo.
set -euo pipefail

REPO_DIR="$(dirname "$(readlink -f "$0")")/editAnything"

# --- clone (skip if already there) ---
if [ ! -d "$REPO_DIR" ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  git clone --recurse-submodules https://github.com/yining-li115/editAnything.git "$REPO_DIR"
else
  echo "[run_h100] $REPO_DIR already exists, skipping clone"
fi
cd "$REPO_DIR"

# --- conda env (skip create if it exists) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^editanything '; then
  conda create -n editanything python=3.10 -y
else
  echo "[run_h100] conda env 'editanything' already exists, skipping create"
fi
conda activate editanything

# --- VideoPainter diffusers fork (skip if already installed) ---
if ! python -c "import diffusers; print(diffusers.__file__)" 2>/dev/null | grep -q "VideoPainter"; then
  cd submodules/VideoPainter
  pip install -r requirements.txt
  pip install -e ./diffusers
  cd ../..
else
  echo "[run_h100] VideoPainter diffusers fork already installed, skipping"
fi

# --- fastmcp ---
python -c "import fastmcp" 2>/dev/null || pip install fastmcp

# --- checkpoints (skip if already downloaded) ---
if [ ! -d ckpt/CogVideoX-5b-I2V ] || [ ! -d ckpt/VideoPainter ]; then
  pip install "huggingface_hub==0.24.1"
  echo "[run_h100] huggingface-cli login required if not already logged in:"
  huggingface-cli login
  huggingface-cli download TencentARC/VideoPainter --local-dir ckpt
  huggingface-cli download THUDM/CogVideoX-5b-I2V  --local-dir ckpt/CogVideoX-5b-I2V
else
  echo "[run_h100] checkpoints already present, skipping download"
fi

echo "[run_h100] ckpt/ layout:"
find ckpt -maxdepth 3 -type d

# videopainter_mcp_server.py comes from the repo clone itself now (committed
# upstream), no separate copy step needed.

# --- launch ---
if [ ! -f videopainter_mcp_server.py ]; then
  echo "[run_h100] ERROR: videopainter_mcp_server.py not found in $REPO_DIR after clone." >&2
  echo "[run_h100] Confirm it's committed and pushed to the editAnything repo." >&2
  exit 1
fi
echo "[run_h100] starting videopainter_mcp_server.py on 127.0.0.1:8100"
python videopainter_mcp_server.py --http --host 127.0.0.1 --port 8100