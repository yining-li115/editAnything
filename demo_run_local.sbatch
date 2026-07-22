#!/bin/bash
# demo_run_local.sbatch — DEMO launcher: whole pipeline on node21, fastest config.
#
# Same as run_local.sbatch but pinned to node21 and with every speed knob made
# explicit so the demo is reproducible: ROSE off, VP_OFFLOAD=none, low RoMa res,
# steps=10 (baked into the code defaults). Starts the VideoPainter MCP server
# locally and runs discord_bot.py against it — no external box, no tunnel.
#
# discord_bot.py blocks forever; SLURM kills it at --time. Resubmit when it dies.
#
# Usage:  sbatch demo_run_local.sbatch
#
#SBATCH --job-name=videoediting-demo
#SBATCH --partition=NORMAL
#SBATCH --account=practical_course
#SBATCH --qos=practical_course
#SBATCH --gres=gpu:1,VRAM:80G
#SBATCH --nodelist=node21              # DEMO: run ONLY on node21 (the 96GB H100 NVL)
#SBATCH --cpus-per-task=5
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/storage/slurm/%u/logs/demo-%j.out

set -euo pipefail

REPO_ROOT="/usr/prakt/s0033/editAnything"
mkdir -p "/storage/slurm/${USER}/logs"

# --- HF cache: SAM3 weights + gated token live HERE on this cluster ---
export HF_HOME="/usr/prakt/s0033/.cache/huggingface"
export TORCH_HOME="$HF_HOME"
export HF_TOKEN="$(cat "$HF_HOME/token")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_ROOT"

# --- cross-env subprocess tools ---
export ROSE_PYTHON="$(conda info --base)/envs/rose/bin/python"
export FIVE_BENCH_PYTHON="$(conda info --base)/envs/five-bench/bin/python"
export RIFE_BIN="$HOME/tools/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"

# --- DEMO speed config (explicit + reproducible) ---
export DISABLE_ROSE=1                 # skip ROSE (slow/optional) for the demo
export VP_OFFLOAD=none                # VideoPainter owns the card -> fastest generation
export ROMA_COARSE_RES=448            # faster RoMa matching (default already 448)
export ROMA_UPSAMPLE_RES=560          # (steps=10 is baked into the code defaults)

# --- VideoPainter runs LOCALLY on node21 ---
export VP_SHARED_FS=1
export H100_HOST=localhost
export VP_SERVER_URL=http://127.0.0.1:8100/sse

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate videopainter
cd "$REPO_ROOT"
pwd; hostname; date; nvidia-smi || true

# --- start the VideoPainter MCP server locally and wait until it's serving ---
VP_LOG="/storage/slurm/${USER}/logs/demo-vp-${SLURM_JOB_ID}.log"
VP_OFFLOAD=none python videopainter_mcp_server.py --http --host 127.0.0.1 --port 8100 > "$VP_LOG" 2>&1 &
VP_PID=$!
trap 'kill $VP_PID 2>/dev/null || true' EXIT
echo "==== VideoPainter server starting (pid $VP_PID) -> $VP_LOG ===="
for _ in $(seq 1 60); do
  grep -q "Uvicorn running" "$VP_LOG" 2>/dev/null && { echo "VP server up"; break; }
  sleep 2
done
grep -m1 "offload mode" "$VP_LOG" 2>/dev/null || true   # will print once VP loads on first generate

echo "==== Starting discord_bot.py (DEMO) ===="
python discord_bot.py
