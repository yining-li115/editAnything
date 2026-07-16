#!/bin/bash
# Runs discord_bot.py as a SLURM job. Unlike a normal pipeline-stage sbatch,
# this process never exits on its own (client.run(TOKEN) blocks forever
# listening for Discord messages) — SLURM kills it the instant --time
# expires, mid-job or not. Bump --time for the demo; for routine testing
# the default below is fine, just resubmit when it dies.
#
# Usage: sbatch run_sbatch.sbatch
#
#SBATCH --job-name=videoediting-bot
#SBATCH --partition=NORMAL
#SBATCH --qos=practical_course
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --constraint="GPU_CC:8.6|GPU_CC:8.9|GPU_CC:9.0"
#SBATCH --exclude=node22
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/storage/slurm/%u/logs/slurm-%j.out

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
mkdir -p "/storage/slurm/${USER}/logs"

export HF_HOME="/storage/slurm/${USER}/hf_cache"
export TORCH_HOME="/storage/slurm/${USER}/hf_cache"
export HF_TOKEN=$(cat "/storage/slurm/${USER}/hf_cache/token")
export RIFE_BIN="$HOME/tools/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VP_OFFLOAD=sequential

# --- H100 (videopainter_generate) ---
# REPLACE_ME if this differs from what you've been using interactively.
export H100_HOST="root@45.135.56.10"
export H100_PORT="20221"
export VP_SHARED_FS=0   # 0 for testing, 1 once this runs from a node that
                         # shares the cluster mount with the H100 (demo day)

# --- ROSE / five-bench ---
export ROSE_PYTHON="$(conda info --base)/envs/rose/bin/python"
export FIVE_BENCH_PYTHON="$(conda info --base)/envs/five-bench/bin/python"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate editanything
cd "$REPO_ROOT"
pwd; hostname; date; nvidia-smi || true

# Tunnel to the H100 in the background. UNVERIFIED: compute nodes on this
# cluster may not have the same outbound network access / SSH agent as your
# interactive login node — if this hangs or the tunnel silently fails,
# check that ssh from a compute node (not the login node) to H100_HOST
# actually works, and that ~/.ssh keys are readable from compute nodes
# (should be, if /storage/slurm or $HOME is a shared network mount, but
# confirm rather than assume).
ssh -p "$H100_PORT" -N -L 8100:127.0.0.1:8100 "$H100_HOST" &
TUNNEL_PID=$!
trap 'kill $TUNNEL_PID 2>/dev/null || true' EXIT

sleep 3   # give the tunnel a moment before the bot's first tool call
echo "==== Starting discord_bot.py (tunnel pid $TUNNEL_PID) ===="
python discord_bot.py
