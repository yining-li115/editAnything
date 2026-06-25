#!/bin/bash
#SBATCH --job-name=mcp_smoke
#SBATCH --partition=PRACT
#SBATCH --qos=practical_course
#SBATCH --nodelist=node9
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/storage/slurm/s0037/logs/slurm-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate videopainter

export HF_HOME="/storage/slurm/s0037/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MCP_HOST=127.0.0.1
export MCP_PORT=8000

REPO=/usr/prakt/s0037/editAnything
cd "$REPO"

mkdir -p /storage/slurm/s0037/logs

SERVER_LOG="/storage/slurm/s0037/logs/mcp_server-${SLURM_JOB_ID}.out"
echo "[smoke] starting MCP server -> $SERVER_LOG"
python components/mcp/mcp_server.py > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[smoke] waiting for server ..."
for i in $(seq 1 60); do
    if python - "$MCP_HOST" "$MCP_PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)
PY
    then echo "[smoke] server up (${i}s)"; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[smoke] server died — see $SERVER_LOG"; exit 1; }
    sleep 1
done

python components/mcp/smoke_test.py
echo "[smoke] done. artifacts under $REPO/outputs/smoke/"