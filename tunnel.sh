#!/usr/bin/env bash
# Run this on the machine that runs discord_bot.py / orchestrator.py.
# Forwards local port 8100 to the H100's videopainter_mcp_server.py, which
# is bound to 127.0.0.1 there — this tunnel is the only way in.
set -euo pipefail
H100_HOST="${1:?usage: tunnel.sh user@h100-ip ssh-port}"
H100_PORT="${2:?usage: tunnel.sh user@h100-ip ssh-port}"
ssh -p "$H100_PORT" -N -L 8100:127.0.0.1:8100 "$H100_HOST"