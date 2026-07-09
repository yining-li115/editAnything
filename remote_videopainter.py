"""remote_videopainter.py — client-side wrapper around the H100's
videopainter_generate MCP tool.

Two modes, controlled by env var VP_SHARED_FS (flip this one variable when
you move from your no-shared-storage test box to the SLURM H100):

  VP_SHARED_FS=0 (default — your current test setup, no shared mount)
      rsync's frames_dir, mask_dir, and the anchor images up to the H100
      over ssh before calling the tool, then rsyncs gen_dir back after.

  VP_SHARED_FS=1 (SLURM demo — H100 node shares the cluster mount)
      no copying; paths are valid on both machines already, so they're
      passed straight through, same as a normal in-process call.

IMPORTANT — checkpoint paths are NOT taken from the caller. model_path /
branch / id_lora sent by the orchestrator (filled in by Gemini, per
contracts/tools.py's schema) describe checkpoint locations on the machine
Gemini thinks it's running on — meaningless on the H100 unless VP_SHARED_FS=1
AND the two machines have the identical repo layout. This wrapper always
overrides them with H100_MODEL_PATH/H100_BRANCH/H100_ID_LORA below, which
should match what setup_h100_videopainter_only.sh produced (relative to the
H100's editAnything/ checkout, or wherever videopainter_mcp_server.py's cwd
is when it runs).

Requires: ssh key auth to H100_HOST already working (no password prompt),
rsync installed locally, and videopainter_mcp_server.py already running on
the H100 with the SSE tunnel from tunnel_to_h100.sh (or direct network path
on the SLURM cluster) up.
"""
from __future__ import annotations
import os
import subprocess
import uuid
from fastmcp import Client

H100_HOST = os.environ["H100_HOST"]  # e.g. user@node21 — required, no default on purpose
H100_PORT = os.environ.get("H100_PORT", "22")
H100_REMOTE_ROOT = os.environ.get("H100_REMOTE_ROOT", "/workspace/vp_jobs")
VP_SHARED_FS = os.environ.get("VP_SHARED_FS", "0") == "1"
VP_SERVER_URL = os.environ.get("VP_SERVER_URL", "http://localhost:8100/sse")

H100_MODEL_PATH = os.environ.get("H100_MODEL_PATH", "ckpt/CogVideoX-5b-I2V")
H100_BRANCH = os.environ.get("H100_BRANCH", "ckpt/VideoPainter/checkpoints/branch")
H100_ID_LORA = os.environ.get("H100_ID_LORA", "ckpt/VideoPainterID/checkpoints")


def _rsync_dir(local_dir: str, remote_dir: str, to_remote: bool):
    ssh_cmd = f"ssh -p {H100_PORT}"  
    if to_remote:
        subprocess.run(["ssh", "-p", H100_PORT, H100_HOST, "mkdir", "-p", remote_dir], check=True)
        src, dst = f"{local_dir.rstrip('/')}/", f"{H100_HOST}:{remote_dir.rstrip('/')}/"
    else:
        os.makedirs(local_dir, exist_ok=True)
        src, dst = f"{H100_HOST}:{remote_dir.rstrip('/')}/", f"{local_dir.rstrip('/')}/"
    subprocess.run(["rsync", "-az", "-e", ssh_cmd, src, dst], check=True)


async def videopainter_generate_remote(
    frames_dir: str,
    mask_dir: str,
    anchor_map: dict,
    gen_dir: str,
    segment_starts: list,
    prompt: str,
    model_path: str = "",   # accepted for call-signature parity, ignored — see module docstring
    branch: str = "",       # ignored
    id_lora: str = "",      # ignored
    dilate: int = 12,
    steps: int = 50,
    guidance: float = 6.0,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    job_id = uuid.uuid4().hex[:8]
    remote_base = f"{H100_REMOTE_ROOT}/{job_id}"

    if VP_SHARED_FS:
        r_frames, r_mask, r_gen, r_anchor_map = frames_dir, mask_dir, gen_dir, anchor_map
    else:
        r_frames = f"{remote_base}/frames"
        r_mask = f"{remote_base}/mask"
        r_gen = f"{remote_base}/gen"
        try:
            _rsync_dir(frames_dir, r_frames, to_remote=True)
            _rsync_dir(mask_dir, r_mask, to_remote=True)
        except subprocess.CalledProcessError as e:
            return {"error": f"rsync input dirs to H100 failed: {e}"}

        anchor_dirs = {os.path.dirname(p) for p in anchor_map.values()}
        if len(anchor_dirs) != 1:
            return {"error": f"anchor_map spans {len(anchor_dirs)} dirs, expected 1 for a clean rsync: {anchor_dirs}"}
        local_anchor_dir = anchor_dirs.pop()
        r_anchor_dir = f"{remote_base}/anchors"
        try:
            _rsync_dir(local_anchor_dir, r_anchor_dir, to_remote=True)
        except subprocess.CalledProcessError as e:
            return {"error": f"rsync anchors to H100 failed: {e}"}
        r_anchor_map = {k: os.path.join(r_anchor_dir, os.path.basename(p)) for k, p in anchor_map.items()}

    try:
        async with Client(VP_SERVER_URL) as client:
            result = await client.call_tool("videopainter_generate", dict(
                frames_dir=r_frames, mask_dir=r_mask, anchor_map=r_anchor_map,
                gen_dir=r_gen, segment_starts=segment_starts, prompt=prompt,
                model_path=H100_MODEL_PATH, branch=H100_BRANCH, id_lora=H100_ID_LORA,
                dilate=dilate, steps=steps, guidance=guidance, seed=seed, resume=resume,
            ))
    except Exception as e:
        return {"error": f"videopainter_generate (remote call) raised: {e}"}

    data = result.data if getattr(result, "data", None) is not None else {"error": "empty result from H100 server"}
    if "error" in data:
        return data

    if not VP_SHARED_FS:
        try:
            _rsync_dir(gen_dir, r_gen, to_remote=False)
        except subprocess.CalledProcessError as e:
            return {"error": f"rsync gen_dir back from H100 failed: {e}", "remote_gen_dir": r_gen}
        data["gen_frames_dir"] = os.path.join(gen_dir, "frames")

    return data