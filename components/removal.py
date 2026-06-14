"""ROSE removal — video object removal WITH side effects (shadow / reflection / light).

ROSE (Kunbyte-AI/ROSE, base Wan2.1-Fun-1.3B-InP) removes the MASKED object *and* the
side effects it casts on the scene -> a clean plate. This is the stage that finally
kills the source object's leftover shadow.

ROSE needs its OWN env (python 3.12, torch 2.6/cu124, stock diffusers) — incompatible
with editanything (torch 2.4 + VideoPainter's diffusers fork) — so we run it as a
SUBPROCESS over files (cross-env, decoupled candidate; later its own MCP tool):

  frames + per-frame object mask  ->  input.mp4 + mask.mp4
     ->  (rose env) submodules/ROSE/inference.py  ->  results/example-1.mp4 (clean)
     ->  per-frame clean plate.

Mask convention (ROSE utils.get_video_and_mask): white (>=240) = remove. Our SAM3
source mask (white object on black) is already correct — no inversion. ROSE resizes
to 480x720 internally and requires video_length = 16n+1.
"""
import os
import glob
import shutil
import tempfile
import subprocess

from contracts import layout

ROSE_ROOT = os.path.join(layout.ROOT, "submodules", "ROSE")
ROSE_PYTHON = os.environ.get("ROSE_PYTHON", "/venv/rose/bin/python")


def _frame_paths(d):
    return sorted(glob.glob(f"{d}/frame_*.png"))


def largest_16np1(n):
    """Largest L = 16k+1 with L <= n (ROSE's required clip length)."""
    if n < 17:
        raise ValueError(f"ROSE needs >=17 frames (length must be 16n+1); got {n}")
    return 16 * ((n - 1) // 16) + 1


def _encode(frame_paths, out_mp4, *, fps=25, lossless=False):
    """Encode ordered png frames -> mp4. Symlinks to a contiguous temp sequence so
    ffmpeg's %d pattern works regardless of the source filenames. Mask videos use
    near-lossless so white stays >=240 (ROSE's remove threshold)."""
    tmp = tempfile.mkdtemp()
    try:
        for i, p in enumerate(frame_paths, 1):
            os.symlink(os.path.abspath(p), os.path.join(tmp, f"f_{i:05d}.png"))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", f"{tmp}/f_%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        cmd += (["-qp", "0"] if lossless else ["-crf", "12"])
        cmd.append(out_mp4)
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


def remove(frames_dir, mask_dir, out_dir, *, video_length=None, prompt=""):
    """Run ROSE -> clean-plate frames (source object + shadow removed).

    frames_dir : source frames frame_*.png
    mask_dir   : per-frame object masks (white = remove; e.g. SAM3 source mask)
    out_dir    : clean-plate frames written to {out_dir}/frames/frame_%05d.png
    video_length: 16n+1; default = largest 16n+1 <= #frames
    Returns the clean-plate frames dir.
    """
    frames = _frame_paths(frames_dir)
    masks = _frame_paths(mask_dir)
    n = min(len(frames), len(masks))
    if n == 0:
        raise FileNotFoundError(f"no frames/masks in {frames_dir} / {mask_dir}")
    L = video_length or largest_16np1(n)
    work = os.path.join(out_dir, "_rose_work")
    os.makedirs(work, exist_ok=True)
    in_mp4 = _encode(frames[:L], os.path.join(work, "input.mp4"))
    mask_mp4 = _encode(masks[:L], os.path.join(work, "mask.mp4"), lossless=True)
    res_dir = os.path.join(work, "results")
    cmd = [ROSE_PYTHON, "inference.py",
           "--validation_videos", os.path.abspath(in_mp4),
           "--validation_masks", os.path.abspath(mask_mp4),
           "--validation_prompts", prompt,
           "--output_dir", os.path.abspath(res_dir),
           "--video_length", str(L),
           "--sample_size", "480", "720"]
    print(f"[removal] ROSE on {L} frames (16n+1) — cwd={ROSE_ROOT}")
    subprocess.run(cmd, check=True, cwd=ROSE_ROOT)
    clean_mp4 = os.path.join(res_dir, "example-1.mp4")
    if not os.path.exists(clean_mp4):
        raise FileNotFoundError(f"ROSE produced no output at {clean_mp4}")
    frames_out = os.path.join(out_dir, "frames")
    os.makedirs(frames_out, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clean_mp4,
                    "-start_number", "1", f"{frames_out}/frame_%05d.png"], check=True)
    print(f"[removal] clean plate -> {frames_out} ({len(_frame_paths(frames_out))} frames)")
    return frames_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ROSE video object removal (clean plate)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--mask_dir", required=True, help="per-frame object masks (white=remove)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--video_length", type=int, default=None, help="16n+1; default auto")
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()
    remove(args.frames_dir, args.mask_dir, args.out_dir,
           video_length=args.video_length, prompt=args.prompt)
