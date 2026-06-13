"""Encode frames -> final video, with optional RIFE anchor de-spiking.

The pipeline works at 720x480 (squished landscape); the final video is unsquished
back to the original portrait resolution here.

Smoothing is NOT 2x frame interpolation. Multi-chunk reanchor inserts one clean
anchor frame at each segment boundary; because that anchor is warped, it shows up
as a single-frame "pop" (the frame differs hugely from both neighbours while the
two neighbours are nearly identical). The fix — matching the original pipeline —
is to REPLACE each boundary anchor frame n with RIFE(frame n-1, frame n+1), the
true motion-midpoint of its neighbours, and keep every other frame and the native
fps untouched. Only the handful of anchor frames change.

RIFE binary (prebuilt, includes models) — point RIFE_BIN at it:
    https://github.com/nihui/rife-ncnn-vulkan/releases
Default: tools/rife-ncnn-vulkan-20221029-ubuntu/ under the repo parent.
"""
import os
import shutil
import subprocess

RIFE_BIN = os.environ.get(
    "RIFE_BIN",
    "/root/project/tools/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan")
RIFE_MODEL = os.environ.get("RIFE_MODEL", "rife-v4.6")


def encode(frames_dir, out_path, size_wh, *, fps=25, crf=18, pattern="frame_%05d.png"):
    """Encode frames in frames_dir to out_path, scaled to size_wh=(W,H)."""
    w, h = size_wh
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", os.path.join(frames_dir, pattern),
        "-vf", f"scale={w}:{h}:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out_path,
    ], check=True)
    print(f"[encode] wrote {out_path} ({w}x{h} @ {fps}fps)")
    return out_path


def rife_mid(a_path, b_path, out_path):
    """RIFE the motion-midpoint of two frames (rife-ncnn-vulkan, GPU/Vulkan)."""
    if not os.path.exists(RIFE_BIN):
        raise FileNotFoundError(f"RIFE binary not found at {RIFE_BIN} (set RIFE_BIN)")
    model_path = os.path.join(os.path.dirname(RIFE_BIN), RIFE_MODEL)
    subprocess.run([RIFE_BIN, "-0", a_path, "-1", b_path, "-o", out_path, "-m", model_path],
                   check=True, stderr=subprocess.DEVNULL)
    return out_path


def despike_anchors(frames_dir, out_dir, anchor_frames, *, pattern="frame_{:05d}.png"):
    """Copy all frames; replace each boundary anchor frame n with RIFE(n-1, n+1).

    anchor_frames: 1-indexed frame numbers to de-spike (segment boundaries).
    Frames at the very start/end (no two neighbours) are left as-is. Returns out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    # copy everything first
    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            shutil.copy(os.path.join(frames_dir, f), os.path.join(out_dir, f))
    fixed = []
    for n in sorted(set(anchor_frames)):
        prev = os.path.join(frames_dir, pattern.format(n - 1))
        nxt = os.path.join(frames_dir, pattern.format(n + 1))
        if os.path.exists(prev) and os.path.exists(nxt):
            rife_mid(prev, nxt, os.path.join(out_dir, pattern.format(n)))
            fixed.append(n)
    print(f"[encode] de-spiked anchor frames {fixed} (RIFE neighbour-midpoint) -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Encode frames -> portrait mp4 (+optional anchor de-spike)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", required=True, help="output WxH, e.g. 480x832")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--pattern", default="frame_%05d.png")
    ap.add_argument("--despike", default=None,
                    help="comma list of 1-indexed anchor frames to RIFE-despike, e.g. 49,97")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    src = args.frames_dir
    if args.despike:
        anchors = [int(x) for x in args.despike.split(",")]
        src = despike_anchors(args.frames_dir, args.frames_dir.rstrip("/") + "_despike", anchors)
    encode(src, args.out, (w, h), fps=args.fps, pattern=args.pattern)
