"""Encode frames -> final video, with optional RIFE frame interpolation (smoothing).

The pipeline works at 720x480 (squished landscape); the final video is unsquished
back to the original portrait resolution here.

Interpolation smooths motion AND softens the 1-frame anchor "pop" at reanchor
segment boundaries (intrinsic to multi-chunk). Preferred backend is **RIFE**
(rife-ncnn-vulkan, GPU via Vulkan) operating on the frame folder (2x frames);
falls back to ffmpeg `minterpolate` if the RIFE binary isn't found.

RIFE binary (prebuilt, includes models) — point RIFE_BIN at it:
    https://github.com/nihui/rife-ncnn-vulkan/releases  (e.g. .../rife-ncnn-vulkan)
Default location used here: tools/rife-ncnn-vulkan-20221029-ubuntu/ under the repo parent.
"""
import os
import glob
import shutil
import subprocess

# RIFE binary + model dir (override via env). Models live next to the binary.
RIFE_BIN = os.environ.get(
    "RIFE_BIN",
    "/root/project/tools/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan")
RIFE_MODEL = os.environ.get("RIFE_MODEL", "rife-v4.6")


def encode(frames_dir, out_path, size_wh, *, fps=25, crf=18, pattern="frame_%05d.png"):
    """Encode frames in frames_dir to out_path, scaled to size_wh=(W,H)."""
    w, h = size_wh
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", os.path.join(frames_dir, pattern),
        "-vf", f"scale={w}:{h}:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        out_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"[encode] wrote {out_path} ({w}x{h} @ {fps}fps)")
    return out_path


def rife_frames(in_dir, out_dir):
    """RIFE 2x frame interpolation on a folder (rife-ncnn-vulkan, GPU/Vulkan).

    N input frames -> ~2N output frames named 00000001.png... Returns (out_dir,
    pattern, n_out) or None if the RIFE binary is unavailable.
    """
    if not os.path.exists(RIFE_BIN):
        return None
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(os.path.dirname(RIFE_BIN), RIFE_MODEL)
    subprocess.run([RIFE_BIN, "-i", in_dir, "-o", out_dir, "-m", model_path],
                   check=True, stderr=subprocess.DEVNULL)
    n = len(glob.glob(f"{out_dir}/*.png"))
    print(f"[encode] RIFE {len(glob.glob(f'{in_dir}/*.png'))}->{n} frames -> {out_dir}")
    return out_dir, "%08d.png", n


def interpolate_video(in_path, out_path, *, target_fps=50, crf=18):
    """Fallback: smooth an existing mp4 via ffmpeg minterpolate."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
        "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out_path,
    ], check=True)
    print(f"[encode] minterpolate -> {out_path} (~{target_fps}fps)")
    return out_path


def encode_interpolated(frames_dir, out_path, size_wh, *, fps=25, crf=18,
                        pattern="frame_%05d.png", work_dir=None):
    """Write a smoothed (2x fps) video. Uses RIFE on the frames if available,
    else minterpolates a base encode. Returns out_path."""
    rife = rife_frames(frames_dir, work_dir or (frames_dir.rstrip("/") + "_rife"))
    if rife:
        rife_dir, rife_pattern, _ = rife
        return encode(rife_dir, out_path, size_wh, fps=fps * 2, crf=crf, pattern=rife_pattern)
    # fallback: base encode then minterpolate the video
    print("[encode] RIFE binary not found; falling back to ffmpeg minterpolate")
    base = out_path.replace(".mp4", "_base.mp4")
    encode(frames_dir, base, size_wh, fps=fps, crf=crf, pattern=pattern)
    return interpolate_video(base, out_path, target_fps=fps * 2, crf=crf)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Encode frames to portrait mp4 (+optional RIFE interpolation)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", required=True, help="output WxH, e.g. 480x832")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--pattern", default="frame_%05d.png")
    ap.add_argument("--interpolate", action="store_true", help="also write a smoothed *_interp.mp4 (RIFE)")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    out = encode(args.frames_dir, args.out, (w, h), fps=args.fps, pattern=args.pattern)
    if args.interpolate:
        base, ext = os.path.splitext(out)
        encode_interpolated(args.frames_dir, f"{base}_interp{ext}", (w, h),
                            fps=args.fps, pattern=args.pattern)
