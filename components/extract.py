"""Extract — video -> frames, plus basic frame-set queries (no model)."""
import os
import glob
import subprocess
import cv2


def has_frames(d):
    return os.path.isdir(d) and len(glob.glob(f"{d}/frame_*.png")) > 0


def extract_frames(video, out_dir, resume=True, max_frames=None):
    if resume and has_frames(out_dir):
        print(f"[extract] reuse frames {out_dir}")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video]
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]   # only decode the first N frames
    cmd += ["-start_number", "1", f"{out_dir}/frame_%05d.png"]
    subprocess.run(cmd, check=True)
    print(f"[extract] {len(glob.glob(f'{out_dir}/frame_*.png'))} frames -> {out_dir}")
    return out_dir


def video_meta(frames_dir):
    paths = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    h, w = cv2.imread(paths[0]).shape[:2]
    return len(paths), (w, h)
