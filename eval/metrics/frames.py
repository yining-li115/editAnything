"""Frame / mask IO + sampling for the benchmark evaluator.

Edited and source videos are frame-aligned (edited = first N frames of source at
native res), so we decode both, sample the same indices (FiVE stride-8 by default),
and load the regenerated per-frame edit masks aligned to those indices.
"""
import glob
import os

import numpy as np
from PIL import Image


def decode_video(path):
    """Decode an mp4 to a list of RGB uint8 HxWx3 arrays (via imageio-ffmpeg)."""
    import imageio.v2 as imageio
    rd = imageio.get_reader(path, "ffmpeg")
    frames = [np.asarray(f)[:, :, :3] for f in rd]
    rd.close()
    return frames


def sample_indices(n, stride=8, max_n=None):
    """FiVE-style: every `stride`-th frame (0-based), optionally capped to max_n."""
    idx = list(range(0, n, stride))
    if max_n:
        idx = idx[:max_n]
    return idx


def load_mask_frames(mask_dir):
    """Sorted per-frame masks (frame_*.png) as binary HxW bool arrays."""
    paths = sorted(glob.glob(f"{mask_dir}/frame_*.png"))
    out = []
    for p in paths:
        m = np.asarray(Image.open(p).convert("L"))
        out.append(m > 127)
    return out


def to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))


def align_masks(mask_frames, idx, size_wh):
    """Pick masks at sampled indices, resize (nearest) to size_wh=(W,H), return
    binary HxW arrays. Falls back to all-ones (whole frame) if a mask is missing."""
    W, H = size_wh
    out = []
    for i in idx:
        if i < len(mask_frames) and mask_frames[i] is not None:
            m = to_pil(mask_frames[i].astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
            out.append(np.asarray(m) > 127)
        else:
            out.append(np.ones((H, W), bool))
    return out
