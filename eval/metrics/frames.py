"""Video / frame / mask IO + sampling shared by every metric.

Frames are returned as RGB uint8 numpy arrays (H, W, 3). Masks are binary uint8
(H, W) in {0, 1}, loaded from the pipeline's roma/masks/frame_*.png (0/255).
"""
import glob
import os

import numpy as np


def load_video(path):
    """Decode a video into a list of RGB uint8 frames."""
    import imageio.v3 as iio

    frames = [np.asarray(f)[:, :, :3] for f in iio.imiter(path)]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames


def sample_indices(total, every=10, n=0):
    """Indices to sample. If n>0, n evenly-spaced; else every Nth frame."""
    if n and n > 0:
        if total <= n:
            return list(range(total))
        return [round(i * (total - 1) / (n - 1)) for i in range(n)]
    every = max(1, int(every))
    return list(range(0, total, every))


def load_masks(mask_dir):
    """Load binary masks (frame_*.png, 0/255) -> list of {0,1} uint8 arrays.

    Returns None if the directory is missing or empty (metric should fall back).
    """
    if not mask_dir or not os.path.isdir(mask_dir):
        return None
    paths = sorted(glob.glob(os.path.join(mask_dir, "frame_*.png")))
    if not paths:
        return None
    import cv2

    masks = []
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        masks.append((m > 127).astype(np.uint8))
    return masks


def crop_to_mask(frame, mask, pad=8):
    """Crop `frame` to the bounding box of `mask` (with padding).

    Falls back to the full frame when the mask is empty.
    """
    if mask is None or mask.sum() == 0:
        return frame
    if mask.shape[:2] != frame.shape[:2]:
        import cv2

        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    y0, y1 = max(0, ys.min() - pad), min(frame.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(frame.shape[1], xs.max() + 1 + pad)
    crop = frame[y0:y1, x0:x1]
    return crop if crop.size else frame


def align_masks_to_frames(masks, frame_indices, n_frames):
    """Pick the masks matching the sampled frame indices.

    Mask count usually equals total frame count. If it differs (e.g. final.mp4
    has a different length than frames_src), index proportionally.
    """
    if masks is None:
        return [None] * len(frame_indices)
    out = []
    for i in frame_indices:
        if len(masks) == n_frames:
            j = i
        else:
            j = round(i * (len(masks) - 1) / max(1, n_frames - 1))
        out.append(masks[min(j, len(masks) - 1)])
    return out
