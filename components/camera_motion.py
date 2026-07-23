"""Camera-motion detector — decides CAMERA motion vs OBJECT-only motion.

Used by the orchestrator to route between:
  - static/near-static camera  -> roma_anchors + videopainter_generate
  - panning/dolly/handheld camera -> mvinpainter_anchors + mvinpainter_generate

Method (Option A — background-only homography/RANSAC):
  For sampled frame pairs (i, i+stride), detect ORB keypoints EXCLUDING the
  object mask region (so a big moving object never gets mistaken for camera
  motion, and a real pan/zoom isn't hidden by the object masking out most of
  the frame). Match keypoints, fit a homography with RANSAC. A pair is
  "coherent camera motion" if:
    - a large fraction of background matches agree with ONE global transform
      (high inlier ratio) -- i.e. the background really is moving as a rigid
      plane/rotation, not just noisy mismatches, AND
    - that transform implies a non-trivial pixel displacement across the
      frame (i.e. it's not just tripod micro-jitter).
  The video is flagged camera_motion=True if a majority of sampled pairs are
  coherent.

This is a lightweight stand-in for full visual odometry / SLAM: cheap (sparse
ORB + RANSAC, no dense flow, no deep model), and robust to the object motion
that VideoPainter/MVInpainter are meant to inpaint, because that region is
never used as evidence.
"""
from __future__ import annotations
import glob

import cv2
import numpy as np


def _frames(d: str) -> list[str]:
    return sorted(glob.glob(f"{d}/frame_*.png"))


def _background_mask(mask_gray: np.ndarray | None, erode_px: int = 15) -> np.ndarray | None:
    """0/255 mask of background-only pixels (object shrunk further away from its
    edges so points near the boundary — which are unreliable due to inpaint-region
    ambiguity — aren't used either)."""
    if mask_gray is None:
        return None
    bg = (mask_gray < 127).astype(np.uint8) * 255
    if erode_px > 0:
        bg = cv2.erode(bg, np.ones((erode_px, erode_px), np.uint8))
    return bg


def _transform_magnitude(H: np.ndarray, w: int, h: int) -> float:
    """Max corner displacement implied by homography H, in pixels."""
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    orig = corners.reshape(-1, 2)
    return float(np.max(np.linalg.norm(warped - orig, axis=1)))


def detect(
    frames_dir: str,
    mask_dir: str | None = None,
    *,
    stride: int = 6,
    max_pairs: int = 40,
    px_threshold: float = 8.0,
    inlier_ratio_thresh: float = 0.6,
    coherent_fraction_thresh: float = 0.5,
    min_points: int = 30,
    mask_erode_px: int = 15,
    fps: float = 25.0,
    min_duration_sec: float = 3.0,
) -> dict:
    """Return a dict describing whether the clip shows CAMERA motion.

    frames_dir: directory of frame_*.png (the ORIGINAL source frames).
    mask_dir:   directory of the per-frame SOURCE object mask (e.g. sam3_mask's
                output). Strongly recommended — without it, a large moving
                object can be mistaken for camera motion. Pass None only if no
                mask exists yet (rare; the orchestrator normally calls this
                right after sam3_mask).
    fps, min_duration_sec: clips shorter than min_duration_sec (computed as
                n_frames / fps) are forced to camera_motion=False WITHOUT
                running any detection — a 2-3s clip rarely gives enough camera
                travel for "moving vs static" to be a meaningful or reliable
                call, and skipping it also saves the ORB/RANSAC cost. Set
                min_duration_sec=0 to disable this gate entirely.

    Returns:
        {
          "camera_motion": bool,
          "coherent_fraction": float,        # fraction of sampled pairs flagged coherent
          "median_transform_px": float,      # typical implied background displacement
          "n_pairs_sampled": int,
          "duration_sec": float,             # n_frames / fps
          "forced_static_short_clip": bool,  # True if the duration gate fired
          "samples": [ {i, j, n_matches, inlier_ratio, transform_px, coherent}, ... ],
        }
    """
    frames = _frames(frames_dir)
    n_total = len(frames)
    duration_sec = n_total / fps if fps > 0 else float("inf")

    if min_duration_sec > 0 and duration_sec < min_duration_sec:
        print(f"[camera_motion] clip is {duration_sec:.1f}s (< {min_duration_sec}s) — "
              f"forcing camera_motion=False without running detection")
        return {
            "camera_motion": False,
            "coherent_fraction": 0.0,
            "median_transform_px": 0.0,
            "n_pairs_sampled": 0,
            "duration_sec": round(duration_sec, 2),
            "forced_static_short_clip": True,
            "samples": [],
        }

    masks = _frames(mask_dir) if mask_dir else None
    n = min(len(frames), len(masks)) if masks else n_total
    if n < 2:
        return {
            "camera_motion": False,
            "coherent_fraction": 0.0,
            "median_transform_px": 0.0,
            "n_pairs_sampled": 0,
            "duration_sec": round(duration_sec, 2),
            "forced_static_short_clip": False,
            "samples": [],
            "reason": "fewer than 2 frames available",
        }

    idxs = list(range(0, n - stride, stride))[:max_pairs]
    if not idxs:
        idxs = [0]

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    samples = []
    for i in idxs:
        j = min(i + stride, n - 1)
        img1 = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frames[j], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        h, w = img1.shape

        bg1 = bg2 = None
        if masks:
            m1 = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
            m2 = cv2.imread(masks[j], cv2.IMREAD_GRAYSCALE)
            bg1 = _background_mask(m1, mask_erode_px)
            bg2 = _background_mask(m2, mask_erode_px)

        kp1, des1 = orb.detectAndCompute(img1, bg1)
        kp2, des2 = orb.detectAndCompute(img2, bg2)
        if des1 is None or des2 is None or len(kp1) < min_points or len(kp2) < min_points:
            samples.append({"i": i, "j": j, "n_matches": 0, "inlier_ratio": 0.0,
                             "transform_px": 0.0, "coherent": False})
            continue

        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:400]
        if len(matches) < min_points:
            samples.append({"i": i, "j": j, "n_matches": len(matches), "inlier_ratio": 0.0,
                             "transform_px": 0.0, "coherent": False})
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        if H is None or inliers is None:
            samples.append({"i": i, "j": j, "n_matches": len(matches), "inlier_ratio": 0.0,
                             "transform_px": 0.0, "coherent": False})
            continue

        inlier_ratio = float(inliers.sum()) / len(inliers)
        transform_px = _transform_magnitude(H, w, h)
        coherent = inlier_ratio >= inlier_ratio_thresh and transform_px >= px_threshold
        samples.append({
            "i": i, "j": j, "n_matches": len(matches),
            "inlier_ratio": round(inlier_ratio, 3),
            "transform_px": round(transform_px, 2),
            "coherent": coherent,
        })

    if not samples:
        return {
            "camera_motion": False,
            "coherent_fraction": 0.0,
            "median_transform_px": 0.0,
            "n_pairs_sampled": 0,
            "duration_sec": round(duration_sec, 2),
            "forced_static_short_clip": False,
            "samples": [],
            "reason": "no valid frame pairs produced enough matches",
        }

    coherent_fraction = sum(s["coherent"] for s in samples) / len(samples)
    median_transform_px = float(np.median([s["transform_px"] for s in samples]))
    camera_motion = coherent_fraction >= coherent_fraction_thresh

    return {
        "camera_motion": camera_motion,
        "coherent_fraction": round(coherent_fraction, 3),
        "median_transform_px": round(median_transform_px, 2),
        "n_pairs_sampled": len(samples),
        "duration_sec": round(duration_sec, 2),
        "forced_static_short_clip": False,
        "samples": samples,
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Detect camera motion vs object-only motion")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--mask_dir", default=None, help="per-frame SOURCE object mask (recommended)")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--px_threshold", type=float, default=8.0)
    args = ap.parse_args()
    result = detect(args.frames_dir, args.mask_dir, stride=args.stride, px_threshold=args.px_threshold)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    print(f"camera_motion = {result['camera_motion']}")