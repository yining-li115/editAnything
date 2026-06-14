"""SAM3 text-prompt tracking -> per-frame source-object mask (decoupled).

Generates the mask for the object to be REPLACED across a folder of
frames, at the frames' native resolution, written as frame_*.png (0/255). This is
fully standalone: VideoPainter never calls SAM3 — the masks are produced here as
files and later fed to generate via --mask_dir.

Note on the final mask: generate needs the edit region to cover BOTH the old
object and the new one. This module produces only the source-object mask;
the target-object region comes from anchors, and the pipeline unions
them. Dilation is applied later inside generate (--dilate), not here.
"""
import os
import glob
import shutil
import numpy as np
import cv2
import torch

from sam3.model_builder import build_sam3_video_predictor


def build_predictor():
    return build_sam3_video_predictor(
        gpus_to_use=range(torch.cuda.device_count()) if torch.cuda.is_available() else None
    )


def track(predictor, frames_dir, text, out_mask_dir, *, work_dir=None):
    """Track `text` across frame_*.png in frames_dir; write per-frame 0/255 masks.

    Masks are written to out_mask_dir with the SAME filenames as the input frames,
    at each frame's native resolution. Returns out_mask_dir.
    """
    text = str(text).strip()
    if not text:
        raise ValueError("sam3_track.track: empty text prompt")

    frame_paths = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no frame_*.png in {frames_dir}")
    names = [os.path.basename(p) for p in frame_paths]
    h0, w0 = cv2.imread(frame_paths[0]).shape[:2]

    # SAM3 start_session wants a folder of integer-named jpgs (a video sequence).
    work_dir = work_dir or os.path.join(out_mask_dir, "_sam3_frames")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    for i, p in enumerate(frame_paths):
        cv2.imwrite(os.path.join(work_dir, f"{i}.jpg"), cv2.imread(p))

    resp = predictor.handle_request(dict(type="start_session", resource_path=work_dir))
    session_id = resp["session_id"]
    outputs = {}
    try:
        predictor.handle_request(dict(type="add_prompt", session_id=session_id,
                                      frame_index=0, text=text))
        for r in predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=session_id)):
            outputs[r["frame_index"]] = r["outputs"]
    finally:
        predictor.handle_request(dict(type="close_session", session_id=session_id))

    os.makedirs(out_mask_dir, exist_ok=True)
    n_hit = 0
    for idx, name in enumerate(names):
        merged = np.zeros((h0, w0), np.uint8)
        out = outputs.get(idx)
        if out is not None and len(out["out_obj_ids"]) > 0:
            bm = out["out_binary_masks"]
            if hasattr(bm, "cpu"):
                bm = bm.cpu().numpy()
            bm = np.asarray(bm)                      # (N_obj, h, w)
            union = np.any(bm > 0.5, axis=0).astype(np.uint8)
            if union.shape != (h0, w0):
                union = cv2.resize(union, (w0, h0), interpolation=cv2.INTER_NEAREST)
            merged = (union > 0).astype(np.uint8)
            if merged.any():
                n_hit += 1
        cv2.imwrite(os.path.join(out_mask_dir, name), (merged * 255).astype(np.uint8))

    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[sam3_track] '{text}': {n_hit}/{len(names)} frames matched -> {out_mask_dir}")
    if n_hit == 0:
        print("[sam3_track] WARNING: 0 frames matched; try a simpler prompt.")
    return out_mask_dir


def mask_image(predictor, image_path, word, *, work_dir, tag="img"):
    """Run SAM3 `word` on a SINGLE image; return its 0/255 mask (HxW uint8).

    Wraps `track` (which wants a frame folder) for the one-image case used to mask
    the new object on ref0 and the old object on frame 0. Mirrors the old
    anchors._sam3_mask helper.
    """
    tmp = os.path.join(work_dir, f"_sam3_{tag}")
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(image_path, os.path.join(tmp, "frame_00001.png"))
    track(predictor, tmp, word, tmp + "_mask")
    return cv2.imread(os.path.join(tmp + "_mask", "frame_00001.png"), 0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SAM3 text-prompt per-frame mask")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--text", required=True, help="object to segment (e.g. the object to remove)")
    ap.add_argument("--out_mask_dir", required=True)
    args = ap.parse_args()
    track(build_predictor(), args.frames_dir, args.text, args.out_mask_dir)
