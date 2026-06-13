"""RoMa propagation — one pass that produces BOTH per-frame edit masks and
per-segment anchors by warping frame-0 content to every frame's viewpoint.

For each frame k, RoMa dense-matches frame0 <-> frame_k (real scene geometry:
table/hand/wall) and gives a warp field:
  - EDIT MASK (every frame): warp the frame-0 edit region (target ∪ source, dilated)
    into frame_k -> per-frame mask that follows the object. This is the region
    VideoPainter is allowed to paint, sized for the new object (not the small source object).
  - ANCHOR (segment starts only): warp the WHOLE clean frame-0 reference into
    frame_s -> the clean "new object at this viewpoint" conditioning frame.
(k == 0 is the identity warp.)

Uses romatch (install --no-deps to keep torch 2.4; use_custom_corr=False -> pure
PyTorch local-correlation, no CUDA extension build).
"""
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F

_roma = None


def load_roma(device="cuda"):
    global _roma
    if _roma is None:
        from romatch import roma_outdoor
        _roma = roma_outdoor(device=device, use_custom_corr=False)
    return _roma


def _sq(img, S, device, mode="bilinear"):
    """numpy HxWxC -> (1,C,S,S) float tensor on device."""
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(device)
    return F.interpolate(t, (S, S), mode=mode, align_corners=False)


def propagate(frames_dir, ref0_rgb, region0, out_anchor_dir, out_mask_dir,
              segment_starts, *, device="cuda"):
    """One RoMa pass over all frames -> per-frame edit masks + per-segment anchors.

    ref0_rgb : HxWx3 uint8 RGB — clean frame-0 reference (new object in place).
    region0  : HxW uint8 (0/255) — frame-0 edit region (target ∪ source, dilated).
    For every frame k: warp region0 -> out_mask_dir/<name> (edit mask).
    For k in segment_starts: warp the whole ref0_rgb -> out_anchor_dir/anchor_<k>.png.
    Returns (out_anchor_dir, out_mask_dir).
    """
    # SAM3 (built before this) leaves torch's default dtype as bfloat16, which makes
    # RoMa's runtime tensors bf16 against its float32 weights -> dtype mismatch.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    roma = load_roma(device)
    frame_paths = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    if not frame_paths:
        torch.set_default_dtype(prev_dtype)
        raise FileNotFoundError(f"no frame_*.png in {frames_dir}")
    f0 = frame_paths[0]
    h0, w0 = cv2.imread(f0).shape[:2]
    os.makedirs(out_anchor_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    ref_rgb = cv2.resize(ref0_rgb, (w0, h0))
    reg0 = cv2.resize(region0, (w0, h0), interpolation=cv2.INTER_NEAREST)
    reg0_3 = np.repeat((reg0 > 127).astype(np.float32)[..., None], 3, 2)
    starts = set(segment_starts)

    for k, fp in enumerate(frame_paths):
        name = os.path.basename(fp)
        if k == 0:
            mask_k = (reg0 > 127).astype(np.uint8)
            anchor_k = ref_rgb
        else:
            with torch.autocast("cuda", enabled=False):
                warp, _ = roma.match(f0, fp, device=device)        # frame0 -> frame_k
            S = warp.shape[1]
            gridA = warp[0][:, S:, 2:].unsqueeze(0).to(device)     # A-coords per B pixel
            m = F.grid_sample(_sq(reg0_3, S, device), gridA, align_corners=False)[0, 0]
            mask_k = (m > 0.5).to(torch.uint8).cpu().numpy()
            mask_k = cv2.resize(mask_k, (w0, h0), interpolation=cv2.INTER_NEAREST)
            if k in starts:
                a = F.grid_sample(_sq(ref_rgb, S, device), gridA, align_corners=False)
                anchor_k = cv2.resize(a[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(),
                                      (w0, h0))
        cv2.imwrite(f"{out_mask_dir}/{name}", (mask_k * 255).astype(np.uint8))
        if k in starts:
            cv2.imwrite(f"{out_anchor_dir}/anchor_{k:04d}.png",
                        cv2.cvtColor(anchor_k, cv2.COLOR_RGB2BGR))
        if k % 25 == 0:
            print(f"[roma] frame {k}/{len(frame_paths)}")
    torch.set_default_dtype(prev_dtype)
    print(f"[roma] edit masks -> {out_mask_dir}; anchors({sorted(starts)}) -> {out_anchor_dir}")
    return out_anchor_dir, out_mask_dir
