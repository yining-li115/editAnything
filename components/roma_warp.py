"""RoMa warp primitives — dense frame0<->frame_k matching + warp ops.

Low-level, model-agnostic. Given frame0 and frame_k, RoMa dense-matches the real
scene geometry (table/hand/wall) and yields a warp field; we then warp frame-0
content (a mask region, or a whole reference image) into frame_k's viewpoint.

Two consumers build on this (each owns its own loop, so they stay independent —
edit_mask is generic/cross-candidate, anchor is VideoPainter-specific):
  - edit_mask: warp the frame-0 edit region -> per-frame edit masks.
  - anchor:    warp the whole clean ref0 -> per-segment-start anchors.
k == 0 is the identity warp (handled by the caller).

Uses romatch (install --no-deps to keep torch 2.4; use_custom_corr=False -> pure
PyTorch local-correlation, no CUDA extension build).
"""
import contextlib
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


@contextlib.contextmanager
def roma_float32():
    """Force float32 default dtype for the duration of a warp loop.

    SAM3 (built before RoMa runs) leaves torch's default dtype as bfloat16, which
    makes RoMa's runtime tensors bf16 against its float32 weights -> dtype mismatch.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _sq(img, S, device, mode="bilinear"):
    """numpy HxWxC -> (1,C,S,S) float tensor on device."""
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(device)
    return F.interpolate(t, (S, S), mode=mode, align_corners=False)


def match(f0_path, fk_path, device="cuda"):
    """RoMa-match frame0 -> frame_k. Returns (gridA, S): the per-B-pixel sampling
    grid (1,S,S,2) into frame0, and the working square size S."""
    roma = load_roma(device)
    with torch.autocast("cuda", enabled=False):
        warp, _ = roma.match(f0_path, fk_path, device=device)
    S = warp.shape[1]
    gridA = warp[0][:, S:, 2:].unsqueeze(0).to(device)     # A-coords per B pixel
    return gridA, S


def warp_binary(region0, gridA, S, out_hw, device="cuda"):
    """Warp a frame-0 binary region (HxW 0/255) into frame_k -> 0/1 uint8 at out_hw."""
    reg3 = np.repeat((region0 > 127).astype(np.float32)[..., None], 3, 2)
    m = F.grid_sample(_sq(reg3, S, device), gridA, align_corners=False)[0, 0]
    mask = (m > 0.5).to(torch.uint8).cpu().numpy()
    return cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)


def warp_rgb(ref_rgb, gridA, S, out_hw, device="cuda"):
    """Warp a frame-0 RGB image into frame_k -> uint8 RGB at out_hw."""
    a = F.grid_sample(_sq(ref_rgb, S, device), gridA, align_corners=False)
    a = a[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return cv2.resize(a, (out_hw[1], out_hw[0]))
