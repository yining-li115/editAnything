"""RoMa propagation — warp the frame-0 new-object (banana) to every viewpoint.

This fills the one gap to general, any-video operation: the reanchor pipeline needs
a per-frame target mask that follows the object and a clean per-segment anchor.
Originally made by warping the frame-0 Gemini edit to each frame with RoMa (code
lost); this re-implements that.

Given the original frames + a frame-0 reference (the Gemini edit, with the new
object placed where the old one was) + the new object's mask on frame 0, we:
  1. RoMa dense-match frame0 <-> frame_k (real scene geometry: table/hand/wall),
  2. warp the frame-0 object RGB + mask into frame_k's geometry (grid_sample),
  3. gate by RoMa certainty,
  -> per-frame target masks (all frames) + per-segment anchors (object composited
     onto the real frame at each segment start).

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


def propagate(frames_dir, ref0_rgb, ref0_mask, out_anchor_dir, out_mask_dir,
              segment_starts, *, device="cuda", cert_thresh=0.4):
    """Warp the frame-0 object to every frame.

    ref0_rgb : HxWx3 uint8 RGB — frame 0 with the new object placed (Gemini edit).
    ref0_mask: HxW uint8 (0/255) — the new object's mask on frame 0.
    Writes out_mask_dir/frame_*.png (per-frame target masks) and
    out_anchor_dir/anchor_<start:04d>.png (per-segment anchors). Returns both dirs.
    """
    # SAM3 (built before this) leaves torch's default dtype as bfloat16, which makes
    # RoMa's runtime tensors bf16 against its float32 weights -> dtype mismatch.
    # Pin float32 (+ no autocast) for the whole RoMa pass, restore afterwards.
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
    starts = set(segment_starts)

    ref_rgb = cv2.resize(ref0_rgb, (w0, h0))
    ref_m3 = cv2.resize(np.repeat((ref0_mask > 127).astype(np.float32)[..., None], 3, 2),
                        (w0, h0), interpolation=cv2.INTER_NEAREST)

    for k, fp in enumerate(frame_paths):
        name = os.path.basename(fp)
        with torch.autocast("cuda", enabled=False):
            warp, cert = roma.match(f0, fp, device=device)   # frame0 -> frame_k
        S = warp.shape[1]
        gridA = warp[0][:, S:, 2:].unsqueeze(0).to(device)   # B-side: A-coords per B pixel
        certB = cert[0][:, S:]                                # (S,S)

        wb = F.grid_sample(_sq(ref_rgb, S, device), gridA, align_corners=False)
        wm = F.grid_sample(_sq(ref_m3, S, device), gridA, align_corners=False)[0, 0]
        wm = ((wm > 0.5) & (certB > cert_thresh)).to(torch.uint8).cpu().numpy()        # (S,S)
        wb = wb[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()                 # (S,S,3)

        wm = cv2.resize(wm, (w0, h0), interpolation=cv2.INTER_NEAREST)
        wb = cv2.resize(wb, (w0, h0))
        cv2.imwrite(f"{out_mask_dir}/{name}", (wm * 255).astype(np.uint8))

        if k in starts:
            base = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            comp = np.where(wm[..., None] > 0, wb, base).astype(np.uint8)
            cv2.imwrite(f"{out_anchor_dir}/anchor_{k:04d}.png",
                        cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        if k % 25 == 0:
            print(f"[roma] frame {k}/{len(frame_paths)} cert~{float(certB.mean()):.2f}")
    torch.set_default_dtype(prev_dtype)
    print(f"[roma] masks -> {out_mask_dir}; anchors({sorted(starts)}) -> {out_anchor_dir}")
    return out_anchor_dir, out_mask_dir
