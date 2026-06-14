"""Seamless composite (CPU only, no model) — kill chunk-boundary background jumps.

Per-segment generation re-synthesizes the whole frame, so the *background*
drifts slightly between segments. Fix: keep only the generated object (inside its
mask, feathered) and paste it onto a single temporally-consistent FIXED PLATE, so
the background is never regenerated per chunk.

Refactor of the validated `composite_onto_rose.py` (renamed: the "rose" plate was
just exp2's cup-removed background; here the plate is generic). For exp3 the plate
is simply the ORIGINAL frames — the per-frame mask covers the old object, so
outside the mask the original has no object and inside the mask we paste the new
one. Feather = MaxFilter(9) + GaussianBlur(4), work res 720x480.
"""
import os
import numpy as np
from PIL import Image, ImageFilter

W, H = 720, 480


def composite(plate_dir, object_dir, mask_dir, out_dir, *, total=300,
              maxfilter=9, feather=4):
    """Paste the masked object from object_dir onto plate_dir, feathered.

    plate_dir : fixed, temporally-consistent background frames (e.g. original frames)
    object_dir: generated frames containing the new object (generate output)
    mask_dir  : per-frame masks marking where the object is
    All keyed by frame_%05d.png (1..total). Returns out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, total + 1):
        n = f"frame_{i:05d}.png"
        bg = Image.open(f"{plate_dir}/{n}").convert("RGB").resize((W, H), Image.LANCZOS)
        obj = Image.open(f"{object_dir}/{n}").convert("RGB")
        if obj.size != (W, H):
            obj = obj.resize((W, H), Image.LANCZOS)
        m = Image.open(f"{mask_dir}/{n}").convert("L").resize((W, H), Image.NEAREST)
        m = m.filter(ImageFilter.MaxFilter(maxfilter)).filter(ImageFilter.GaussianBlur(feather))
        a = (np.asarray(m, float) / 255.0)[..., None]
        out = (np.asarray(obj, float) * a + np.asarray(bg, float) * (1 - a)).astype(np.uint8)
        Image.fromarray(out).save(f"{out_dir}/{n}")
    print(f"[composite] composited {total} frames -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Seamless composite onto a fixed plate")
    ap.add_argument("--plate_dir", required=True, help="fixed background frames (e.g. original frames)")
    ap.add_argument("--object_dir", required=True, help="generated frames (generate output/frames)")
    ap.add_argument("--mask_dir", required=True, help="per-frame masks frame_*.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--maxfilter", type=int, default=9)
    ap.add_argument("--feather", type=int, default=4)
    args = ap.parse_args()
    composite(args.plate_dir, args.object_dir, args.mask_dir, args.out_dir,
              total=args.total, maxfilter=args.maxfilter, feather=args.feather)
