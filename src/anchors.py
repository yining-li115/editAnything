"""Anchor / target-region propagation — the pluggable, currently-stubbed piece.

The best (reanchor) result needs, per segment, a CLEAN anchor frame showing the
NEW object at that segment's viewpoint, plus a per-frame target-region mask that
follows the object as the camera moves. In the original experiment both came from
warping the frame-0 edit to every viewpoint (RoMa) — that code was lost and is the
one real gap to general, any-video operation.

This module defines the interface and ships a phase-A backend that simply LOADS
pre-made assets (the validated cup2 anchors + masks). A future RoMa/Gemini backend
plugs in behind the same interface without touching the rest of the pipeline.

Interface:
  provider.anchor_for_start(start) -> PIL.Image          # clean anchor for a segment
  provider.target_mask_dir -> str | None                 # per-frame new-object masks
"""
import os
import glob


class AssetsAnchors:
    """Phase-A backend: load ready-made anchors + target masks from a folder.

    Expected layout (matches exp3_bundle/inputs):
      <assets_dir>/anchors/ff_<start:04d>.png   (or anchor_<start:04d>.png)
      <assets_dir>/banana_masks/frame_*.png     (per-frame new-object masks)
    """
    def __init__(self, assets_dir, anchors_subdir="anchors", masks_subdir="banana_masks"):
        self.assets_dir = assets_dir
        self.anchors_dir = os.path.join(assets_dir, anchors_subdir)
        md = os.path.join(assets_dir, masks_subdir)
        self.target_mask_dir = md if os.path.isdir(md) else None

    def _anchor_path(self, start):
        for tmpl in (f"ff_{start:04d}.png", f"anchor_{start:04d}.png"):
            p = os.path.join(self.anchors_dir, tmpl)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"no anchor for start={start} in {self.anchors_dir} "
            f"(looked for ff_{start:04d}.png / anchor_{start:04d}.png)")

    def anchor_for_start(self, start):
        from PIL import Image
        return Image.open(self._anchor_path(start)).convert("RGB")

    def available_starts(self):
        starts = []
        for p in glob.glob(os.path.join(self.anchors_dir, "*.png")):
            stem = os.path.splitext(os.path.basename(p))[0]
            digits = "".join(c for c in stem if c.isdigit())
            if digits:
                starts.append(int(digits))
        return sorted(set(starts))


class RomaAnchors:
    """General backend: warp a frame-0 reference to every viewpoint with RoMa.

    Inputs:
      frames_dir   : original frames frame_*.png
      ref0_path    : frame-0 reference (Gemini edit: new object placed on frame 0)
      target_word  : noun for SAM3 to segment the new object on ref0 (e.g. "banana")
                     — only used if ref0_mask_path is not given
      work_dir     : where generated anchors/ + masks/ are written
      segment_starts : 0-based segment starts needing anchors
      ref0_mask_path : optional precomputed frame-0 object mask (skips SAM3)

    Produces the same interface as AssetsAnchors (anchor_for_start + target_mask_dir),
    computed lazily on first access.
    """
    def __init__(self, frames_dir, ref0_path, target_word, work_dir, segment_starts,
                 *, device="cuda", ref0_mask_path=None):
        self.frames_dir = frames_dir
        self.ref0_path = ref0_path
        self.target_word = target_word
        self.work_dir = work_dir
        self.segment_starts = list(segment_starts)
        self.device = device
        self.ref0_mask_path = ref0_mask_path
        self.anchors_dir = os.path.join(work_dir, "anchors")
        self._mask_dir = os.path.join(work_dir, "masks")
        self._prepared = False

    def _ref0_mask(self, ref0_rgb_path):
        """Frame-0 object mask: use the provided one, else SAM3 on ref0."""
        import cv2
        if self.ref0_mask_path:
            return cv2.imread(self.ref0_mask_path, 0)
        import tempfile, shutil
        import sam3_track
        tmp = os.path.join(self.work_dir, "_ref0_sam3")
        os.makedirs(tmp, exist_ok=True)
        shutil.copy(ref0_rgb_path, os.path.join(tmp, "frame_00001.png"))
        sam3_track.track(sam3_track.build_predictor(), tmp, self.target_word, tmp + "_mask")
        return cv2.imread(os.path.join(tmp + "_mask", "frame_00001.png"), 0)

    def prepare(self):
        if self._prepared:
            return
        # Reuse a previous run's RoMa outputs if already on disk (skip recompute).
        if (glob.glob(os.path.join(self._mask_dir, "frame_*.png"))
                and glob.glob(os.path.join(self.anchors_dir, "anchor_*.png"))):
            print(f"[anchors] reusing cached RoMa outputs in {self.work_dir}")
            self._prepared = True
            return
        import cv2
        import roma_propagate
        ref0_rgb = cv2.cvtColor(cv2.imread(self.ref0_path), cv2.COLOR_BGR2RGB)
        ref0_mask = self._ref0_mask(self.ref0_path)
        roma_propagate.propagate(self.frames_dir, ref0_rgb, ref0_mask,
                                 self.anchors_dir, self._mask_dir, self.segment_starts,
                                 device=self.device)
        self._prepared = True

    @property
    def target_mask_dir(self):
        self.prepare()
        return self._mask_dir

    def anchor_for_start(self, start):
        from PIL import Image
        self.prepare()
        p = os.path.join(self.anchors_dir, f"anchor_{start:04d}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"RoMa anchor missing for start={start}: {p}")
        return Image.open(p).convert("RGB")


def get_anchor_provider(backend, **kwargs):
    if backend == "assets":
        return AssetsAnchors(kwargs["assets_dir"],
                             anchors_subdir=kwargs.get("anchors_subdir", "anchors"),
                             masks_subdir=kwargs.get("masks_subdir", "banana_masks"))
    if backend == "roma":
        return RomaAnchors(kwargs["frames_dir"], kwargs["ref0_path"], kwargs["target_word"],
                           kwargs["work_dir"], kwargs["segment_starts"],
                           device=kwargs.get("device", "cuda"),
                           ref0_mask_path=kwargs.get("ref0_mask_path"))
    raise ValueError(f"unknown anchor backend: {backend!r} (use 'assets' or 'roma')")
