"""Anchor component — per-segment clean conditioning frames (VideoPainter-specific).

The whole clean ref0 RoMa-warped into each segment-start viewpoint -> the "new
object at this viewpoint" I2V condition that stops the inserted object dissolving
after the first clip. ONLY VideoPainter's per-chunk reanchor consumes this; other
candidates ignore it. Kept separate from edit_mask (which is generic).

Backends:
  - roma:   warp ref0 to each segment start.  (any video)
  - assets: load prepared anchor images.
"""
import os
import glob
import cv2

from components import roma_warp


def _frames(frames_dir):
    return sorted(glob.glob(f"{frames_dir}/frame_*.png"))


class AssetsAnchor:
    """Load prepared anchors from <assets_dir>/<anchors_subdir>/ (ff_/anchor_ names)."""
    def __init__(self, assets_dir, anchors_subdir="anchors"):
        self.anchors_dir = os.path.join(assets_dir, anchors_subdir)

    def _path(self, start):
        for tmpl in (f"ff_{start:04d}.png", f"anchor_{start:04d}.png"):
            p = os.path.join(self.anchors_dir, tmpl)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"no anchor for start={start} in {self.anchors_dir} "
            f"(looked for ff_{start:04d}.png / anchor_{start:04d}.png)")

    def anchor_for_start(self, start):
        from PIL import Image
        return Image.open(self._path(start)).convert("RGB")


class RomaAnchor:
    """Warp the clean ref0 into each segment-start viewpoint (per-start cache)."""
    def __init__(self, frames_dir, ref0_path, work_dir, segment_starts, *, device="cuda"):
        self.frames_dir = frames_dir
        self.ref0_path = ref0_path
        self.work_dir = work_dir
        self.segment_starts = list(segment_starts)
        self.device = device
        self.anchors_dir = os.path.join(work_dir, "anchors")
        self._prepared = False

    def _path(self, start):
        return os.path.join(self.anchors_dir, f"anchor_{start:04d}.png")

    def prepare(self):
        if self._prepared:
            return
        os.makedirs(self.anchors_dir, exist_ok=True)
        missing = [s for s in self.segment_starts if not os.path.exists(self._path(s))]
        if not missing:
            print(f"[anchor] reusing cached anchors in {self.anchors_dir}")
            self._prepared = True
            return
        frame_paths = _frames(self.frames_dir)
        f0 = frame_paths[0]
        hf, wf = cv2.imread(f0).shape[:2]
        ref_rgb = cv2.resize(cv2.cvtColor(cv2.imread(self.ref0_path), cv2.COLOR_BGR2RGB), (wf, hf))
        with roma_warp.roma_float32():
            for s in missing:
                if s == 0:
                    anchor = ref_rgb
                else:
                    gridA, S = roma_warp.match(f0, frame_paths[s], device=self.device)
                    anchor = roma_warp.warp_rgb(ref_rgb, gridA, S, (hf, wf), device=self.device)
                cv2.imwrite(self._path(s), cv2.cvtColor(anchor, cv2.COLOR_RGB2BGR))
                print(f"[anchor] start {s} -> {self._path(s)}")
        self._prepared = True

    def anchor_for_start(self, start):
        from PIL import Image
        self.prepare()
        p = self._path(start)
        if not os.path.exists(p):
            raise FileNotFoundError(f"RoMa anchor missing for start={start}: {p}")
        return Image.open(p).convert("RGB")


def get_anchor(backend, **kw):
    if backend == "assets":
        return AssetsAnchor(kw["assets_dir"], anchors_subdir=kw.get("anchors_subdir", "anchors"))
    if backend == "roma":
        return RomaAnchor(kw["frames_dir"], kw["ref0_path"], kw["work_dir"], kw["segment_starts"],
                          device=kw.get("device", "cuda"))
    raise ValueError(f"unknown anchor backend: {backend!r}")
