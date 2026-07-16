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
        return Image.open(self.anchor_path_for_start(start)).convert("RGB")

    def anchor_path_for_start(self, start):
        self.prepare()
        p = self._path(start)
        if not os.path.exists(p):
            raise FileNotFoundError(f"RoMa anchor missing for start={start}: {p}")
        return p


class MVInpainterAnchor:
    """Per-chunk anchors via a MVInpainter single-mode pass (clean at ALL viewpoints,
    incl. steep top-down — unlike RoMa's 2D warp of ref0, which shears large views).

    Runs ONE wide-baseline single-mode group over `n_views` evenly-sampled frames -> a
    sparse set of GENERATED anchors (cached); anchor_path_for_start(s) returns the
    sampled anchor nearest frame s. Same interface as RomaAnchor, so any generator's
    per-chunk loop can consume it (the MVInpainter-anchor axis of the anchor×generator
    matrix). Reuses components.mvinpainter.generate() UNCHANGED for the pass."""

    def __init__(self, frames_dir, ref0_path, mask_dir, work_dir, *,
                 n_views=24, prompt="", name="mvi_anchor"):
        self.frames_dir = frames_dir
        self.ref0_path = ref0_path
        self.mask_dir = mask_dir
        self.work_dir = work_dir
        self.n_views = n_views
        self.prompt = prompt
        self.name = name
        self.run_dir = os.path.join(work_dir, "mvi_anchor_run")
        self._map = None                     # sampled frame index -> anchor path

    def prepare(self):
        if self._map is not None:
            return
        frames_out = os.path.join(self.run_dir, "frames")
        if not glob.glob(f"{frames_out}/frame_*.png"):
            from components import mvinpainter
            mvinpainter.generate(self.frames_dir, self.mask_dir, self.ref0_path, self.run_dir,
                                 mode="single", nframe=self.n_views, prompt=self.prompt,
                                 smooth=False, name=self.name)
        self._map = {int(os.path.basename(p).split("_")[1].split(".")[0]): p
                     for p in sorted(glob.glob(f"{frames_out}/frame_*.png"))}
        if not self._map:
            raise FileNotFoundError(f"MVInpainter anchor pass produced no frames in {frames_out}")
        print(f"[anchor] MVInpainter anchors: {len(self._map)} views at "
              f"{sorted(self._map)[:3]}..{sorted(self._map)[-1]}")

    def anchor_path_for_start(self, start):
        self.prepare()
        return self._map[min(self._map, key=lambda k: abs(k - start))]   # nearest sampled view

    def anchor_for_start(self, start):
        from PIL import Image
        return Image.open(self.anchor_path_for_start(start)).convert("RGB")


def get_anchor(backend, **kw):
    if backend == "assets":
        return AssetsAnchor(kw["assets_dir"], anchors_subdir=kw.get("anchors_subdir", "anchors"))
    if backend == "roma":
        return RomaAnchor(kw["frames_dir"], kw["ref0_path"], kw["work_dir"], kw["segment_starts"],
                          device=kw.get("device", "cuda"))
    if backend == "mvinpainter":
        return MVInpainterAnchor(kw["frames_dir"], kw["ref0_path"], kw["mask_dir"], kw["work_dir"],
                                 n_views=kw.get("n_views", 24), prompt=kw.get("prompt", ""),
                                 name=kw.get("name", "mvi_anchor"))
    raise ValueError(f"unknown anchor backend: {backend!r}")
