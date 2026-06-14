"""Edit-mask component — the per-frame region a candidate is allowed to repaint.

GENERIC, cross-candidate artifact (VideoPainter / removal / inpaint / future all
consume it): the frame-0 (target ∪ source) region, RoMa-warped to every frame so
it follows the object. No VideoPainter knowledge here.

Backends:
  - roma:   build region0 from SAM3 masks (target on ref0, source on frame0),
            take its bbox + dilate, then warp per frame.  (any video)
  - assets: load a prepared per-frame mask folder.

NOTE (kept verbatim from the old anchors.RomaAnchors for behaviour-parity): the
frame-0 region is the BOUNDING BOX of (target ∪ source) + `dilate` padding. The
irregular-hull / scale-relative-dilate improvement is a future change to THIS file
(a knob for the tuning agent), deliberately not done in the decoupling refactor.
"""
import os
import glob
import numpy as np
import cv2

from components import roma_warp


def _frames(frames_dir):
    return sorted(glob.glob(f"{frames_dir}/frame_*.png"))


class AssetsEditMask:
    """Load a prepared per-frame edit-mask folder (<assets_dir>/<masks_subdir>/)."""
    def __init__(self, assets_dir, masks_subdir="banana_masks"):
        md = os.path.join(assets_dir, masks_subdir)
        self._mask_dir = md if os.path.isdir(md) else None

    @property
    def mask_dir(self):
        return self._mask_dir


class RomaEditMask:
    """Build frame-0 (target∪source) bbox+dilate region, RoMa-warp it per frame."""
    def __init__(self, frames_dir, ref0_path, target_word, work_dir, *,
                 device="cuda", ref0_mask_path=None, source_word=None, dilate=12):
        self.frames_dir = frames_dir
        self.ref0_path = ref0_path
        self.target_word = target_word
        self.source_word = source_word
        self.work_dir = work_dir
        self.device = device
        self.ref0_mask_path = ref0_mask_path
        self.dilate = dilate
        self._mask_dir = os.path.join(work_dir, "masks")
        self._prepared = False

    def _build_region0(self):
        """frame-0 edit region = bbox of (SAM3 target on ref0 ∪ SAM3 source on frame0) + pad."""
        import components.sam3_mask as sam3_mask
        frame0 = _frames(self.frames_dir)[0]
        ref0_rgb = cv2.cvtColor(cv2.imread(self.ref0_path), cv2.COLOR_BGR2RGB)
        h0, w0 = ref0_rgb.shape[:2]
        predictor = sam3_mask.build_predictor()
        if self.ref0_mask_path:
            target_mask0 = cv2.imread(self.ref0_mask_path, 0)
        else:
            target_mask0 = sam3_mask.mask_image(predictor, self.ref0_path,
                                                self.target_word, work_dir=self.work_dir, tag="newobj")
        source_mask0 = sam3_mask.mask_image(predictor, frame0, self.source_word,
                                            work_dir=self.work_dir, tag="srcobj")
        target_mask0 = cv2.resize(target_mask0, (w0, h0), interpolation=cv2.INTER_NEAREST)
        source_mask0 = cv2.resize(source_mask0, (w0, h0), interpolation=cv2.INTER_NEAREST)
        u = (target_mask0 > 127) | (source_mask0 > 127)
        ys, xs = np.where(u)
        pad = max(0, int(self.dilate))
        x0, x1 = max(0, xs.min() - pad), min(w0 - 1, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(h0 - 1, ys.max() + pad)
        region0 = np.zeros((h0, w0), np.uint8)
        region0[y0:y1 + 1, x0:x1 + 1] = 255
        return region0

    def prepare(self):
        if self._prepared:
            return
        if glob.glob(os.path.join(self._mask_dir, "frame_*.png")):
            print(f"[edit_mask] reusing cached masks in {self._mask_dir}")
            self._prepared = True
            return
        region0 = self._build_region0()
        frame_paths = _frames(self.frames_dir)
        hf, wf = cv2.imread(frame_paths[0]).shape[:2]
        reg0 = cv2.resize(region0, (wf, hf), interpolation=cv2.INTER_NEAREST)
        os.makedirs(self._mask_dir, exist_ok=True)
        f0 = frame_paths[0]
        with roma_warp.roma_float32():
            for k, fp in enumerate(frame_paths):
                name = os.path.basename(fp)
                if k == 0:
                    mask_k = (reg0 > 127).astype(np.uint8)
                else:
                    gridA, S = roma_warp.match(f0, fp, device=self.device)
                    mask_k = roma_warp.warp_binary(reg0, gridA, S, (hf, wf), device=self.device)
                cv2.imwrite(f"{self._mask_dir}/{name}", (mask_k * 255).astype(np.uint8))
                if k % 25 == 0:
                    print(f"[edit_mask] frame {k}/{len(frame_paths)}")
        print(f"[edit_mask] per-frame edit masks -> {self._mask_dir}")
        self._prepared = True

    @property
    def mask_dir(self):
        self.prepare()
        return self._mask_dir


def union_masks(source_dir, target_dir, out_dir):
    """Per-frame OR of two mask dirs (keyed by frame name), at source resolution.
    Used by the assets/sam3 path (mask_mode=union) to combine SAM3 source ∪ target."""
    os.makedirs(out_dir, exist_ok=True)
    names = [os.path.basename(p) for p in sorted(glob.glob(f"{source_dir}/frame_*.png"))]
    for n in names:
        a = cv2.imread(f"{source_dir}/{n}", 0)
        m = (a > 127).astype(np.uint8)
        tp = f"{target_dir}/{n}" if target_dir else None
        if tp and os.path.exists(tp):
            b = cv2.imread(tp, 0)
            if b.shape != a.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = ((m > 0) | (b > 127)).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/{n}", (m * 255).astype(np.uint8))
    print(f"[edit_mask] union edit-region masks -> {out_dir} ({len(names)} frames)")
    return out_dir


def get_edit_mask(backend, **kw):
    if backend == "assets":
        return AssetsEditMask(kw["assets_dir"], masks_subdir=kw.get("masks_subdir", "banana_masks"))
    if backend == "roma":
        return RomaEditMask(kw["frames_dir"], kw["ref0_path"], kw["target_word"], kw["work_dir"],
                            device=kw.get("device", "cuda"), ref0_mask_path=kw.get("ref0_mask_path"),
                            source_word=kw.get("source_word"), dilate=kw.get("dilate", 12))
    raise ValueError(f"unknown edit_mask backend: {backend!r}")
