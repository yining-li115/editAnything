"""MVInpainter route — an ALTERNATIVE generator to VideoPainter.

MVInpainter (ewrfcas/MVInpainter, SD1.5-inpaint + AnimateDiff) is a multi-view
IMAGE inpainter, not a video model: it inpaints the new object consistently across
a set of <=~24 wide-baseline views that all share ONE reference (ref0). It has no
temporal layer, so as a dense video it flickers and drifts at large viewpoints
(see docs) — kept here as a decoupled candidate the judge/user can pick against
VideoPainter.

It needs its OWN env (python 3.8, mmflow/mmcv + SD1.5 + AnimateDiff), incompatible
with editanything, so it runs as a SUBPROCESS over files — same pattern as ROSE.

Pipeline (inputs are identical to VideoPainter: frames + per-frame edit masks + ref0):
  crop the object band -> split frames into `reference_split` interleaved groups
  (the authors' long-video grouping, done as input SCENES so the repo stays stock)
  -> (mvinpainter env) test_nvs.py -> 512^2 per-view crops
  -> uncrop + feather-composite back onto the original frames
  -> (optional) motion-compensated temporal smoothing to reduce inter-group flicker.
"""
import os
import glob
import math
import shutil
import subprocess

import cv2
import numpy as np

from contracts import layout

# Separate env + repo (sibling of editAnything). Override via env vars, like ROSE.
MVI_ROOT = os.environ.get("MVINPAINTER_ROOT", os.path.join(os.path.dirname(layout.ROOT), "MVInpainter"))
MVI_PYTHON = os.environ.get("MVINPAINTER_PYTHON", "/venv/mvinpainter/bin/python")
MVI_MODEL = os.environ.get("MVINPAINTER_MODEL", "check_points/mvinpainter_o_512")
DEFAULT_STEPS = 50

def _frames(d):
    return sorted(glob.glob(f"{d}/frame_*.png"))


def crop_band(mask_paths, pad=8):
    """Full-width band [y0:y1] that contains the object across ALL frames (+pad).
    MVInpainter resizes to a square internally; feeding this band (vs the full
    portrait) keeps the object big and minimises aspect distortion."""
    y0, y1, H, W = 10**9, -1, None, None
    for p in mask_paths:
        m = cv2.imread(p, 0)
        H, W = m.shape
        ys, _ = np.where(m > 127)
        if len(ys):
            y0 = min(y0, int(ys.min()))
            y1 = max(y1, int(ys.max()))
    if y1 < 0:
        raise ValueError("all masks empty — cannot locate object band")
    return max(0, y0 - pad), min(H, y1 + pad), 0, W


def _temporal_smooth(frames_out, window=((-1, 0.5), (1, 0.5), (-2, 0.25), (2, 0.25))):
    """Motion-compensated temporal smoothing: warp neighbour frames onto each frame
    via optical flow, then weighted-average. Reduces MVInpainter's inter-view flicker
    while preserving motion (the object stays put; high-freq jitter is averaged out)."""
    fs = _frames(frames_out)
    imgs = [cv2.imread(f) for f in fs]
    grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    h, w = imgs[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    N = len(imgs)
    for t in range(N):
        acc = imgs[t].astype(np.float32)
        wsum = 1.0
        for dt, wt in window:
            s = t + dt
            if 0 <= s < N:
                flow = cv2.calcOpticalFlowFarneback(grays[t], grays[s], None, 0.5, 3, 21, 3, 5, 1.2, 0)
                mapx = (gx + flow[..., 0]).astype(np.float32)
                mapy = (gy + flow[..., 1]).astype(np.float32)
                warped = cv2.remap(imgs[s], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                acc += warped.astype(np.float32) * wt
                wsum += wt
        cv2.imwrite(fs[t], (acc / wsum).astype(np.uint8))
    print(f"[mvinpainter] temporal-smoothed {N} frames")


def _encode(frame_paths, out_mp4, fps=8):
    """Encode an ordered list of png frames -> mp4 (symlink to a contiguous seq so
    ffmpeg's %d works regardless of source names)."""
    tmp = out_mp4 + "_seq"
    os.makedirs(tmp, exist_ok=True)
    for k, p in enumerate(frame_paths, 1):
        lk = os.path.join(tmp, f"a_{k:05d}.png")
        if not os.path.lexists(lk):
            os.symlink(os.path.abspath(p), lk)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", f"{tmp}/a_%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4],
                   check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def generate(frames_dir, mask_dir, ref0_path, out_dir, *, nframe=24, mode="single",
             reference_split=None, prompt="", res=512, smooth=True, feather=3.0, cfg=7.5,
             anchors_fps=8, steps=DEFAULT_STEPS, name="mvi_run", env=None):
    """Run MVInpainter as the generator; write full-frame results to {out_dir}/frames.

    mode:
      - "single" (default): sample `nframe` frames evenly across the clip (step =
        round(#frames / nframe)), run them as ONE consistent group, composite, and
        concatenate into a sparse mp4. This is the cleanest MVInpainter output (one
        wide-baseline group, no inter-group jitter) — the `mvinpainter_step*_ANCHORS`
        style. Output is SPARSE (nframe frames), not the full clip length.
      - "dense": interleave the whole clip into `reference_split` groups (frames[j::N],
        the authors' long-video grouping as input scenes), tile back to every frame,
        then motion-compensated temporal-smooth. Full length but flickers.
    Returns the frames dir.
    """
    frames = _frames(frames_dir)
    masks = _frames(mask_dir)
    n = min(len(frames), len(masks))
    if n == 0:
        raise FileNotFoundError(f"no frames/masks in {frames_dir} / {mask_dir}")
    y0, y1, x0, x1 = crop_band(masks)
    CH, CW = y1 - y0, x1 - x0

    def crop(im):
        return im[y0:y1, x0:x1]

    # ---- group assignment ----
    if mode == "single":
        step = max(1, round(n / nframe))
        groups = {"w00": list(range(0, n, step))[:nframe]}
    elif mode == "dense":
        N = reference_split or max(1, math.ceil(n / nframe))
        if math.ceil(n / N) > 32:
            raise ValueError(f"group size {math.ceil(n/N)} > 32 (MVInpainter PE cap); raise reference_split")
        groups = {f"w{j:02d}": list(range(j, n, N))[:nframe] for j in range(N)}
    else:
        raise ValueError(f"mode must be single|dense, got {mode!r}")

    # ---- build input scenes (crop the object band; each scene shares ref0) ----
    scenes = os.path.join(out_dir, "_mvi_scenes")
    if os.path.exists(scenes):
        shutil.rmtree(scenes)
    ref0c = crop(cv2.imread(ref0_path))
    covered = set()
    grp_nframe = 0
    for gname, idxs in groups.items():
        if len(idxs) < 2:
            continue
        grp_nframe = max(grp_nframe, len(idxs))
        sd = os.path.join(scenes, gname)
        for sub in ("removal", "warp_masks", "obj_bbox"):
            os.makedirs(f"{sd}/{sub}", exist_ok=True)
        for i in idxs:
            nm = os.path.basename(frames[i])
            cv2.imwrite(f"{sd}/removal/{nm}", crop(cv2.imread(frames[i])))
            cv2.imwrite(f"{sd}/warp_masks/{nm}", crop(cv2.imread(masks[i])))
            covered.add(i)
        cv2.imwrite(f"{sd}/obj_bbox/0000.png", ref0c)

    # ---- stock test_nvs in the mvinpainter env (subprocess) ----
    cmd = [MVI_PYTHON, "test_nvs.py", "--load_path", MVI_MODEL,
           "--dataset_root", os.path.abspath(scenes), "--output_path", name,
           "--edited_index", "0", "--resume_from_checkpoint", "best", "--val_cfg", str(cfg),
           "--img_height", str(res), "--img_width", str(res), "--sampling_interval", "1.0",
           "--nframe", str(grp_nframe), "--prompt", prompt, "--limit_frame", str(grp_nframe),
           "--save_images", "--inference_steps", str(steps)]
    print(f"[mvinpainter] mode={mode}: {len(groups)} group(s) x{grp_nframe}, band y[{y0}:{y1}] f"steps={steps} — cwd={MVI_ROOT}  ")
    subprocess.run(cmd, check=True, cwd=MVI_ROOT, env={**os.environ, **(env or {})})

    cand = sorted(glob.glob(os.path.join(MVI_ROOT, "outputs", name + "*")))
    if not cand:
        raise FileNotFoundError(f"MVInpainter produced no output under {MVI_ROOT}/outputs/{name}*")
    mvmap = {os.path.basename(p): p for p in glob.glob(os.path.join(cand[-1], "w*", "frame_*.png"))}

    # ---- uncrop + feather-composite back onto the original frames ----
    frames_out = os.path.join(out_dir, "frames")
    os.makedirs(frames_out, exist_ok=True)
    written, missing = [], 0
    scan = range(n) if mode == "dense" else sorted(covered)   # single = only sampled frames
    for i in scan:
        nm = os.path.basename(frames[i])
        orig = cv2.imread(frames[i])
        if nm in mvmap:
            mv = cv2.resize(cv2.imread(mvmap[nm]), (CW, CH), interpolation=cv2.INTER_CUBIC)
            canvas = orig.copy()
            canvas[y0:y1, x0:x1] = mv
            m = (cv2.imread(masks[i], 0) > 127).astype(np.float32)
            m = cv2.GaussianBlur(m, (0, 0), feather)[..., None]
            out = (m * canvas.astype(np.float32) + (1 - m) * orig.astype(np.float32)).astype(np.uint8)
        elif mode == "dense":
            out, missing = orig, missing + 1        # uncovered tail -> source
        else:
            continue
        p = os.path.join(frames_out, nm)
        cv2.imwrite(p, out)
        written.append(p)
    if missing:
        print(f"[mvinpainter] WARNING: {missing} tail frame(s) not covered — left as source")
    print(f"[mvinpainter] composited {len(written)} frame(s) -> {frames_out}")

    if mode == "single":
        # the sparse single-group set IS the output (mvinpainter_step*_ANCHORS style)
        _encode(sorted(written), os.path.join(out_dir, "mvi_anchors.mp4"), fps=anchors_fps)
        print(f"[mvinpainter] single-group anchors -> mvi_anchors.mp4 ({len(written)} frames @{anchors_fps}fps)")
    else:
        if smooth:
            _temporal_smooth(frames_out)
    return frames_out


def generate_chunked(frames_dir, mask_dir, anchor_path_for_start, out_dir, *,
                     segment_starts, chunk=20, prompt="", res=512, cfg=7.5, steps=DEFAULT_STEPS, env=None):
    """Multi-chunk fill with MVInpainter as the per-chunk generator.

    Same shape as videopainter's per-segment reanchor loop, but MVInpainter fills each
    chunk. Each chunk = `chunk` consecutive frames starting at `start`, generated as ONE
    single-mode group conditioned on THAT chunk's own anchor (anchor_path_for_start(start)
    — e.g. a clean MVInpainter single-mode anchor). Narrow within-chunk baseline keeps the
    object from deforming; per-chunk reanchoring stops it dissolving across the clip.

    anchor_path_for_start: callable start -> path of that chunk's anchor image (full frame).
    Returns the assembled frames dir {out_dir}/frames.
    """
    frames = _frames(frames_dir)
    n = len(frames)
    stems = [os.path.basename(f) for f in frames]
    work = os.path.join(out_dir, "_chunks")
    final = os.path.join(out_dir, "frames")
    os.makedirs(final, exist_ok=True)
    for start in segment_starts:
        idxs = list(range(start, min(start + chunk, n)))
        if len(idxs) < 2:
            continue
        anchor = anchor_path_for_start(start)
        cdir = os.path.join(work, f"chunk_{start:05d}")
        cf, cm = os.path.join(cdir, "frames"), os.path.join(cdir, "masks")
        os.makedirs(cf, exist_ok=True)
        os.makedirs(cm, exist_ok=True)
        for i in idxs:
            for src, dst in ((os.path.join(frames_dir, stems[i]), os.path.join(cf, stems[i])),
                             (os.path.join(mask_dir, stems[i]),   os.path.join(cm, stems[i]))):
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(src), dst)
        print(f"[mvinpainter] chunk {start}..{idxs[-1]}  anchor={os.path.basename(str(anchor))}", flush=True)
        fout = generate(cf, cm, str(anchor), cdir, mode="single", nframe=len(idxs),
                        prompt=prompt, smooth=False, res=res, cfg=cfg, steps=steps,
                        name=f"chunk_{start:05d}", env=env)
        for p in sorted(glob.glob(f"{fout}/frame_*.png")):
            shutil.copy(p, os.path.join(final, os.path.basename(p)))
    got = len(glob.glob(f"{final}/frame_*.png"))
    print(f"[mvinpainter] chunked fill -> {final} ({got}/{n} frames)", flush=True)
    return final


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MVInpainter generator (cross-env subprocess)")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--mask_dir", required=True, help="per-frame edit masks (white = new object region)")
    ap.add_argument("--ref0", required=True, help="frame-0 reference: new object on frame 0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="single", choices=["single", "dense"])
    ap.add_argument("--nframe", type=int, default=24)
    ap.add_argument("--reference_split", type=int, default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                help="diffusion inference steps (test_nvs.py's --inference_step; default 50)")
    ap.add_argument("--no_smooth", action="store_true")
    args = ap.parse_args()
    generate(args.frames_dir, args.mask_dir, args.ref0, args.out_dir, mode=args.mode,
             nframe=args.nframe, reference_split=args.reference_split,
             prompt=args.prompt, smooth=not args.no_smooth, steps=args.steps)
