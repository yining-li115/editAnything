"""End-to-end video object replacement (SAM3 + Gemini + VideoPainter), decoupled.

One command: video + source object + target description -> final portrait video.
Everything lands under editAnything/outputs/<name>/.

Stages (each cached; --resume reuses existing outputs):
  1. extract     video -> frames_src/frame_%05d.png        (native size/fps)
  2. source mask SAM3(source) -> mask_src/                 [sam3_track]
  3. edit region mask = source-mask  (∪ target-mask from anchors) -> mask/
  4. anchors     per-segment clean anchors + target masks  [anchors backend]
  5. generate    VideoPainter multi-chunk reanchor -> gen/frames/   [generate]
  6. composite   feather object onto fixed plate (orig frames) -> composite/  [composite]
  7. encode      -> final.mp4 (portrait) (+ _interp.mp4)            [encode]

Phase A (any video, gap stubbed): the anchor/target-mask propagation is not yet
general — pass --backend assets --assets <dir> with prepared anchors+masks (e.g.
exp3_bundle/inputs) to reproduce. The rest of the pipeline is fully general.

What we use from VideoPainter: ONLY its diffusers fork + the 3 checkpoints, via
generate. SAM3 and Gemini are separate stages producing files.
"""
import os
import glob
import shutil
import subprocess
import argparse
import yaml
import numpy as np
import cv2

import anchors
import composite as composite_step
import encode as encode_step

HERE = os.path.dirname(os.path.abspath(__file__))   # src/
ROOT = os.path.dirname(HERE)                          # repo root (parent of src/)
CLIP = 49


def _has_frames(d):
    return os.path.isdir(d) and len(glob.glob(f"{d}/frame_*.png")) > 0


def extract_frames(video, out_dir, resume=True):
    if resume and _has_frames(out_dir):
        print(f"[pipeline] reuse frames {out_dir}")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    "-start_number", "1", f"{out_dir}/frame_%05d.png"], check=True)
    print(f"[pipeline] extracted {len(glob.glob(f'{out_dir}/frame_*.png'))} frames -> {out_dir}")
    return out_dir


def video_meta(frames_dir):
    paths = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    h, w = cv2.imread(paths[0]).shape[:2]
    return len(paths), (w, h)


def default_segments(n, clip=CLIP, step=48):
    starts = list(range(0, max(1, n - clip + 1), step))
    tail = n - clip
    if tail > starts[-1]:
        starts.append(tail)
    return starts


def union_masks(source_dir, target_dir, out_dir):
    """Per-frame OR of two mask dirs (keyed by frame name), at source resolution."""
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
    print(f"[pipeline] edit-region masks -> {out_dir} ({len(names)} frames)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="End-to-end video object replacement")
    ap.add_argument("--config", help="YAML config file; its values become defaults, CLI flags override")
    # inputs (required, from config or CLI)
    ap.add_argument("--video", help="input video (or use --frames_dir)")
    ap.add_argument("--frames_dir", help="pre-extracted frames frame_00001.png... (skip extraction)")
    ap.add_argument("--source", help="OLD object to remove — SAM3 noun (e.g. 'cup')")
    ap.add_argument("--target", help="NEW object description (e.g. 'a ripe yellow banana')")
    ap.add_argument("--prompt", help="global generation prompt for the edited video")
    ap.add_argument("--name", help="run name -> outputs/<name>/")
    # anchors / masks
    ap.add_argument("--backend", default="assets", choices=["assets", "roma"])
    ap.add_argument("--assets", help="assets dir for backend=assets (anchors/ + banana_masks/)")
    # backend=roma: warp a frame-0 reference (Gemini edit) to every viewpoint
    ap.add_argument("--ref0", help="(roma) frame-0 reference image: new object placed on frame 0")
    ap.add_argument("--ref0_mask", default=None, help="(roma) optional frame-0 object mask (else SAM3)")
    ap.add_argument("--target_word", default=None, help="(roma) noun for SAM3 to segment the new "
                    "object on ref0 (default: --target)")
    ap.add_argument("--mask_mode", default="union", choices=["union", "source", "target"],
                    help="edit region: SAM3∪target (default), SAM3 only, or provider target only")
    # generation params (validated exp3 reanchor)
    ap.add_argument("--segment_starts", default=None, help="comma list; default auto from frame count")
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    # output
    ap.add_argument("--out_root", default=None,
                    help="base dir for outputs (default: the repo dir -> editAnything/outputs/<name>)")
    ap.add_argument("--out_size", default=None, help="final WxH (default = native frame size)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--interpolate", action="store_true",
                    help="RIFE de-spike: replace each segment-boundary anchor frame with "
                         "RIFE(neighbour, neighbour) to remove the reanchor pop (native fps)")
    # model paths
    ap.add_argument("--model_path", default=f"{ROOT}/VideoPainter/ckpt/CogVideoX-5b-I2V")
    ap.add_argument("--branch", default=f"{ROOT}/VideoPainter/ckpt/VideoPainter/checkpoints/branch")
    ap.add_argument("--id_lora", default=f"{ROOT}/VideoPainter/ckpt/VideoPainterID/checkpoints")
    # flow control
    ap.add_argument("--no_composite", action="store_true",
                    help="skip seamless composite; encode straight from gen frames "
                         "(keeps natural object grounding, but background shifts at segment boundaries)")
    ap.add_argument("--resume", action="store_true", help="reuse existing stage outputs")
    ap.add_argument("--stop_after", default=None,
                    choices=["extract", "mask", "generate", "composite", "encode"])

    # Config file: load YAML and inject as argparse defaults (CLI flags still override).
    cfg_path = ap.parse_known_args()[0].config
    if cfg_path:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in ap._actions}
        unknown = set(cfg) - known
        if unknown:
            print(f"[pipeline] WARNING: ignoring unknown config keys: {sorted(unknown)}")
        ap.set_defaults(**{k: v for k, v in cfg.items() if k in known and v is not None})
    args = ap.parse_args()

    # required (from config or CLI)
    missing = [k for k in ("source", "target", "prompt", "name") if not getattr(args, k)]
    if missing:
        ap.error(f"missing required: {missing} (set in --config or on the CLI)")
    assert args.video or args.frames_dir, "need 'video' or 'frames_dir' (config or CLI)"

    out_root = args.out_root or ROOT                    # default: editAnything/outputs/<name>
    run = os.path.join(out_root, "outputs", args.name)
    os.makedirs(run, exist_ok=True)
    d_frames = os.path.join(run, "frames_src")
    d_mask_src = os.path.join(run, "mask_src")
    d_mask = os.path.join(run, "mask")
    d_gen = os.path.join(run, "gen")
    d_comp = os.path.join(run, "composite")

    # 1. frames
    if args.frames_dir:
        d_frames = args.frames_dir
    else:
        assert args.video, "need --video or --frames_dir"
        extract_frames(args.video, d_frames, resume=args.resume)
    n_frames, native_size = video_meta(d_frames)
    out_size = (tuple(int(v) for v in args.out_size.lower().split("x"))
                if args.out_size else native_size)
    # Multi-chunk only kicks in past the model's single-pass limit (CLIP=49).
    if args.segment_starts:
        ss = args.segment_starts                      # may be a list (config) or "0,48" (CLI)
        starts = [int(x) for x in (ss if isinstance(ss, (list, tuple)) else str(ss).split(","))]
    else:
        starts = default_segments(n_frames)
    mode = "single-pass" if n_frames <= CLIP else f"multi-chunk ({len(starts)} segments)"
    print(f"[pipeline] {n_frames} frames, native={native_size}, out={out_size}, "
          f"mode={mode}, segments={starts}")
    if args.stop_after == "extract":
        return

    # 2-4. anchors + edit-region masks
    if args.backend == "assets":
        provider = anchors.get_anchor_provider("assets", assets_dir=args.assets)
        target_mask_dir = getattr(provider, "target_mask_dir", None)
    else:  # roma: RoMa-propagate frame-0 -> per-frame edit masks + per-segment anchors
        assert args.ref0, "backend=roma needs --ref0 (frame-0 reference image)"
        provider = anchors.get_anchor_provider(
            "roma", frames_dir=d_frames, ref0_path=args.ref0,
            target_word=(args.target_word or args.target), source_word=args.source,
            ref0_mask_path=args.ref0_mask, work_dir=os.path.join(run, "roma"),
            segment_starts=starts, dilate=args.dilate)
        target_mask_dir = provider.target_mask_dir   # triggers RoMa: edit masks + anchors

    if args.backend == "roma":
        # edit region = the RoMa-propagated frame-0 (target ∪ source) bbox region
        # (big enough for the new object); the per-frame SAM3 stage isn't needed here.
        d_mask = target_mask_dir
    elif args.mask_mode == "target":
        assert target_mask_dir, "mask_mode=target but provider has no target masks"
        d_mask = target_mask_dir
    else:
        if not (args.resume and _has_frames(d_mask_src)):
            import sam3_track
            sam3_track.track(sam3_track.build_predictor(), d_frames, args.source, d_mask_src)
        if args.mask_mode == "source":
            d_mask = d_mask_src
        else:  # union
            union_masks(d_mask_src, target_mask_dir, d_mask)
    if args.stop_after == "mask":
        return

    # 5. generate (load models once, loop segments)
    if not (args.resume and _has_frames(os.path.join(d_gen, "frames"))):
        import generate
        pipe = generate.load_pipeline(args.model_path, args.branch, args.id_lora)
        generate.generate(pipe, d_frames, d_mask, provider.anchor_for_start, d_gen,
                             segment_starts=starts, total=CLIP, prompt=args.prompt,
                             dilate=args.dilate, steps=args.steps, guidance=args.guidance,
                             seed=args.seed)
    if args.stop_after == "generate":
        return

    # 6. composite onto fixed plate (= original frames). Default OFF for RoMa anchors:
    # the warped anchor's hand/region, pasted within the mask, ghosts against the
    # original hand. gen frames already keep the original background (replace_gt),
    # so composite is unneeded there and would only re-introduce the ghost.
    skip_composite = args.no_composite or args.backend == "roma"
    if skip_composite:
        src_dir = os.path.join(d_gen, "frames")
        print(f"[pipeline] composite skipped (backend={args.backend}); using gen frames")
    else:
        composite_step.composite(d_frames, os.path.join(d_gen, "frames"), d_mask, d_comp,
                               total=n_frames)
        src_dir = d_comp
    if args.stop_after == "composite":
        return

    # 7. encode portrait. --interpolate = RIFE anchor de-spike (replace each segment-
    # boundary anchor frame with RIFE(neighbour, neighbour); native fps unchanged).
    final = os.path.join(run, "final.mp4")
    anchor_frames = [s + 1 for s in starts if s > 0]   # 1-indexed boundary frames
    if args.interpolate and anchor_frames:
        src_dir = encode_step.despike_anchors(src_dir, os.path.join(run, "despike_frames"),
                                              anchor_frames)
    encode_step.encode(src_dir, final, out_size, fps=args.fps)
    print(f"[pipeline] DONE -> {final}")


if __name__ == "__main__":
    main()
