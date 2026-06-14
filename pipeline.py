"""End-to-end video object replacement (SAM3 + Gemini + VideoPainter), decoupled.

Thin local runner: fixed-chain orchestrator that calls the `components` directly
(no MCP). Each stage is a standalone component; this file only wires them.

Stages (each cached; --resume reuses existing outputs):
  1. extract     video -> frames_src/                          [components.extract]
  2. edit mask   per-frame region to repaint                   [components.edit_mask]
  3. anchors     per-segment clean anchors (VideoPainter)      [components.anchor]
  4. (sam3 src)  source mask, for assets/union mask modes      [components.sam3_mask]
  5. generate    VideoPainter multi-chunk reanchor -> gen/     [components.videopainter]
  6. composite   feather object onto fixed plate               [components.composite]
  7. encode      -> final.mp4 (+ RIFE anchor de-spike)         [components.encode]
"""
import os
import glob
import argparse
import yaml

from components import extract
from components import edit_mask as edit_mask_mod
from components import anchor as anchor_mod
from components import composite as composite_step
from components import encode as encode_step
from components import videopainter
from contracts import layout

CLIP = videopainter.CLIP
_VP = layout.MODELS["videopainter"]


def _limit_frames(src_dir, n, dst_dir):
    """Return a dir of symlinks to the first n frame_*.png of src_dir (for --max_frames
    when frames are pre-extracted; the --video path limits at ffmpeg decode instead)."""
    os.makedirs(dst_dir, exist_ok=True)
    for p in sorted(glob.glob(f"{src_dir}/frame_*.png"))[:n]:
        link = os.path.join(dst_dir, os.path.basename(p))
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(p), link)
    print(f"[pipeline] limited to first {n} frames -> {dst_dir}")
    return dst_dir


def main():
    ap = argparse.ArgumentParser(description="End-to-end video object replacement")
    ap.add_argument("--config", help="YAML config file; its values become defaults, CLI flags override")
    # inputs
    ap.add_argument("--video", help="input video (or use --frames_dir)")
    ap.add_argument("--frames_dir", help="pre-extracted frames frame_00001.png... (skip extraction)")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="process only the first N frames (quick tests, any length)")
    ap.add_argument("--source", help="OLD object to remove — SAM3 noun (e.g. 'cup')")
    ap.add_argument("--target", help="NEW object description (e.g. 'a ripe yellow banana')")
    ap.add_argument("--prompt", help="global generation prompt for the edited video")
    ap.add_argument("--name", help="run name -> outputs/<name>/")
    # anchors / masks
    ap.add_argument("--backend", default="assets", choices=["assets", "roma"])
    ap.add_argument("--assets", help="assets dir for backend=assets (anchors/ + banana_masks/)")
    ap.add_argument("--ref0", help="(roma) frame-0 reference image: new object placed on frame 0")
    ap.add_argument("--ref0_mask", default=None, help="(roma) optional frame-0 object mask (else SAM3)")
    ap.add_argument("--target_word", default=None, help="(roma) noun for SAM3 to segment the new "
                    "object on ref0 (default: --target)")
    ap.add_argument("--mask_mode", default="union", choices=["union", "source", "target"],
                    help="edit region: SAM3∪target (default), SAM3 only, or provider target only")
    # generation params (validated exp3 reanchor)
    ap.add_argument("--segment_starts", default=None, help="comma list; default auto from frame count")
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--region_shape", default="rect", choices=["bbox", "rect", "hull"],
                    help="frame-0 edit region from target∪source: rect=rotated quad (default), "
                         "hull=tight irregular silhouette, bbox=rigid rectangle")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    # output
    ap.add_argument("--out_root", default=None, help="base dir for outputs (default: repo -> outputs/<name>)")
    ap.add_argument("--out_size", default=None, help="final WxH (default = native frame size)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--interpolate", action="store_true",
                    help="RIFE de-spike: replace each segment-boundary anchor frame with "
                         "RIFE(neighbour, neighbour) to remove the reanchor pop (native fps)")
    # model paths (defaults from the contracts registry)
    ap.add_argument("--model_path", default=_VP["model_path"])
    ap.add_argument("--branch", default=_VP["branch"])
    ap.add_argument("--id_lora", default=_VP["id_lora"])
    # flow control
    ap.add_argument("--no_composite", action="store_true",
                    help="skip seamless composite; encode straight from gen frames")
    ap.add_argument("--removal", default="none", choices=["none", "rose"],
                    help="rose = ROSE clean-plate removal of the source object + its shadow, "
                         "then composite the new object onto that clean plate (post-processing)")
    ap.add_argument("--resume", action="store_true", help="reuse existing stage outputs")
    ap.add_argument("--stop_after", default=None,
                    choices=["extract", "mask", "generate", "removal", "composite", "encode"])

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

    missing = [k for k in ("source", "target", "prompt", "name") if not getattr(args, k)]
    if missing:
        ap.error(f"missing required: {missing} (set in --config or on the CLI)")
    assert args.video or args.frames_dir, "need 'video' or 'frames_dir' (config or CLI)"

    rp = layout.RunPaths(args.name, out_root=args.out_root)
    os.makedirs(rp.root, exist_ok=True)
    d_frames = rp.frames_src

    # 1. frames
    if args.frames_dir:
        d_frames = args.frames_dir
        if args.max_frames:
            d_frames = _limit_frames(d_frames, args.max_frames, os.path.join(rp.root, "frames_used"))
    else:
        extract.extract_frames(args.video, d_frames, resume=args.resume, max_frames=args.max_frames)
    n_frames, native_size = extract.video_meta(d_frames)
    out_size = (tuple(int(v) for v in args.out_size.lower().split("x"))
                if args.out_size else native_size)
    if args.segment_starts:
        ss = args.segment_starts                      # list (config) or "0,48" (CLI)
        starts = [int(x) for x in (ss if isinstance(ss, (list, tuple)) else str(ss).split(","))]
    else:
        starts = videopainter.default_segments(n_frames)
    mode = "single-pass" if n_frames <= CLIP else f"multi-chunk ({len(starts)} segments)"
    print(f"[pipeline] {n_frames} frames, native={native_size}, out={out_size}, "
          f"mode={mode}, segments={starts}")
    if args.stop_after == "extract":
        return

    # 2-4. edit-region masks + per-segment anchors
    if args.backend == "assets":
        em = edit_mask_mod.get_edit_mask("assets", assets_dir=args.assets)
        an = anchor_mod.get_anchor("assets", assets_dir=args.assets)
        target_mask_dir = em.mask_dir
    else:  # roma
        assert args.ref0, "backend=roma needs --ref0 (frame-0 reference image)"
        em = edit_mask_mod.get_edit_mask(
            "roma", frames_dir=d_frames, ref0_path=args.ref0,
            target_word=(args.target_word or args.target), source_word=args.source,
            ref0_mask_path=args.ref0_mask, work_dir=rp.roma, dilate=args.dilate,
            region_shape=args.region_shape)
        an = anchor_mod.get_anchor(
            "roma", frames_dir=d_frames, ref0_path=args.ref0,
            work_dir=rp.roma, segment_starts=starts)
        target_mask_dir = em.mask_dir   # triggers RoMa edit masks

    if args.backend == "roma":
        d_mask = target_mask_dir        # frame-0 (target∪source) bbox, warped per frame
    elif args.mask_mode == "target":
        assert target_mask_dir, "mask_mode=target but provider has no target masks"
        d_mask = target_mask_dir
    else:
        if not (args.resume and extract.has_frames(rp.mask_src)):
            from components import sam3_mask
            sam3_mask.track(sam3_mask.build_predictor(), d_frames, args.source, rp.mask_src)
        if args.mask_mode == "source":
            d_mask = rp.mask_src
        else:  # union
            d_mask = edit_mask_mod.union_masks(rp.mask_src, target_mask_dir, rp.mask)
    if args.stop_after == "mask":
        return

    # 5. generate (load models once, loop segments)
    if not (args.resume and extract.has_frames(rp.gen_frames)):
        pipe = videopainter.load_pipeline(args.model_path, args.branch, args.id_lora)
        videopainter.generate(pipe, d_frames, d_mask, an.anchor_for_start, rp.gen,
                              segment_starts=starts, total=CLIP, prompt=args.prompt,
                              dilate=args.dilate, steps=args.steps, guidance=args.guidance,
                              seed=args.seed)
    if args.stop_after == "generate":
        return

    # 5.5 ROSE removal -> clean plate (source object + its shadow removed). Runs in
    # its own `rose` conda env as a subprocess; needs a per-frame SOURCE mask (cup
    # silhouette), which the roma backend doesn't otherwise compute, so derive it here.
    clean_plate = None
    if args.removal == "rose":
        from components import removal
        if not (args.resume and extract.has_frames(rp.mask_src)):
            from components import sam3_mask
            sam3_mask.track(sam3_mask.build_predictor(), d_frames, args.source, rp.mask_src)
        if not (args.resume and extract.has_frames(rp.clean_frames)):
            removal.remove(d_frames, rp.mask_src, rp.removal)
        clean_plate = rp.clean_frames
    if args.stop_after == "removal":
        return

    # 6. composite. Default OFF for RoMa anchors (replace_gt already keeps the
    # background) — UNLESS removal=rose, where we composite the new object onto the
    # ROSE clean plate so the source object's shadow leaves the final output.
    if args.removal == "rose":
        total = len(glob.glob(f"{clean_plate}/frame_*.png"))   # clean plate is 16n+1 (<= n)
        composite_step.composite(clean_plate, rp.gen_frames, d_mask, rp.composite, total=total)
        src_dir = rp.composite
        print(f"[pipeline] composited onto ROSE clean plate ({total} frames)")
    elif args.no_composite or args.backend == "roma":
        src_dir = rp.gen_frames
        print(f"[pipeline] composite skipped (backend={args.backend}); using gen frames")
    else:
        composite_step.composite(d_frames, rp.gen_frames, d_mask, rp.composite, total=n_frames)
        src_dir = rp.composite
    if args.stop_after == "composite":
        return

    # 7. encode portrait (+ optional RIFE anchor de-spike).
    anchor_frames = [s + 1 for s in starts if s > 0]   # 1-indexed boundary frames
    if args.interpolate and anchor_frames:
        src_dir = encode_step.despike_anchors(src_dir, rp.despike, anchor_frames)
    encode_step.encode(src_dir, rp.final, out_size, fps=args.fps)
    print(f"[pipeline] DONE -> {rp.final}")


if __name__ == "__main__":
    main()
