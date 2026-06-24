"""Single-stage runner — a toggle to test ONE pipeline step in isolation.

Pick a stage; inputs default to the run dir (outputs/<name>/) + config, and any
input can be overridden for ad-hoc tests. Complements pipeline.py (full run).

  python stage.py --config config.yaml --stage edit_mask
  python stage.py --config config.yaml --stage anchor
  python stage.py --config config.yaml --stage removal     --mask_dir outputs/<name>/mask_src
  python stage.py --config config.yaml --stage generate    --segment_starts 0
  python stage.py --config config.yaml --stage composite   --plate_dir <clean> --object_dir <gen>
  python stage.py --config config.yaml --stage encode      --frames_dir <dir>

Stages: extract | sam3 | edit_mask | anchor | removal | generate | composite | encode
(gemini ref0 prep is data-preprocessing — use `python -m components.gemini_edit`.)
"""
import argparse
import os
import glob
import yaml

from contracts import layout

STAGES = ["extract", "sam3", "edit_mask", "anchor", "removal", "generate", "composite", "encode"]


def _starts(args, n):
    from components import videopainter
    if args.segment_starts:
        ss = args.segment_starts
        return [int(x) for x in (ss if isinstance(ss, (list, tuple)) else str(ss).split(","))]
    return videopainter.default_segments(n)


def main():
    ap = argparse.ArgumentParser(description="Run a single pipeline stage (toggle)")
    ap.add_argument("--config")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--name")
    # inputs / overrides (default from config + run dir)
    ap.add_argument("--video")
    ap.add_argument("--frames_dir")
    ap.add_argument("--mask_dir", help="per-frame masks (sam3/removal/composite input)")
    ap.add_argument("--out_dir")
    ap.add_argument("--plate_dir", help="composite background plate (e.g. ROSE clean frames)")
    ap.add_argument("--object_dir", help="composite object frames (e.g. gen frames)")
    ap.add_argument("--source"); ap.add_argument("--target"); ap.add_argument("--target_word")
    ap.add_argument("--prompt"); ap.add_argument("--ref0"); ap.add_argument("--ref0_mask")
    ap.add_argument("--segment_starts", default=None)
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--region_shape", default="rect", choices=["bbox", "rect", "hull"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video_length", type=int, default=None, help="(removal) 16n+1; default auto")
    ap.add_argument("--out_size", default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--backend", default="roma", choices=["roma", "assets"])
    ap.add_argument("--model_path", default=layout.MODELS["videopainter"]["model_path"])
    ap.add_argument("--branch", default=layout.MODELS["videopainter"]["branch"])
    ap.add_argument("--id_lora", default=layout.MODELS["videopainter"]["id_lora"])

    cfg_path = ap.parse_known_args()[0].config
    if cfg_path:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in ap._actions}
        ap.set_defaults(**{k: v for k, v in cfg.items() if k in known and v is not None})
    args = ap.parse_args()

    rp = layout.RunPaths(args.name) if args.name else None
    frames = args.frames_dir or (rp.frames_src if rp else None)
    roma_dir = os.path.join(rp.root, "roma") if rp else None

    def n_frames():
        return len(glob.glob(f"{frames}/frame_*.png"))

    s = args.stage
    if s == "extract":
        from components import extract
        assert args.video and rp, "extract needs --video and --name"
        extract.extract_frames(args.video, rp.frames_src, resume=False)

    elif s == "sam3":
        from components import sam3_mask
        out = args.out_dir or (rp.mask_src if rp else None)
        sam3_mask.track(sam3_mask.build_predictor(), frames, args.source, out)

    elif s == "edit_mask":
        from components import edit_mask
        em = edit_mask.get_edit_mask(args.backend, frames_dir=frames, ref0_path=args.ref0,
                                     target_word=(args.target_word or args.target), source_word=args.source,
                                     ref0_mask_path=args.ref0_mask, work_dir=roma_dir, dilate=args.dilate,
                                     region_shape=args.region_shape, assets_dir=args.out_dir)
        print("[stage] edit masks ->", em.mask_dir)

    elif s == "anchor":
        from components import anchor
        an = anchor.get_anchor(args.backend, frames_dir=frames, ref0_path=args.ref0,
                               work_dir=roma_dir, segment_starts=_starts(args, n_frames()),
                               assets_dir=args.out_dir)
        an.prepare() if hasattr(an, "prepare") else None
        print("[stage] anchors ready (work_dir=%s)" % roma_dir)

    elif s == "removal":
        from components import removal
        mask = args.mask_dir or (rp.mask_src if rp else None)
        out = args.out_dir or (os.path.join(rp.root, "clean") if rp else None)
        removal.remove(frames, mask, out, video_length=args.video_length, prompt=args.prompt or "")

    elif s == "generate":
        from components import videopainter, anchor
        an = anchor.get_anchor(args.backend, frames_dir=frames, ref0_path=args.ref0,
                               work_dir=roma_dir, segment_starts=_starts(args, n_frames()),
                               assets_dir=args.out_dir)
        d_mask = args.mask_dir or os.path.join(roma_dir, "masks")
        out = args.out_dir or rp.gen
        pipe = videopainter.load_pipeline(args.model_path, args.branch, args.id_lora)
        videopainter.generate(pipe, frames, d_mask, an.anchor_for_start, out,
                              segment_starts=_starts(args, n_frames()), prompt=args.prompt,
                              dilate=args.dilate, steps=args.steps, guidance=args.guidance, seed=args.seed)

    elif s == "composite":
        from components import composite
        plate = args.plate_dir or frames
        obj = args.object_dir or (rp.gen_frames if rp else None)
        mask = args.mask_dir or os.path.join(roma_dir, "masks")
        out = args.out_dir or (rp.composite if rp else None)
        composite.composite(plate, obj, mask, out, total=len(glob.glob(f"{obj}/frame_*.png")))

    elif s == "encode":
        from components import encode
        src = args.frames_dir or (rp.gen_frames if rp else None)
        out = args.out_dir or (rp.final if rp else "out.mp4")
        size = (tuple(int(v) for v in args.out_size.lower().split("x")) if args.out_size else None)
        assert size, "encode needs --out_size WxH"
        encode.encode(src, out, size, fps=args.fps)

    print(f"[stage] '{s}' done.")


if __name__ == "__main__":
    main()
