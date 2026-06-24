#!/usr/bin/env python3
"""Generate per-case pipeline configs (+ Gemini ref0s) from eval/prompt_spec.yaml.

For each (video, tier) it writes eval/configs/<video>_<tier>.yaml and, unless
--no-ref0, generates the frame-0 reference via the Gemini edit stage (resized to
the exact frame size — nano-banana does NOT preserve resolution, and ref0 must be
pixel-aligned with frame 0 for RoMa).

    # configs only (review first):
    python eval/gen_configs.py --spec eval/prompt_spec.yaml --no-ref0
    # configs + ref0s (needs GEMINI_API_KEY via .env):
    python eval/gen_configs.py --spec eval/prompt_spec.yaml
    # subset:
    python eval/gen_configs.py --videos cup0,pomelo_in_sunlight --tiers easy,medium,hard

ROSE policy: --removal {none,rose,auto}. 'auto' enables rose only on videos whose
failure_mode is shadow/reflection (where source-shadow leakage matters).
"""
import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

TIERS = ["easy", "medium", "hard"]
ROSE_AUTO_MODES = {"shadow", "reflection"}


def frame0(video_path, out_png):
    """Extract frame 0 of a video to a PNG (via imageio)."""
    import imageio.v3 as iio
    from PIL import Image
    for f in iio.imiter(video_path):
        Image.fromarray(f[:, :, :3]).save(out_png)
        return out_png
    raise ValueError(f"no frames in {video_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=os.path.join(HERE, "prompt_spec.yaml"))
    ap.add_argument("--video-root", default=os.path.join(REPO, "input", "benchmark"))
    ap.add_argument("--ref0-dir", default=os.path.join(REPO, "input", "ref0"))
    ap.add_argument("--config-dir", default=os.path.join(HERE, "configs"))
    ap.add_argument("--videos", default="", help="comma list to subset (default: all)")
    ap.add_argument("--tiers", default=",".join(TIERS), help="comma list of tiers")
    ap.add_argument("--max-frames", type=int, default=49)
    ap.add_argument("--long-videos", default="",
                    help="comma list of videos to ALSO emit at --long-frames "
                         "(multi-segment, exercises reanchor); names get a '_long' suffix")
    ap.add_argument("--long-frames", type=int, default=97,
                    help="frame count for the --long-videos subset (>49 spans >1 segment)")
    ap.add_argument("--removal", default="none", choices=["none", "rose", "auto"])
    ap.add_argument("--region-shape", default="rect")
    ap.add_argument("--source-mask", default="track", choices=["track", "warp"],
                    help="per-frame SAM3 source ∪ RoMa target (track, default) | legacy warped seed")
    ap.add_argument("--offload", default="sequential", choices=["sequential", "model", "none"],
                    help="VideoPainter VRAM/speed baked into each config: sequential (~22GB, "
                         "safe on any card, default) | model | none (~44GB, fastest, needs 80GB)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--no-ref0", action="store_true", help="write configs only, skip Gemini ref0")
    ap.add_argument("--ref0-model", default=None, help="override Gemini image model")
    ap.add_argument("--force-ref0", action="store_true", help="regenerate ref0 even if present")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))["videos"]
    want_vids = [v.strip() for v in args.videos.split(",") if v.strip()] or list(spec)
    want_tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    long_vids = {v.strip() for v in args.long_videos.split(",") if v.strip()}
    os.makedirs(args.config_dir, exist_ok=True)
    os.makedirs(args.ref0_dir, exist_ok=True)

    written, ref0_made, ref0_failed = [], 0, []
    for vid in want_vids:
        if vid not in spec:
            print(f"[skip] {vid}: not in spec"); continue
        s = spec[vid]
        video = os.path.join(args.video_root, f"{vid}.mp4")
        if not os.path.exists(video):
            print(f"[skip] {vid}: no video at {video}"); continue
        rose = (args.removal == "rose" or
                (args.removal == "auto" and s.get("failure_mode") in ROSE_AUTO_MODES))
        f0 = None
        for tier in want_tiers:
            t = s[tier]
            name = f"{vid}_{tier}"
            ref0 = os.path.join(args.ref0_dir, f"{name}.png")
            if not args.no_ref0 and (args.force_ref0 or not os.path.exists(ref0)):
                from components import gemini_edit
                if f0 is None:
                    f0 = os.path.join(args.ref0_dir, f"{vid}_frame0.png")
                    frame0(video, f0)
                try:
                    from PIL import Image
                    gemini_edit.edit_image(f0, ref0, source=s["source"], target=t["target"],
                                           model=args.ref0_model or gemini_edit.DEFAULT_MODEL)
                    sz = Image.open(f0).size
                    Image.open(ref0).convert("RGB").resize(sz, Image.LANCZOS).save(ref0)
                    ref0_made += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[ref0 FAIL] {name}: {e}"); ref0_failed.append(name)
            # One config at the standard frame count; for the long subset, an
            # extra `_long` config at --long-frames that spans >1 segment (and so
            # exercises per-segment reanchor). Both reuse the same ref0.
            variants = [(name, args.max_frames)]
            if vid in long_vids:
                variants.append((f"{name}_long", args.long_frames))
            for cfg_name, mf in variants:
                cfg = {
                    "video": os.path.relpath(video, REPO),
                    "source": s["source"],
                    "target": t["target"],
                    "target_word": t["target_word"],
                    "prompt": f'{t["target"]}, {s["scene"]}',
                    "name": cfg_name,
                    "backend": "roma",
                    "ref0": os.path.relpath(ref0, REPO),
                    "dilate": 12,
                    "region_shape": args.region_shape,
                    "source_mask": args.source_mask,
                    "offload": args.offload,
                    "steps": args.steps,
                    "guidance": args.guidance,
                    "seed": args.seed,
                    "max_frames": mf,
                    "removal": "rose" if rose else "none",
                    "out_size": None,
                    "fps": args.fps,
                    "interpolate": True,
                    "no_composite": False,
                    "model_label": "VideoPainter",
                    "failure_mode": s.get("failure_mode", ""),
                    "tier": tier,
                }
                path = os.path.join(args.config_dir, f"{cfg_name}.yaml")
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
                written.append(path)

    print(f"\n[gen] wrote {len(written)} configs -> {args.config_dir}")
    if not args.no_ref0:
        print(f"[gen] generated {ref0_made} ref0 images -> {args.ref0_dir}")
        if ref0_failed:
            print(f"[gen] {len(ref0_failed)} ref0 FAILED: {ref0_failed}")


if __name__ == "__main__":
    main()
