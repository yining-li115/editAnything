"""CLI wrapper for anchors.RomaAnchors — exposes pipeline.py's roma-backend call.

python run_roma.py --frames_dir ... --ref0 replacement.png \
    --target_word banana --source_word cup \
    --work_dir outputs/<name>/roma --segment_starts 0,48,96,144,192,240,251 \
    [--ref0_mask ...] [--dilate 12]

Mirrors pipeline.py's:
    provider = anchors.get_anchor_provider("roma", frames_dir=..., ref0_path=...,
                                            target_word=..., source_word=...,
                                            work_dir=..., segment_starts=..., dilate=...)
    target_mask_dir = provider.target_mask_dir   # triggers RoMa
"""
import argparse
import anchors

ap = argparse.ArgumentParser()
ap.add_argument("--frames_dir", required=True)
ap.add_argument("--ref0", required=True)
ap.add_argument("--target_word", required=True)
ap.add_argument("--source_word", required=True)
ap.add_argument("--work_dir", required=True)
ap.add_argument("--segment_starts", required=True, help="comma list, e.g. 0,48,96,144,192,240,251")
ap.add_argument("--ref0_mask", default=None)
ap.add_argument("--dilate", type=int, default=12)
args = ap.parse_args()

starts = [int(x) for x in args.segment_starts.split(",")]
provider = anchors.get_anchor_provider(
    "roma", frames_dir=args.frames_dir, ref0_path=args.ref0,
    target_word=args.target_word, source_word=args.source_word,
    work_dir=args.work_dir, segment_starts=starts,
    ref0_mask_path=args.ref0_mask, dilate=args.dilate)
provider.prepare()
print(f"DONE anchors={provider.anchors_dir} masks={provider.target_mask_dir}")