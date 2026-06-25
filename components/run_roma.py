"""CLI wrapper for the RoMa backend — produces per-frame edit masks + per-segment
anchors, matching pipeline.py's roma branch. Replaces the stale version that
called the old unified `anchors` module (which the refactor split into
`components.edit_mask` and `components.anchor`).

What it builds, mirroring pipeline.py (backend=roma, source_mask=track):
  1. RoMa edit masks      : edit_mask.get_edit_mask("roma", ...).mask_dir
                            = frame-0 (target∪source) hull, warped per frame.
  2. per-frame SAM3 source: sam3_mask.track(... source ...) -> mask_src
                            (covers the moving/occluded old object).
  3. union (+dilate)      : edit_mask.union_masks(mask_src, roma_masks) -> final
                            edit region (closes the moving-source leak).
  4. per-segment anchors  : anchor.get_anchor("roma", ...) -> anchors_dir.

Prints machine-parseable KEY=VALUE lines (ANCHORS_DIR=, MASK_DIR=, N_ANCHORS=,
N_MASKS=) on the last lines so a subprocess wrapper can read the result paths
without guessing the layout.

    python components/run_roma.py \
        --frames_dir outputs/<name>/frames_src \
        --ref0 outputs/<name>/ref0.png \
        --target_word banana --source_word cup \
        --work_dir outputs/<name>/roma \
        --mask_src outputs/<name>/mask_src \
        --union_dir outputs/<name>/mask \
        --segment_starts 0,48,96,144,192,240,251 \
        [--ref0_mask ...] [--dilate 12] [--region_shape rect] \
        [--source_mask track|warp]
"""
import argparse
import glob
import os

from components import edit_mask as edit_mask_mod
from components import anchor as anchor_mod


def _count(d, pattern="frame_*.png"):
    return len(glob.glob(os.path.join(d, pattern))) if os.path.isdir(d) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--ref0", required=True)
    ap.add_argument("--target_word", required=True)
    ap.add_argument("--source_word", required=True)
    ap.add_argument("--work_dir", required=True, help="roma work dir -> {masks,anchors}/")
    ap.add_argument("--segment_starts", required=True,
                    help="comma list, e.g. 0,48,96,144,192,240,251")
    ap.add_argument("--mask_src", default=None,
                    help="(source_mask=track) dir for per-frame SAM3 source masks")
    ap.add_argument("--union_dir", default=None,
                    help="(source_mask=track) dir for the unioned edit region "
                         "(source ∪ warped target). If unset, defaults to work_dir/union.")
    ap.add_argument("--ref0_mask", default=None)
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--region_shape", default="rect", choices=["bbox", "rect", "hull"])
    ap.add_argument("--source_mask", default="track", choices=["track", "warp"])
    args = ap.parse_args()

    starts = [int(x) for x in args.segment_starts.split(",")]

    # 1. RoMa edit masks (frame-0 target∪source hull, warped per frame).
    em = edit_mask_mod.get_edit_mask(
        "roma", frames_dir=args.frames_dir, ref0_path=args.ref0,
        target_word=args.target_word, source_word=args.source_word,
        ref0_mask_path=args.ref0_mask, work_dir=args.work_dir,
        dilate=args.dilate, region_shape=args.region_shape)
    roma_mask_dir = em.mask_dir          # property access triggers prepare()/warp

    # 2-3. source_mask=track: per-frame SAM3 source ∪ warped target, then dilate.
    if args.source_mask == "track":
        import components.sam3_mask as sam3_mask
        mask_src = args.mask_src or os.path.join(args.work_dir, "mask_src")
        if _count(mask_src) == 0:
            sam3_mask.track(sam3_mask.build_predictor(), args.frames_dir,
                            args.source_word, mask_src)
        else:
            print(f"[run_roma] reusing per-frame source masks in {mask_src}")
        union_dir = args.union_dir or os.path.join(args.work_dir, "union")
        final_mask_dir = edit_mask_mod.union_masks(
            mask_src, roma_mask_dir, union_dir, dilate=args.dilate)
    else:  # warp: warped (target∪source) seed only (legacy, leaks a moving source)
        final_mask_dir = roma_mask_dir

    # 4. per-segment anchors (whole ref0 warped to each segment start).
    an = anchor_mod.get_anchor(
        "roma", frames_dir=args.frames_dir, ref0_path=args.ref0,
        work_dir=args.work_dir, segment_starts=starts)
    an.prepare()
    anchors_dir = an.anchors_dir

    print("DONE")
    print(f"ANCHORS_DIR={anchors_dir}")
    print(f"MASK_DIR={final_mask_dir}")
    print(f"N_ANCHORS={_count(anchors_dir, 'anchor_*.png')}")
    print(f"N_MASKS={_count(final_mask_dir)}")
    print(f"EXPECTED_N_ANCHORS={len(starts)}")


if __name__ == "__main__":
    main()