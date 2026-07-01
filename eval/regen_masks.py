#!/usr/bin/env python3
"""Regenerate the per-frame edit-region masks for the internal cases.

The results pack ships edited videos but NOT the masks the pipeline used. We
rebuild them by reusing the SAME components the pipeline did (SAM3 + RoMa), so the
mask == the region the model was allowed to change == FiVE's "edit mask" for
outside-mask preservation. Mirrors pipeline.py stages 2-4 (roma edit mask +
per-frame SAM3 source track + union), per each case's recorded mask_params.

Run in the `editanything` conda env (needs torch 2.4 / sam3 / romatch):
  HF_HOME=/workspace/.hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /venv/editanything/bin/python scripts/regen_masks.py --cases data/internal/cases.jsonl
  # smoke test: --limit 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval/ (for paths.py)
import paths  # noqa: E402
sys.path.insert(0, paths.EDITANYTHING)                # repo root, for `components`

from components import extract                         # noqa: E402
from components import edit_mask as edit_mask_mod      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=paths.CASES_JSONL)
    ap.add_argument("--cache", default=paths.CACHE_ROOT,
                    help="where source frames + roma work dirs live (reused across runs)")
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (smoke test)")
    ap.add_argument("--force", action="store_true", help="recompute even if mask_dir already full")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases)]
    if args.limit:
        cases = cases[:args.limit]

    # shared SAM3 predictor for the per-frame source track (built once)
    import gc
    import torch
    from components import roma_warp

    predictor = None
    ok, fail = 0, []
    processed = 0
    for k, c in enumerate(cases, 1):
        name = c["name"]
        mp = c["mask_params"]
        mask_dir = c["mask_dir"]
        if not args.force and extract.has_frames(mask_dir):
            print(f"[{k}/{len(cases)}] {name}: masks exist, skip")
            ok += 1
            continue
        try:
            with torch.no_grad():   # inference only — don't retain RoMa autograd graphs
                n = int(mp.get("max_frames") or c["metadata"]["num_frames"])
                frames_dir = os.path.join(args.cache, "frames_src", name)
                work_dir = os.path.join(args.cache, "roma", name)
                os.makedirs(frames_dir, exist_ok=True)
                # 1. source frames (first n, native res) — aligns with the edited clip
                extract.extract_frames(c["source_video"], frames_dir, resume=True, max_frames=n)

                # 2. RoMa edit-region masks (warped frame-0 target∪source hull/rect + dilate)
                em = edit_mask_mod.get_edit_mask(
                    "roma", frames_dir=frames_dir, ref0_path=c["ref0"],
                    target_word=(c.get("target_word") or c.get("target_object")),
                    source_word=c["source_object"], work_dir=work_dir,
                    dilate=mp.get("dilate", 12), region_shape=mp.get("region_shape", "rect"))
                target_mask_dir = em.mask_dir      # triggers RoMa warp

                # 3. source_mask=track: union with per-frame SAM3 source (matches pipeline)
                if mp.get("source_mask", "track") == "track":
                    import components.sam3_mask as sam3_mask
                    if predictor is None:
                        predictor = sam3_mask.build_predictor()
                    src_track = os.path.join(args.cache, "mask_src", name)
                    if not extract.has_frames(src_track):
                        sam3_mask.track(predictor, frames_dir, c["source_object"], src_track)
                    edit_mask_mod.union_masks(src_track, target_mask_dir, mask_dir,
                                              dilate=mp.get("dilate", 12))
                else:
                    # warp-only: the roma target mask IS the edit region
                    edit_mask_mod.union_masks(target_mask_dir, None, mask_dir)
            print(f"[{k}/{len(cases)}] {name}: masks -> {mask_dir} ({len(os.listdir(mask_dir))} frames)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{len(cases)}] {name}: FAILED {type(e).__name__}: {e}")
            fail.append((name, str(e)))
        finally:
            # bound GPU memory growth over the long loop (SAM3+RoMa OOM'd ~case 29)
            processed += 1
            gc.collect(); torch.cuda.empty_cache()
            if processed % 12 == 0:          # periodic full teardown; models reload lazily
                predictor = None
                roma_warp._roma = None
                gc.collect(); torch.cuda.empty_cache()

    print(f"\n[regen_masks] done: {ok}/{len(cases)} ok, {len(fail)} failed")
    for n, e in fail:
        print(f"  FAIL {n}: {e[:120]}")


if __name__ == "__main__":
    main()
