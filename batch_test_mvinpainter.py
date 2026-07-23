"""
batch_test_mvinpainter_steps.py — time each stage of the mvinpainter route
separately: SAM3 source masking, the MVInpainter anchor pass, and the
MVInpainter generation fill (swept across several --inference_step values so
you can pick the fastest step count that still looks acceptable).

SAM3 and the anchor pass each run ONCE (their own cost isn't affected by the
generation step count you're testing); only the generation fill is repeated
per step value, using components.mvinpainter.generate() directly (mode=
"single") on one chunk of frames — NOT the full generate_chunked() loop —
since you only need to compare one chunk's speed/quality across step counts.

Usage:
  python batch_test_mvinpainter_steps.py \
      --frames_dir outputs/mvi_cake/frames_src \
      --source cake \
      --mask_dir outputs/mvi_cake/roma/masks \
      --ref0 outputs/mvi_cake/ref0.png \
      --prompt "a glazed chocolate donut" \
      --steps 10,15,20,25,30

  --source is optional: give it to time sam3_mask too (recomputed with
  resume=True, so if outputs/<frames_dir's run>/mask_src already exists this
  is near-instant and you'll see that reflected in its timing).
"""
import argparse
import os
import shutil
import sys
import glob
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mcp_server as srv
from components import mvinpainter
from components.anchor import get_anchor


def _timed(label, fn):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    print(f">>> {label}: {dt:.1f}s")
    return result, dt


def main():
    ap = argparse.ArgumentParser(description="Time SAM3 / mvinpainter anchors / mvinpainter generation separately")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--mask_dir", required=True, help="edit-region mask (white = new object region) used by anchors + generation")
    ap.add_argument("--ref0", required=True, help="frame-0 reference: new object on frame 0")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--source", default=None, help="SAM3 noun to mask (optional — times sam3_mask if given)")
    ap.add_argument("--out_root", default=os.path.join(_HERE, "outputs", "mvi_steps_test"))
    ap.add_argument("--nframe", type=int, default=24, help="frames in the test group (mode=single)")
    ap.add_argument("--anchor_steps", type=int, default=None,
                    help="inference steps for the (single, un-swept) anchor pass; default = component's own default")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", default="10,15,20,25,30", help="comma list of --inference_step values to sweep for GENERATION")
    ap.add_argument("--force", action="store_true", help="rerun even if output for a step count already exists")
    args = ap.parse_args()

    durations = {}

    # ── SAM3 source masking (optional) ──────────────────────────────────────
    if args.source:
        _, dt = _timed("sam3_mask", lambda: srv.sam3_mask(
            frames_dir=args.frames_dir, source_word=args.source,
            out_dir=os.path.join(args.out_root, "mask_src"), resume=True))
        durations["sam3_mask"] = dt
    else:
        print("\n[skip] no --source given — sam3_mask timing skipped")

    # ── MVInpainter anchor pass (single-mode, once) ─────────────────────────
    anchor_work_dir = os.path.join(args.out_root, "anchor")

    def _run_anchor():
        an = get_anchor(
            "mvinpainter", frames_dir=args.frames_dir, ref0_path=args.ref0,
            mask_dir=args.mask_dir, work_dir=anchor_work_dir, n_views=args.nframe,
            prompt=args.prompt, name="steps_test_anchor", steps=args.anchor_steps,
        )
        an.prepare()
        return an

    _, dt = _timed(f"mvinpainter_anchors (steps={args.anchor_steps or 'default'})", _run_anchor)
    durations["mvinpainter_anchors"] = dt

    # ── MVInpainter anchors (once, for all chunk tests) ─────────────────────
    n_frames = len(glob.glob(os.path.join(args.frames_dir, "frame_*.png")))
    chunk_size = 20
    segment_starts = list(range(0, n_frames, chunk_size))[:4]  # Only first 4 chunks
    print(f"\n[chunked test] n_frames={n_frames}, chunk_size={chunk_size}, segment_starts={segment_starts}")
    
    anchor_work_dir = os.path.join(args.out_root, "anchors_chunked")
    an = get_anchor(
        "mvinpainter", frames_dir=args.frames_dir, ref0_path=args.ref0,
        mask_dir=args.mask_dir, work_dir=anchor_work_dir, n_views=args.nframe,
        prompt=args.prompt, name="chunked_test", steps=args.anchor_steps,
    )
    an.prepare()
    
    def anchor_for_start(s):
        return an.anchor_path_for_start(s)

    # ── MVInpainter generation fill (chunked), swept across --inference_step values ──
    step_values = [int(s) for s in args.steps.split(",")]
    gen_results = []

    for steps in step_values:
        run_dir = os.path.join(args.out_root, f"steps_{steps}_chunked")
        if args.force and os.path.exists(run_dir):
            shutil.rmtree(run_dir)

        out_dir, dt = _timed(f"mvinpainter_generate_chunked (steps={steps})", lambda s=steps: mvinpainter.generate_chunked(
            args.frames_dir, args.mask_dir, anchor_for_start, run_dir,
            segment_starts=segment_starts, chunk=chunk_size, prompt=args.prompt, steps=s, name=f"chunked_test_{s}",
        ))
        n_out_frames = len(glob.glob(os.path.join(out_dir, "frame_*.png")))
        gen_results.append((steps, dt, n_out_frames, out_dir))
        print(f">>> steps={steps}: {dt:.1f}s total ({dt / max(n_out_frames, 1):.2f}s/frame), "
              f"{n_out_frames} frames -> {out_dir}")    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    if "sam3_mask" in durations:
        print(f"{'sam3_mask':<28} {durations['sam3_mask']:>10.1f}s")
    print(f"{'mvinpainter_anchors':<28} {durations['mvinpainter_anchors']:>10.1f}s")
    print()
    print(f"{'mvinpainter_generate':<28} {'time(s)':>10}  {'s/frame':>8}  frames_dir")
    baseline = gen_results[-1][1] if gen_results else None   # highest step count = baseline
    for steps, dt, n_frames, out_dir in gen_results:
        speedup = f"  ({baseline / dt:.2f}x vs steps={gen_results[-1][0]})" if baseline and dt > 0 else ""
        print(f"  steps={steps:<4}{'':<15} {dt:>10.1f}  {dt / max(n_frames, 1):>8.2f}  {out_dir}{speedup}")

    total = sum(durations.values()) + sum(dt for _, dt, _, _ in gen_results)
    print(f"\nTotal wall-clock time across all timed stages: {total:.1f}s")
    print("\nInspect each steps_<N>/frames/ directory side by side to judge the "
          "quality cutoff — pick the lowest steps value where output still looks "
          "acceptable, not just the fastest one.")


if __name__ == "__main__":
    main()