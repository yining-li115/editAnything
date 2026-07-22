"""
Manual pipeline test: extract -> sam3_mask -> gemini_edit -> detect_camera_motion
-> roma_edit_mask -> mvinpainter_anchors -> mvinpainter_generate

Bypasses the Gemini orchestrator entirely — calls the mcp_server.py tool
functions directly, in order, and stops right after mvinpainter_generate.
Use this to validate the camera-motion route's plumbing (and tune
detect_camera_motion's thresholds) without depending on the LLM to pick the
right tools in the right order.

Requires a clip with REAL camera motion (pan/dolly/handheld) — if
detect_camera_motion reports camera_motion=False on your test clip, this
script stops early and tells you so (it only exercises the mvinpainter route).

Usage:
    python test_mvinpainter_chain.py --video path/to/panning_clip.mp4 \
        --source cup --target "a ripe yellow banana" --name mvi_test
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mcp_server as srv


def _p(label: str, result: dict) -> dict:
    print(f"\n=== {label} ===")
    print(json.dumps(result, indent=2, default=str)[:2000])
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{label} failed: {result['error']}\n{result.get('detail', '')}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Manual test: chain up through mvinpainter_generate")
    ap.add_argument("--video", required=True)
    ap.add_argument("--source", required=True, help="object noun to remove, e.g. 'cup'")
    ap.add_argument("--target", required=True, help="description of the new object")
    ap.add_argument("--name", default="mvi_chain_test", help="run name -> outputs/<name>/")
    ap.add_argument("--max_frames", type=int, default=None, help="limit clip length for a quick test")
    ap.add_argument("--n_views", type=int, default=24, help="mvinpainter_anchors sampled views")
    ap.add_argument("--chunk", type=int, default=20, help="mvinpainter_generate reanchor chunk size")
    ap.add_argument("--force_mvinpainter", action="store_true",
                    help="run the mvinpainter route even if camera_motion=False "
                         "(useful for testing the tool plumbing on any clip)")
    args = ap.parse_args()

    out_root = os.path.join(_HERE, "outputs", args.name)
    os.makedirs(out_root, exist_ok=True)

    # 1. extract
    r = _p("extract_frames", srv.extract_frames(
        video_path=args.video, out_dir=os.path.join(out_root, "frames_src"),
        max_frames=args.max_frames))
    frames_dir, n_frames = r["frames_dir"], r["n_frames"]

    # 2. sam3 source mask
    r = _p("sam3_mask", srv.sam3_mask(
        frames_dir=frames_dir, source_word=args.source,
        out_dir=os.path.join(out_root, "mask_src")))
    mask_src_dir = r["mask_dir"]

    # 3. gemini edit -> ref0
    r = _p("gemini_edit", srv.gemini_edit(
        frame0_path=os.path.join(frames_dir, "frame_00001.png"),
        out_path=os.path.join(out_root, "ref0.png"),
        source=args.source, target=args.target))
    ref0_path = r["ref0_path"]

    # 4. detect camera motion (background-only, excludes the SOURCE mask)
    r = _p("detect_camera_motion", srv.detect_camera_motion(
        frames_dir=frames_dir, mask_dir=mask_src_dir))
    camera_motion = r["camera_motion"]
    print(f"\n>>> camera_motion = {camera_motion}  "
          f"(coherent_fraction={r['coherent_fraction']}, "
          f"median_transform_px={r['median_transform_px']}, "
          f"n_pairs_sampled={r['n_pairs_sampled']})")

    if not camera_motion and not args.force_mvinpainter:
        print("\n[!] camera_motion=False on this clip — the roma_anchors + "
              "videopainter_generate route would be picked instead. This script "
              "only exercises the mvinpainter route; use a panning/handheld clip, "
              "or pass --force_mvinpainter to run it anyway for a plumbing test.")
        return

    # 5. roma edit mask (TARGET/edit region — same mask both generators consume)
    r = _p("roma_edit_mask", srv.roma_edit_mask(
        frames_dir=frames_dir, ref0_path=ref0_path,
        target_word=args.target, source_word=args.source,
        work_dir=os.path.join(out_root, "roma")))
    edit_mask_dir = r["mask_dir"]

    segment_starts = list(range(0, n_frames, 48)) or [0]
    print(f"\n>>> segment_starts = {segment_starts}")

    # 6. mvinpainter anchors
    r = _p("mvinpainter_anchors", srv.mvinpainter_anchors(
        frames_dir=frames_dir, ref0_path=ref0_path, mask_dir=edit_mask_dir,
        work_dir=os.path.join(out_root, "mvi_anchor"),
        segment_starts=segment_starts, n_views=args.n_views,
        prompt=args.target))
    anchor_map = r["anchor_map"]

    # 7. mvinpainter generate
    r = _p("mvinpainter_generate", srv.mvinpainter_generate(
        frames_dir=frames_dir, mask_dir=edit_mask_dir, anchor_map=anchor_map,
        gen_dir=os.path.join(out_root, "gen"), segment_starts=segment_starts,
        prompt=args.target, chunk=args.chunk))
    print(f"\n>>> DONE: {r['n_frames_generated']} frames -> {r['gen_frames_dir']}")
    print(">>> Inspect a few frames there directly, or ffmpeg-encode a quick "
          "preview: ffmpeg -framerate 8 -i "
          f"{r['gen_frames_dir']}/frame_%05d.png -pix_fmt yuv420p /tmp/preview.mp4")


if __name__ == "__main__":
    main()