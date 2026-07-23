"""
run_ab_comparison.py — for each video, run the shared prep stages ONCE, then
BOTH generation routes (videopainter and mvinpainter), producing two final
videos per source clip for side-by-side comparison.

Shared (run once per video — no duplicated SAM3/Gemini/edit-mask work):
  extract_frames -> sam3_mask (SOURCE mask) + gemini_edit (ref0) -> roma_edit_mask
  (TARGET/edit mask, shared by BOTH generators) -> detect_camera_motion
  (informational only here — both routes run regardless, so you can compare the
  actual outputs against what the router WOULD have picked).

Route A — videopainter:
  roma_anchors -> videopainter_generate -> composite -> encode
  -> outputs/<name>/videopainter.mp4

Route B — mvinpainter:
  mvinpainter_anchors -> mvinpainter_generate -> composite -> encode
  -> outputs/<name>/mvinpainter.mp4

Every stage goes through orchestrator.call_tool(), so tool dispatch (including
the remote-H100 routing for videopainter_generate, exactly as
orchestrator.run() does it) is not re-implemented/re-guessed here — this stays
in sync automatically with orchestrator.py.

Setup: same env vars as orchestrator.py / mcp_server.py — HF_HOME, HF_TOKEN,
GEMINI_API_KEY, MVINPAINTER_PYTHON, MVINPAINTER_ROOT, H100_HOST, etc.

Usage:
  1. Fill in the VIDEOS list below (or pass --config videos.json, same shape):

       {"name": "cup_banana_1", "video_path": "raw/cup1.mp4",
        "source": "cup",                       # SAM3 noun to remove
        "target": "a ripe yellow banana",       # full description -> generation prompt
        "target_word": "banana",                # optional: short noun for SAM3 on ref0
                                                  # (defaults to 'target' if omitted, but
                                                  # a short noun matches roma_edit_mask's
                                                  # SAM3 call better than a full phrase)
        "max_frames": 100,                      # optional: cap clip length for a quick test
        "mvi_chunk": 20, "mvi_n_views": 24,      # optional: mvinpainter route params
        "vp_steps": 10, "vp_guidance": 6.0,      # optional: videopainter route params
        "seed": 42, "dilate": 12}

  2. python run_ab_comparison.py
     python run_ab_comparison.py --only videopainter   # just one route
     python run_ab_comparison.py --config videos.json
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from orchestrator import call_tool
from contracts import layout

_VP = layout.MODELS["videopainter"]

# ── Edit this list directly, or supply --config videos.json with the same shape ──
VIDEOS = [
    # {"name": "cup_banana_1", "video_path": "raw/cup1.mp4",
    #  "source": "cup", "target": "a ripe yellow banana", "target_word": "banana",
    #  "max_frames": 100},
]


def _call(label: str, name: str, **kwargs) -> dict:
    printable = {k: v for k, v in kwargs.items() if not isinstance(v, dict)}
    print(f"\n=== [{label}] {name}({printable}) ===")
    result = call_tool(name, kwargs)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{name} failed: {result['error']}\n{result.get('detail', '')}")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2, default=str)[:1500])
    return result


def run_shared(v: dict, out_root: str) -> dict:
    """extract -> sam3 mask -> gemini edit -> roma edit mask -> camera motion
    (informational). Returns everything both routes need, computed once."""
    name = v["name"]
    out_dir = os.path.join(out_root, name)

    r = _call(name, "extract_frames", video_path=v["video_path"],
              out_dir=os.path.join(out_dir, "frames_src"),
              max_frames=v.get("max_frames"), resume=True)
    frames_dir, n_frames = r["frames_dir"], r["n_frames"]

    r = _call(name, "sam3_mask", frames_dir=frames_dir, source_word=v["source"],
              out_dir=os.path.join(out_dir, "mask_src"), resume=True)
    mask_src_dir = r["mask_dir"]

    r = _call(name, "gemini_edit", frame0_path=os.path.join(frames_dir, "frame_00001.png"),
              out_path=os.path.join(out_dir, "ref0.png"),
              source=v["source"], target=v["target"])
    ref0_path = r["ref0_path"]

    r = _call(name, "roma_edit_mask", frames_dir=frames_dir, ref0_path=ref0_path,
              target_word=v.get("target_word", v["target"]), source_word=v["source"],
              work_dir=os.path.join(out_dir, "roma"), dilate=v.get("dilate", 12))
    edit_mask_dir = r["mask_dir"]

    r = _call(name, "detect_camera_motion", frames_dir=frames_dir, mask_dir=mask_src_dir,
              inlier_ratio_thresh=v.get("inlier_ratio_thresh", 0.6),
              coherent_fraction_thresh=v.get("coherent_fraction_thresh", 0.5))
    print(f"\n>>> [{name}] detect_camera_motion -> camera_motion={r['camera_motion']} "
          f"(informational only — both routes run regardless)")

    return {
        "out_dir": out_dir, "frames_dir": frames_dir, "n_frames": n_frames,
        "mask_src_dir": mask_src_dir, "ref0_path": ref0_path,
        "edit_mask_dir": edit_mask_dir, "camera_motion": r["camera_motion"],
    }


def run_videopainter_route(v: dict, shared: dict, prompt: str) -> str:
    from components.videopainter import default_segments
    name = v["name"]
    out_dir = shared["out_dir"]
    frames_dir, mask_dir, n_frames = shared["frames_dir"], shared["edit_mask_dir"], shared["n_frames"]
    segment_starts = default_segments(n_frames)

    r = _call(name, "roma_anchors", frames_dir=frames_dir, ref0_path=shared["ref0_path"],
              work_dir=os.path.join(out_dir, "roma"), segment_starts=segment_starts)
    anchor_map = r["anchor_map"]

    r = _call(name, "videopainter_generate", frames_dir=frames_dir, mask_dir=mask_dir,
              anchor_map=anchor_map, gen_dir=os.path.join(out_dir, "gen_videopainter"),
              segment_starts=segment_starts, prompt=prompt,
              model_path=v.get("model_path", _VP["model_path"]),
              branch=v.get("branch", _VP["branch"]), id_lora=v.get("id_lora", _VP["id_lora"]),
              dilate=v.get("dilate", 12), steps=v.get("vp_steps", 10),
              guidance=v.get("vp_guidance", 6.0), seed=v.get("seed", 42))
    gen_frames_dir = r["gen_frames_dir"]

    r = _call(name, "composite", bg_frames_dir=frames_dir, gen_frames_dir=gen_frames_dir,
              mask_dir=mask_dir, out_dir=os.path.join(out_dir, "composite_videopainter"),
              total=n_frames)
    composite_dir = r["composite_dir"]

    r = _call(name, "encode", frames_dir=composite_dir,
              out_path=os.path.join(out_dir, "videopainter.mp4"),
              source_frames_dir=frames_dir, segment_starts=segment_starts, interpolate=True)
    print(f"\n>>> [{name}] videopainter route DONE -> {r['video_path']}")
    return r["video_path"]


def run_mvinpainter_route(v: dict, shared: dict, prompt: str) -> str:
    name = v["name"]
    out_dir = shared["out_dir"]
    frames_dir, mask_dir, n_frames = shared["frames_dir"], shared["edit_mask_dir"], shared["n_frames"]
    chunk = v.get("mvi_chunk", 20)
    segment_starts = list(range(0, n_frames, chunk)) or [0]

    r = _call(name, "mvinpainter_anchors", frames_dir=frames_dir, ref0_path=shared["ref0_path"],
              mask_dir=mask_dir, work_dir=os.path.join(out_dir, "mvi_anchor"),
              segment_starts=segment_starts, n_views=v.get("mvi_n_views", 24),
              prompt=prompt, name=f"{name}_mvi_anchor")
    anchor_map = r["anchor_map"]

    r = _call(name, "mvinpainter_generate", frames_dir=frames_dir, mask_dir=mask_dir,
              anchor_map=anchor_map, gen_dir=os.path.join(out_dir, "gen_mvinpainter"),
              segment_starts=segment_starts, prompt=prompt, chunk=chunk)
    gen_frames_dir = r["gen_frames_dir"]

    r = _call(name, "composite", bg_frames_dir=frames_dir, gen_frames_dir=gen_frames_dir,
              mask_dir=mask_dir, out_dir=os.path.join(out_dir, "composite_mvinpainter"),
              total=n_frames)
    composite_dir = r["composite_dir"]

    # interpolate=False: RIFE de-spike targets VideoPainter's 49-frame segment
    # boundaries specifically; not meaningful for mvinpainter's chunk boundaries.
    r = _call(name, "encode", frames_dir=composite_dir,
              out_path=os.path.join(out_dir, "mvinpainter.mp4"),
              source_frames_dir=frames_dir, segment_starts=segment_starts, interpolate=False)
    print(f"\n>>> [{name}] mvinpainter route DONE -> {r['video_path']}")
    return r["video_path"]


def main():
    ap = argparse.ArgumentParser(description="Run BOTH videopainter and mvinpainter routes per video")
    ap.add_argument("--config", help="JSON file: list of video dicts (see VIDEOS above for shape)")
    ap.add_argument("--out_root", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--only", choices=["videopainter", "mvinpainter", "both"], default="both",
                    help="run one route only, or both (default)")
    args = ap.parse_args()

    videos = VIDEOS
    if args.config:
        with open(args.config) as f:
            videos = json.load(f)
    if not videos:
        print("No videos configured. Edit VIDEOS at the top of this script, "
              "or pass --config videos.json (see the module docstring for the shape).")
        return
    missing = [v["name"] for v in videos if not all(v.get(k) for k in ("video_path", "source", "target"))]
    if missing:
        print(f"These entries are missing video_path/source/target: {missing}")
        return

    results = {}
    for v in videos:
        name = v["name"]
        print(f"\n{'#' * 70}\n# {name}\n{'#' * 70}")
        prompt = v.get("prompt", v["target"])
        shared = run_shared(v, args.out_root)
        results[name] = {"camera_motion": shared["camera_motion"]}
        if args.only in ("videopainter", "both"):
            results[name]["videopainter"] = run_videopainter_route(v, shared, prompt)
        if args.only in ("mvinpainter", "both"):
            results[name]["mvinpainter"] = run_mvinpainter_route(v, shared, prompt)

    print(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()