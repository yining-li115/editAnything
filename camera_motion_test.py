"""
batch_test_camera_motion.py — run components.camera_motion.detect() on several
raw video files and sweep thresholds to find values that correctly separate
STATIC from MOVING-camera clips.

Thresholds this script helps you refine (all params of camera_motion.detect):
  - px_threshold             min implied background displacement (px) for a
                              frame pair's homography fit to count as "coherent".
                              Static/tripod footage should sit near 0; genuine
                              pans/dollies should sit well above it.
  - inlier_ratio_thresh      min RANSAC inlier ratio for a pair to count as
                              "coherent". LOWER this for parallax-heavy/handheld
                              footage — a translating camera + scene depth means
                              no single homography fits all background points
                              well, so inlier ratio stays low even under real
                              motion.
  - coherent_fraction_thresh min fraction of sampled pairs that must be
                              "coherent" for camera_motion=True (majority-vote
                              cutoff, was hardcoded 0.5, now tunable). For
                              parallax-heavy clips this may need to go well
                              below 0.5, since only a minority of pairs will
                              ever fit one global homography even under strong
                              real motion.
  - stride                   frame gap between sampled pairs. NOT swept by the
                              --sweep grid below (changing it requires
                              recomputing features, unlike the three above which
                              are free to sweep from cached samples) — if you
                              want to test different strides, rerun this script
                              with --stride <N> --force for each value you want
                              to compare.

This script handles frame extraction and (optional) SAM3 source masking for
you — you only give it raw video files.

Usage:
  1. Fill in the VIDEOS list below (or pass --config videos.json with the same
     shape), one entry per test clip, each with your OWN ground-truth judgment:

       {"name": "handheld_pan_1", "video_path": "raw/pan1.mp4",
        "source": "cup", "label": "motion"}
       {"name": "tripod_1", "video_path": "raw/tripod1.mp4",
        "source": "cup", "label": "static"}

     "source" is the SAM3 object noun to mask out (optional but recommended —
     excludes the edited/moving object so it's never mistaken for camera
     motion; omit it if the clip has no object to exclude, e.g. an empty scene).
     "max_frames" (optional, int) caps extraction length for a quick test.

  2a. See current numbers for every video at one fixed set of thresholds:
        python batch_test_camera_motion.py

  2b. Grid-search thresholds that correctly classify every LABELED video:
        python batch_test_camera_motion.py --sweep

Re-running is cheap after the first pass: extract_frames/sam3_mask are called
with resume=True (reuse existing outputs/<name>/ contents), and raw per-pair
detection samples are cached to camera_motion_samples_cache.json, keyed by
(video name, stride) — add --force to recompute everything from scratch (e.g.
after re-extracting frames, or to test a different stride).
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mcp_server as srv
from components import camera_motion

# ── Edit this list directly, or supply --config videos.json with the same shape ──
VIDEOS = [
    # {"name": "handheld_pan_1", "video_path": "raw/pan1.mp4",
    #  "source": "cup", "label": "motion"},
    # {"name": "tripod_1", "video_path": "raw/tripod1.mp4",
    #  "source": "cup", "label": "static"},
]

VIDEOS = [
    {"name": "cake-full", "video_path": "/workspace/editAnything/input/camera_motion_test/cake-full.mp4", "source": "cake", "label": "motion"},
    {"name": "cake_first_frames",       "video_path": "/workspace/editAnything/input/camera_motion_test/cake_first_frames.mp4", "source": "cake", "label": "static"},
    {"name": "cup5_camera_moving_full",   "video_path": "/workspace/editAnything/input/camera_motion_test/cup5_camera_moving_full.mp4", "source": "cup", "label": "motion"},
    {"name": "cup5-camera-moving-short_big_motion",   "video_path": "/workspace/editAnything/input/camera_motion_test/cup5-camera-moving-short_big_motion.mp4", "source": "cup", "label": "motion"},
    {"name": "cup5-camera-moving-short_small_motion",   "video_path": "/workspace/editAnything/input/camera_motion_test/cup5-camera-moving-short_small_motion.mp4", "source": "cup", "label": "static"},
    {"name": "iced_coffee_with_maple_leaves",   "video_path": "/workspace/editAnything/input/camera_motion_test/iced_coffee_with_maple_leaves.mp4", "source": "iced coffee", "label": "static"},
    {"name": "cup2",   "video_path": "/workspace/editAnything/input/cup2.mp4", "source": "cup", "label": "static"},
    {"name": "pomelo_in_sunlight",   "video_path": "/workspace/editAnything/input/camera_motion_test/pomelo_in_sunlight.mp4", "source": "pomelo", "label": "static"},
    {"name": "woman_holding_donut",   "video_path": "/workspace/editAnything/input/camera_motion_test/woman_holding_donut.mp4", "source": "donut", "label": "static"},

]

CACHE_PATH = os.path.join(_HERE, "camera_motion_samples_cache.json")
OUT_ROOT = os.path.join(_HERE, "outputs")


def _prepare_frames_and_mask(v: dict):
    """extract_frames (+ sam3_mask if 'source' is given) for one video entry.
    Reuses outputs/<name>/ across reruns via resume=True. Returns
    (frames_dir, mask_dir_or_None)."""
    name = v["name"]
    out_dir = os.path.join(OUT_ROOT, name)

    r = srv.extract_frames(
        video_path=v["video_path"],
        out_dir=os.path.join(out_dir, "frames_src"),
        max_frames=v.get("max_frames"),
        resume=True,
    )
    if "error" in r:
        raise RuntimeError(f"extract_frames failed for {name}: {r['error']}\n{r.get('detail', '')}")
    frames_dir = r["frames_dir"]
    print(f"  [{name}] extract_frames -> {r['n_frames']} frames")

    mask_dir = None
    if v.get("source"):
        r = srv.sam3_mask(
            frames_dir=frames_dir, source_word=v["source"],
            out_dir=os.path.join(out_dir, "mask_src"), resume=True,
        )
        if "error" in r:
            raise RuntimeError(f"sam3_mask failed for {name}: {r['error']}\n{r.get('detail', '')}")
        mask_dir = r["mask_dir"]
        print(f"  [{name}] sam3_mask -> {r['n_masks']} masks")
    else:
        print(f"  [{name}] no 'source' given — running detect_camera_motion without a mask")

    return frames_dir, mask_dir


def compute_all_samples(videos, stride, max_pairs, cache_path, force=False):
    """extract_frames + sam3_mask (if configured) once per video, then run
    detect() once per video (the expensive part: ORB + matching + RANSAC),
    caching raw per-pair samples to disk so every threshold sweep below is
    instant."""
    cache = {}
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    for v in videos:
        key = f"{v['name']}::stride{stride}"
        if key in cache and not force:
            print(f"[cache] reusing samples for {v['name']}")
            continue
        print(f"[compute] {v['name']} (video_path={v['video_path']}) ...")
        frames_dir, mask_dir = _prepare_frames_and_mask(v)
        r = camera_motion.detect(
            frames_dir, mask_dir, stride=stride, max_pairs=max_pairs,
        )
        cache[key] = {"label": v.get("label"), "samples": r["samples"]}
        print(f"  -> {len(r['samples'])} pairs sampled")

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    return cache


def recompute(samples, px_threshold, inlier_ratio_thresh, coherent_fraction_thresh):
    coherent = [s for s in samples
                if s["n_matches"] > 0
                and s["inlier_ratio"] >= inlier_ratio_thresh
                and s["transform_px"] >= px_threshold]
    frac = len(coherent) / len(samples) if samples else 0.0
    return frac, frac >= coherent_fraction_thresh


def report_single(cache, px_threshold, inlier_ratio_thresh, coherent_fraction_thresh):
    print(f"\n{'video':<22} {'label':<8} {'coherent_frac':>14} {'median_px':>10} {'camera_motion':>14}  correct?")
    print("-" * 82)
    for key, entry in cache.items():
        name = key.split("::")[0]
        samples = entry["samples"]
        label = entry.get("label")
        frac, motion = recompute(samples, px_threshold, inlier_ratio_thresh, coherent_fraction_thresh)
        median_px = sorted(s["transform_px"] for s in samples)[len(samples) // 2] if samples else 0.0
        expected = None if label is None else (label == "motion")
        correct = "?" if expected is None else ("OK" if motion == expected else "WRONG")
        print(f"{name:<22} {str(label):<8} {frac:>14.3f} {median_px:>10.1f} {str(motion):>14}  {correct}")
    print("-" * 82)
    print(f"px_threshold={px_threshold}  inlier_ratio_thresh={inlier_ratio_thresh}  "
          f"coherent_fraction_thresh={coherent_fraction_thresh}")


def sweep(cache):
    """Grid-search threshold combos; report which ones classify every LABELED
    video correctly. Only videos with a 'label' set are used for scoring."""
    labeled = {k: v for k, v in cache.items() if v.get("label") in ("static", "motion")}
    if not labeled:
        print("[sweep] no labeled videos in cache — add 'label': 'static'|'motion' "
              "to entries in VIDEOS (or your --config json) first.")
        return

    best = []
    for px in (4.0, 6.0, 8.0, 10.0, 12.0):
        for ir in (0.6, 0.5, 0.4, 0.3, 0.25, 0.2):
            for maj in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15):
                correct = 0
                for entry in labeled.values():
                    frac, motion = recompute(entry["samples"], px, ir, maj)
                    if motion == (entry["label"] == "motion"):
                        correct += 1
                if correct == len(labeled):
                    best.append((px, ir, maj))

    print(f"\n=== {len(best)} threshold combo(s) correctly classify all "
          f"{len(labeled)} labeled video(s) ===")
    for px, ir, maj in best[:30]:
        print(f"  px_threshold={px:<5} inlier_ratio_thresh={ir:<5} coherent_fraction_thresh={maj:<5}")
    if len(best) > 30:
        print(f"  ... ({len(best) - 30} more)")
    if not best:
        print("  None found in this grid. Either your labeled videos aren't "
              "separable by this method with these ranges, or you need more "
              "labeled videos to narrow it down — a single video of each class "
              "can be satisfied by too many combos to trust any one of them.")
    elif len(best) > 1:
        print("\n  Multiple combos work on this dataset — prefer one with some "
              "margin (not sitting right at one video's exact numbers), and "
              "keep testing it against NEW clips before trusting it in "
              "production. A threshold that only works on the videos it was "
              "tuned on is not validated yet.")


def main():
    ap = argparse.ArgumentParser(description="Batch-test camera_motion.detect() thresholds across videos")
    ap.add_argument("--config", help="JSON file: list of video dicts (see VIDEOS above for shape)")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max_pairs", type=int, default=40)
    ap.add_argument("--force", action="store_true", help="recompute samples even if cached")
    ap.add_argument("--px_threshold", type=float, default=8.0)
    ap.add_argument("--inlier_ratio_thresh", type=float, default=0.6)
    ap.add_argument("--coherent_fraction_thresh", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true", help="grid-search thresholds against labeled videos")
    args = ap.parse_args()

    videos = VIDEOS
    if args.config:
        with open(args.config) as f:
            videos = json.load(f)
    if not videos:
        print("No videos configured. Edit the VIDEOS list at the top of this "
              "script, or pass --config videos.json (see the module docstring "
              "for the expected shape — each entry needs at least 'name' and "
              "'video_path').")
        return
    missing = [v["name"] for v in videos if not v.get("video_path")]
    if missing:
        print(f"These entries are missing 'video_path': {missing}")
        return

    cache = compute_all_samples(videos, args.stride, args.max_pairs, CACHE_PATH, force=args.force)

    if args.sweep:
        sweep(cache)
    else:
        report_single(cache, args.px_threshold, args.inlier_ratio_thresh, args.coherent_fraction_thresh)


if __name__ == "__main__":
    main()