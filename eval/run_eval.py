#!/usr/bin/env python3
"""Run the benchmark metrics over the generated videos listed in a manifest.

    python eval/run_eval.py --manifest eval/manifest.json
    python eval/run_eval.py --manifest eval/manifest.json --metrics clip,temporal,dino
    python eval/run_eval.py --manifest eval/manifest.json --limit 3   # smoke test

Writes one results/<video_id>.json per case (schema = BENCHMARK.md "Per Test Case
Output"). Heavy models (CLIP, DINOv2, Gemini) load once and are reused across
cases. Per-case results are cached; re-runs skip finished cases unless --force.
Run eval/aggregate.py afterwards for the summary table.
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from components.gemini_edit import load_dotenv  # noqa: E402

# Load GEMINI_API_KEY (and friends) from the repo .env so the VLM judge works
# without the key being exported in the shell. Same no-dependency loader the
# Gemini edit stage uses; a key already in the environment takes precedence.
load_dotenv(os.path.join(REPO, ".env"))

from metrics import frames as fr  # noqa: E402

ALL_METRICS = ["clip", "temporal", "dino", "vlm"]


def normalize_clip(raw, lo, hi):
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


def final_score(rec, weights, clip_norm):
    """Weighted blend of available metrics; weights renormalized over present ones."""
    parts = {}
    if rec.get("clip_score") is not None:
        parts["clip"] = normalize_clip(rec["clip_score"], clip_norm["lo"], clip_norm["hi"])
    if rec.get("temporal_consistency") is not None:
        parts["temporal"] = rec["temporal_consistency"]
    if rec.get("dino_score") is not None:
        parts["dino"] = rec["dino_score"]
    if rec.get("vlm_judge") and rec["vlm_judge"].get("overall") is not None:
        parts["vlm"] = float(rec["vlm_judge"]["overall"]) / 10.0
    if not parts:
        return None
    wsum = sum(weights[k] for k in parts)
    if wsum == 0:
        return None
    return round(sum(weights[k] * v for k, v in parts.items()) / wsum, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--metrics", default=",".join(ALL_METRICS),
                    help="comma list of: " + ",".join(ALL_METRICS))
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (smoke test)")
    ap.add_argument("--force", action="store_true", help="recompute cases with cached results")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.manifest) as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[:args.limit]

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in ALL_METRICS]
    if bad:
        ap.error(f"unknown metrics: {bad}")
    os.makedirs(args.out, exist_ok=True)

    # ---- lazily build the backends actually needed (load once) ----
    clip_scorer = dino_scorer = vlm = None
    if "clip" in metrics or "temporal" in metrics:
        from metrics.clip_metrics import ClipScorer
        print("[eval] loading CLIP ...")
        clip_scorer = ClipScorer(cfg["clip_model"])
    if "dino" in metrics:
        from metrics.dino_metrics import DinoScorer
        print("[eval] loading DINOv2 ...")
        dino_scorer = DinoScorer(cfg["dino_model"])
    if "vlm" in metrics:
        from metrics.vlm_judge import GeminiJudge, VlmUnavailable
        try:
            vlm = GeminiJudge(cfg["vlm_model"])
        except VlmUnavailable as e:
            print(f"[eval] VLM judge disabled: {e}")
            metrics = [m for m in metrics if m != "vlm"]

    every, n = cfg.get("sample_every", 10), cfg.get("sample_n", 0)

    for k, case in enumerate(cases, 1):
        vid = case["video_id"]
        out_path = os.path.join(args.out, f"{vid}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"[{k}/{len(cases)}] {vid}: cached, skip")
            continue
        if not os.path.exists(case["video"]):
            print(f"[{k}/{len(cases)}] {vid}: MISSING video {case['video']}, skip")
            continue

        print(f"[{k}/{len(cases)}] {vid}: decoding ...")
        all_frames = fr.load_video(case["video"])
        idx = fr.sample_indices(len(all_frames), every=every, n=n)
        frames = [all_frames[i] for i in idx]

        rec = {
            "video_id": vid,
            "object_prompt": case.get("object_prompt", ""),
            "replace_prompt": case.get("replace_prompt", ""),
            "model": case.get("model", ""),
            "n_frames_total": len(all_frames),
            "n_frames_sampled": len(frames),
            "clip_score": None,
            "temporal_consistency": None,
            "dino_score": None,
            "vlm_judge": None,
        }

        if clip_scorer is not None:
            cs, tc = clip_scorer.score(frames, case["replace_prompt"])
            if "clip" in metrics:
                rec["clip_score"] = round(cs, 4)
            if "temporal" in metrics:
                rec["temporal_consistency"] = round(tc, 4)

        if dino_scorer is not None:
            masks = fr.align_masks_to_frames(
                fr.load_masks(case.get("mask_dir")), idx, len(all_frames))
            if all(m is None for m in masks):
                print(f"    [dino] no masks for {vid}; using full frames")
            rec["dino_score"] = round(dino_scorer.score(frames, masks), 4)

        if vlm is not None:
            try:
                rec["vlm_judge"] = vlm.judge(
                    all_frames, case["replace_prompt"], cfg.get("vlm_keyframes", 8))
            except Exception as e:  # noqa: BLE001 - never let one case kill the run
                print(f"    [vlm] failed for {vid}: {e}")

        rec["final_score"] = final_score(rec, cfg["weights"], cfg["clip_norm"])

        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print(f"    -> {out_path}  final={rec['final_score']}")

    print(f"\n[eval] done. Aggregate with:\n    python {os.path.join('eval', 'aggregate.py')} "
          f"--results {args.out}")


if __name__ == "__main__":
    main()
