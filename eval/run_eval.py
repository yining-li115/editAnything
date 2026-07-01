#!/usr/bin/env python3
"""Run the rebuilt benchmark metrics over the cases in a JSONL manifest.

Reuses FiVE-Bench's MetricsCalculator (automatic metrics) + a Gemini judge, then
blends into the 4 dimensions + caps (metrics/scoring.py). One record per case ->
outputs/eval_results/<model>_<source>_case_scores.jsonl. Per-case cached; resumable.

Run in the `five-bench` env (needs FiVE deps + cotracker + pyiqa + google-genai):
  /venv/five-bench/bin/python scripts/run_eval.py --cases data/internal/cases.jsonl \
      --mode full --out outputs/eval_results/our_model_internal_case_scores.jsonl
  # smoke: --mode smoke --limit 3
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval/ (holds metrics/ + paths.py)
import paths                               # noqa: E402
from metrics import frames as fr           # noqa: E402
from metrics import scoring                # noqa: E402

MODES = {
    "smoke": dict(stride=8, keyframes=4, mfs=False, niqe=True, vlm=True),
    "dev":   dict(stride=8, keyframes=6, mfs=True,  niqe=True, vlm=True),
    "full":  dict(stride=8, keyframes=8, mfs=True,  niqe=True, vlm=True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=paths.CASES_JSONL)
    ap.add_argument("--scoring", default=paths.SCORING_YAML)
    ap.add_argument("--cache", default=paths.CACHE_ROOT,
                    help="source-frames dir (from regen_masks) used for MFS")
    ap.add_argument("--out", default=os.path.join(paths.EVAL_RESULTS, "our_model_internal_case_scores.jsonl"))
    ap.add_argument("--mode", default="full", choices=list(MODES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_mfs", action="store_true"); ap.add_argument("--no_niqe", action="store_true")
    ap.add_argument("--no_vlm", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.scoring))
    m = MODES[args.mode]
    use_mfs = m["mfs"] and not args.no_mfs
    use_niqe = m["niqe"] and not args.no_niqe
    use_vlm = m["vlm"] and not args.no_vlm
    stride = cfg["sampling"]["frame_stride"]

    cases = [json.loads(l) for l in open(args.cases)]
    if args.limit:
        cases = cases[:args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    per_case_dir = os.path.join(os.path.dirname(args.out), "cases")
    os.makedirs(per_case_dir, exist_ok=True)

    # ---- load backends once ----
    from metrics.five_adapter import FiveMetrics
    print(f"[eval] loading FiVE metrics (mfs={use_mfs}, niqe={use_niqe}) ...")
    five = FiveMetrics(enable_mfs=use_mfs, enable_niqe=use_niqe)
    judge = None
    if use_vlm:
        from metrics.vlm_judge import GeminiJudge, VlmUnavailable
        try:
            judge = GeminiJudge()
        except VlmUnavailable as e:
            print(f"[eval] VLM judge disabled: {e}")

    records = []
    for k, case in enumerate(cases, 1):
        vid = case["case_id"]
        cache_path = os.path.join(per_case_dir, f"{vid}.json")
        if os.path.exists(cache_path) and not args.force:
            records.append(json.load(open(cache_path)))
            print(f"[{k}/{len(cases)}] {vid}: cached")
            continue
        try:
            src_all = fr.decode_video(case["source_video"])
            edit_all = fr.decode_video(case["edited_video"])
            n = min(len(edit_all), len(src_all))
            mask_all = fr.load_mask_frames(case["mask_dir"])
            idx = fr.sample_indices(n, stride=stride)
            W, H = edit_all[0].shape[1], edit_all[0].shape[0]
            src_s = [src_all[i] for i in idx]
            edit_s = [edit_all[i] for i in idx]
            masks_s = fr.align_masks(mask_all, idx, (W, H))

            raw = five.frame_metrics(src_s, edit_s, masks_s, case["target_prompt"])
            if use_mfs:
                # MFS needs a source-frames dir. Reuse the pipeline's own frames_src
                # (a gen-time intermediate) if present, else the eval cache.
                pipe_frames = os.path.join(paths.PIPELINE_OUTPUTS, case["name"], "frames_src")
                src_dir = (pipe_frames if os.path.isdir(pipe_frames)
                           else os.path.join(args.cache, "frames_src", case["name"]))
                masks_full = fr.align_masks(mask_all, list(range(n)), (W, H))
                raw["mfs"] = five.mfs(src_dir, case["edited_video"], masks_full)

            judge_out = None
            if judge is not None:
                try:
                    judge_out = judge.judge(src_all[:n], edit_all[:n], case,
                                            n_keyframes=m["keyframes"])
                except Exception as e:  # noqa: BLE001
                    print(f"    [vlm] failed for {vid}: {e}")

            sc = scoring.score_case(raw, judge_out or {}, cfg,
                                    tier=case.get("replacement_transformation_tier"))
            rec = {
                "case_id": vid, "benchmark_source": case["benchmark_source"],
                "edit_type": case.get("edit_type"),
                "tier": case.get("replacement_transformation_tier"),
                "failure_tags": case.get("failure_tags", []),
                "model": case.get("model"),
                "raw_metrics": raw,
                **sc,
                "judge": judge_out,
            }
            json.dump(rec, open(cache_path, "w"), indent=2, ensure_ascii=False)
            records.append(rec)
            d = sc["dimensions"]
            print(f"[{k}/{len(cases)}] {vid}: final={sc['final_score']} "
                  f"ES={d['edit_success']} SP={d['source_preservation']} "
                  f"TC={d['temporal_consistency']} RQ={d['rendering_quality']} "
                  f"caps={sc['caps_applied']}")
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{len(cases)}] {vid}: ERROR {type(e).__name__}: {e}")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[eval] wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
