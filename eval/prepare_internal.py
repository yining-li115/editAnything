#!/usr/bin/env python3
"""Convert our 90 internal edited videos into the unified benchmark case schema.

Reads the per-case pipeline configs that produced editAnything_results_20260624/,
remaps their (repo-relative) paths onto the results pack layout, probes video
metadata, and emits one JSONL line per case (design doc §5). No model is re-run —
the edited videos already exist; this only builds the evaluation manifest.

  python scripts/prepare_internal.py \
    --results_root editAnything_results_20260624 \
    --out data/internal/cases.jsonl
"""
import argparse
import glob
import json
import os
import sys
import subprocess

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # eval/ (for paths.py)
import paths  # noqa: E402


def probe(path):
    """Return (num_frames, [w,h], fps) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    w, h, rate, n = out.split(",")[:4]
    num, den = (rate.split("/") + ["1"])[:2]
    fps = round(float(num) / float(den), 3) if float(den) else float(num)
    return int(n), [int(w), int(h)], fps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default=paths.RESULTS_PACK,
                    help="the results pack (input_videos/, final_videos/, ref0/, prompts/configs/)")
    ap.add_argument("--out", default=paths.CASES_JSONL)
    ap.add_argument("--mask_root", default=paths.MASK_ROOT,
                    help="fallback dir where masks are REGENERATED if the pipeline's own aren't found")
    ap.add_argument("--pipeline_outputs", default=paths.PIPELINE_OUTPUTS,
                    help="pipeline run dir; if outputs/<name>/mask exists, reuse it instead of regenerating")
    args = ap.parse_args()

    def resolve_mask_dir(name):
        """Prefer the pipeline's OWN edit mask (produced at generation time) over
        regenerating. mask = the exact region the model was allowed to edit."""
        for cand in (os.path.join(args.pipeline_outputs, name, "mask"),        # track-mode union mask
                     os.path.join(args.pipeline_outputs, name, "roma", "masks")):  # warp-mode target mask
            if glob.glob(os.path.join(cand, "frame_*.png")):
                return os.path.abspath(cand), "pipeline"
        return os.path.abspath(os.path.join(args.mask_root, name)), "regen"

    root = os.path.abspath(args.results_root)
    cfgs = sorted(glob.glob(f"{root}/prompts/configs/*.yaml"))
    assert cfgs, f"no configs under {root}/prompts/configs/"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cases, skipped = [], []
    for cp in cfgs:
        c = yaml.safe_load(open(cp))
        name = c["name"]                                   # e.g. cup2_medium
        src_id = os.path.splitext(os.path.basename(c["video"]))[0]   # input/benchmark/cup2.mp4 -> cup2
        source_video = f"{root}/input_videos/{src_id}.mp4"
        edited_video = f"{root}/final_videos/{name}.mp4"
        ref0 = f"{root}/ref0/{os.path.basename(c['ref0'])}"

        missing = [p for p in (source_video, edited_video, ref0) if not os.path.exists(p)]
        if missing:
            skipped.append((name, missing))
            continue

        n_edit, res_edit, fps_edit = probe(edited_video)
        tier = c.get("tier")
        fmode = c.get("failure_mode")
        mask_dir, mask_src = resolve_mask_dir(name)
        case = {
            "case_id": f"internal_{name}",
            "benchmark_source": "internal",
            "name": name,
            "source_video": source_video,
            "edited_video": edited_video,
            "ref0": ref0,
            # prompts
            "source_object": c.get("source"),                 # SAM3 noun for the OLD object
            "target_object": c.get("target"),                 # short new-object description
            "target_word": c.get("target_word") or c.get("target"),  # SAM3 noun for ref0
            "edit_instruction": f"Replace the {c.get('source')} with {c.get('target')}.",
            "source_prompt": f"a video of {c.get('source')}",  # derived (we have no annotated source caption)
            "target_prompt": c.get("prompt") or c.get("target"),   # rich generation prompt
            "edit_type": "object_replacement",
            "source_type": "real",
            # internal stress-test fields
            "replacement_transformation_tier": tier,          # easy | medium | hard
            "failure_tags": [fmode] if fmode else [],
            "model": c.get("model_label", "VideoPainter"),
            # edit-region mask: reuse the pipeline's own if present, else regen dir
            "mask_dir": mask_dir,
            "mask_source": mask_src,          # "pipeline" (reused) | "regen" (needs regen_masks.py)
            # params needed to regenerate the exact edit-region mask (SAM3 + RoMa) if reuse fails
            "mask_params": {
                "dilate": c.get("dilate", 12),
                "region_shape": c.get("region_shape", "rect"),
                "source_mask": c.get("source_mask", "track"),
                "max_frames": c.get("max_frames", n_edit),
            },
            "metadata": {
                "num_frames": n_edit,
                "fps": fps_edit,
                "resolution": res_edit,
            },
            "config_path": os.path.abspath(cp),
        }
        cases.append(case)

    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # summary
    from collections import Counter
    tiers = Counter(c["replacement_transformation_tier"] for c in cases)
    tags = Counter(t for c in cases for t in c["failure_tags"])
    print(f"[prepare_internal] wrote {len(cases)} cases -> {args.out}")
    print(f"  tiers: {dict(tiers)}")
    print(f"  failure_tags: {dict(tags)}")
    if skipped:
        print(f"  SKIPPED {len(skipped)} (missing files):")
        for n, m in skipped[:10]:
            print(f"    {n}: {[os.path.basename(x) for x in m]}")


if __name__ == "__main__":
    main()
