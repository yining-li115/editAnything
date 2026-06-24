#!/usr/bin/env python3
"""Build eval/manifest.json from the pipeline configs used to run the benchmark.

The pipeline does not persist a config into each output dir, so the prompts live
in the per-case config YAMLs you ran. Point this at them:

    python eval/build_manifest.py --configs eval/configs/*.yaml
    python eval/build_manifest.py --configs path/to/case_configs/   # a directory

Each config contributes one manifest entry:
    video_id       <- name
    object_prompt  <- source
    replace_prompt <- target        (short; used for CLIP + VLM)
    full_prompt    <- prompt        (the long generation prompt, kept for reference)
    video          <- outputs/<name>/final.mp4
    mask_dir       <- outputs/<name>/roma/masks   (for the DINO crop)

Edit the resulting manifest.json freely; run_eval.py reads it verbatim.
"""
import argparse
import glob
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand_configs(patterns):
    paths = []
    for pat in patterns:
        if os.path.isdir(pat):
            paths += glob.glob(os.path.join(pat, "*.yaml")) + glob.glob(os.path.join(pat, "*.yml"))
        else:
            paths += glob.glob(pat)
    return sorted(set(paths))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="config YAML files, globs, or a directory")
    ap.add_argument("--outputs-root", default=os.path.join(REPO, "outputs"),
                    help="base dir holding outputs/<name>/ (default: repo outputs/)")
    ap.add_argument("--model", default="VideoPainter",
                    help="model label recorded per case (overridden by `model` in a config)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "manifest.json"))
    args = ap.parse_args()

    configs = expand_configs(args.configs)
    if not configs:
        ap.error("no config files matched")

    entries, missing = [], []
    for cfg_path in configs:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        name = cfg.get("name")
        if not name:
            print(f"[skip] {cfg_path}: no `name` field")
            continue
        run_dir = os.path.join(args.outputs_root, name)
        video = os.path.join(run_dir, "final.mp4")
        mask_dir = os.path.join(run_dir, "roma", "masks")
        if not os.path.exists(video):
            missing.append((name, video))
        entries.append({
            "video_id": name,
            "object_prompt": cfg.get("source", ""),
            "replace_prompt": cfg.get("target", ""),
            "full_prompt": cfg.get("prompt", ""),
            "model": cfg.get("model_label", args.model),
            "video": video,
            "mask_dir": mask_dir if os.path.isdir(mask_dir) else "",
        })

    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"[manifest] wrote {len(entries)} cases -> {args.out}")
    if missing:
        print(f"[manifest] WARNING: {len(missing)} case(s) have no final.mp4 yet "
              "(run the pipeline first):")
        for name, video in missing:
            print(f"    {name}: {video}")


if __name__ == "__main__":
    main()
