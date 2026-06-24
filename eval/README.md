# Evaluation

Scores generated object-replacement videos against the benchmark metrics: three
no-ground-truth automatic metrics (CLIP score, temporal consistency, DINO score)
plus a Gemini VLM judge, blended into a final score.

We evaluate the **output** video of each run — `outputs/<name>/final.mp4` —
using that run's `roma/masks/` for the DINO object crop.

## Install

```bash
pip install -r eval/requirements.txt
export GEMINI_API_KEY=...        # only needed for the VLM judge
```

## Run it all (one-shot)

Whole eval over the configs you ran (manifest → metrics → aggregate):

```bash
python eval/build_manifest.py --configs eval/configs && \
python eval/run_eval.py       --manifest eval/manifest.json && \
python eval/aggregate.py      --results eval/results
```

Or the full benchmark — generate every config's video, then eval — in one script
(assumes `eval/configs/` + ref0s exist; run `eval/gen_configs.py` first if not):

```bash
bash eval/run_benchmark.sh    # run_batch -> manifest -> eval -> aggregate
```

Both are resumable: `run_eval` caches each `results/<id>.json` and `run_batch`
skips videos that already have a `final.mp4`. The per-stage docs follow.

## 1. Build the manifest

The manifest maps each output video to its prompts + mask dir. Prompts come from
the per-case pipeline configs (the pipeline doesn't copy a config into each
output dir, so point this at the configs you ran):

```bash
python eval/build_manifest.py --configs eval/configs/*.yaml
# or a directory of configs:
python eval/build_manifest.py --configs path/to/case_configs/
```

This writes `eval/manifest.json`. Each entry:

```json
{
  "video_id": "cup2_rose",
  "object_prompt": "cup",
  "replace_prompt": "a ripe yellow banana",
  "full_prompt": "a ripe yellow banana resting on ...",
  "model": "VideoPainter",
  "video": "outputs/cup2_rose/final.mp4",
  "mask_dir": "outputs/cup2_rose/roma/masks"
}
```

Edit it freely — `run_eval.py` reads it verbatim. (Cases whose `final.mp4`
doesn't exist yet are listed as warnings; run the pipeline first.)

## 2. Run the metrics

```bash
python eval/run_eval.py --manifest eval/manifest.json
# subset of metrics (e.g. skip the VLM judge):
python eval/run_eval.py --metrics clip,temporal,dino
# smoke test on the first 2 cases:
python eval/run_eval.py --limit 2
```

Writes one `eval/results/<video_id>.json` per case (the BENCHMARK.md
"Per Test Case Output" schema). Models load once; finished cases are cached
(use `--force` to recompute).

## 3. Aggregate

```bash
python eval/aggregate.py --results eval/results
```

Writes `eval/summary.md` (the Benchmark Summary Table + averages) and
`eval/summary.csv`, and prints the table.

## Layout

```
eval/
  config.yaml          # sampling params, model ids, final-score weights
  build_manifest.py    # configs -> manifest.json
  run_eval.py          # manifest -> per-case results/*.json
  aggregate.py         # results/*.json -> summary.md + summary.csv
  metrics/
    frames.py          # video/frame/mask IO + sampling + mask crop
    clip_metrics.py    # CLIP score + temporal consistency (shared CLIP model)
    dino_metrics.py    # DINO score over masked object crops
    vlm_judge.py       # Gemini judge (current google-genai SDK)
```

## Notes

- **Final score** = `0.3·CLIP(norm) + 0.2·Temporal + 0.2·DINO + 0.3·VLM(/10)`,
  weights in `config.yaml`. If a metric is skipped/unavailable, weights are
  renormalized over the present ones. CLIP raw cosine is small, so it's linearly
  normalized via `clip_norm.lo/hi` before weighting — tune these on your data.
- **DINO** crops the object region from `roma/masks/`. If a run has no masks
  (e.g. `removal=rose` without RoMa masks), it warns and falls back to full
  frames, which measures whole-scene rather than object consistency.
- **VLM model** is configurable (`vlm_model` in `config.yaml`); defaults to a
  current Gemini model via the `google-genai` SDK (BENCHMARK.md's
  `google.generativeai` + `gemini-1.5-pro` are deprecated). Disabled
  automatically if no API key is set.
```
