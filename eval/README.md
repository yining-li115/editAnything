# Evaluation — FiVE-based benchmark

Scores edited videos on **four editing-specific dimensions** and blends them into a
final score with **critical-failure caps**. It reuses **FiVE-Bench's real metric
code** (`FiVE-Bench/evaluation/metrics_calculator.py`) for the automatic metrics and
a **Gemini judge** for the subjective checklist — replacing the old CLIP-weighted
blend, which was too CLIP-sensitive and hid obvious failures.

```
Final = 0.35·EditSuccess + 0.30·SourcePreservation + 0.20·Temporal + 0.15·RenderingQuality
        then critical-failure caps (edit_not_completed→≤0.40, source_reappears→≤0.50, …)
```

| Dimension | What feeds it |
| --- | --- |
| **Edit Success** | Gemini edit-success checklist (0.7) + CLIP(edited↔target) (0.3) |
| **Source Preservation** | outside-mask PSNR/SSIM/LPIPS/structure_distance (0.5) + Gemini preservation checklist (0.5) |
| **Temporal Consistency** | CoTracker Motion-Fidelity-Score (0.5) + Gemini temporal checklist (0.5) |
| **Rendering Quality** | NIQE (pyiqa) |

Weights, normalization ranges, and caps all live in [`scoring.yaml`](scoring.yaml).

## Two environments (file-based handoff)

- **`editanything`** (torch 2.4 / sam3 / romatch) — only `regen_masks.py` needs it.
- **`five-bench`** (torch 2.4.1 + torchmetrics + cotracker + pyiqa + google-genai) —
  everything else. Build per `FiVE-Bench/INSTALL.md`; also `pip install transformers==4.49.0`
  (torchmetrics CLIPScore breaks on transformers 5.x) and `imageio-ffmpeg`.

External tools are vendored as siblings of this repo (like RIFE/ROSE) and resolved in
[`paths.py`](paths.py) (env-overridable): `../FiVE-Bench`, `../co-tracker/checkpoints/scaled_offline.pth`.

## Run it (internal 30×3 stress-test)

```bash
# 1. build the unified case manifest from the results pack (no model re-run)
python eval/prepare_internal.py            # -> data/internal/cases.jsonl

# 2. edit-region masks. PREFERRED: reuse the pipeline's own outputs/<name>/mask.
#    Only when those are absent (e.g. the shipped results pack) regenerate them:
HF_HOME=... /venv/editanything/bin/python eval/regen_masks.py   # SAM3+RoMa, fallback only

# 3. score (four dimensions + caps) — reuses FiVE's MetricsCalculator + Gemini judge
HF_HOME=... /venv/five-bench/bin/python eval/run_eval.py --mode full   # smoke: --mode smoke --limit 3

# 4. aggregate -> summary.json/csv + failure_gallery.html (by tier, by failure_tag, caps)
python eval/aggregate_results.py
```

Outputs (gitignored): `data/internal/cases.jsonl`, `outputs/eval_results/*_case_scores.jsonl`
(+ `cases/*.json`, per-case cached/resumable), `outputs/reports/summary.{json,csv}` + `failure_gallery.html`.

## Masks: reuse, don't recompute

The mask = the region the model was allowed to edit (source∪target, RoMa-warped). The
pipeline **already produces this at generation time** (`outputs/<name>/mask/`), so
`prepare_internal.py` points each case's `mask_dir` there and records `mask_source:
"pipeline"`. `regen_masks.py` is a **fallback** for pre-existing videos whose masks
weren't kept (the shipped pack excludes them). In the agentic loop, generation just
made the mask → eval reads it directly, no regeneration.

## Layout

| Path | Role |
| --- | --- |
| `paths.py` | path registry (external deps + data/outputs), env-overridable |
| `scoring.yaml` | dimension weights, metric normalization, critical-failure caps |
| `prepare_internal.py` | results pack → unified `cases.jsonl` (tier + failure_tags, mask reuse) |
| `regen_masks.py` | (fallback) regenerate edit masks via SAM3+RoMa — the only part needing `editanything` |
| `metrics/five_adapter.py` | wraps FiVE `MetricsCalculator` (Qwen FiVE-Acc stubbed → Gemini); adds NIQE |
| `metrics/vlm_judge.py` | Gemini structured checklist + critical flags |
| `metrics/scoring.py` | normalize + combine 4 dims + apply caps |
| `metrics/frames.py` | video/mask decode + frame sampling |
| `run_eval.py` | manifest → per-case 4-dim scores |
| `aggregate_results.py` | scores → summary + by-tier/failure_tag + failure gallery |
| `prompt_spec.yaml`, `gen_configs.py` | benchmark *authoring* (30×3 spec → per-case configs); not scoring |

## Notes

- **Score is more discriminative than the old blend** (std 0.127→0.183 on the 90 cases)
  and adds outside-mask preservation + MFS + NIQE + caps, none of which the old eval had.
- **Known metric caveats:** MFS is weak on near-static scenes (drags Temporal uniformly);
  NIQE is source-resolution-dependent (penalizes low-res sources). Both are re-tunable in
  `scoring.yaml`.
- **`benchmark_source`** in each case labels the *dataset* (`internal` here), not the
  metric — the metrics are FiVE-Bench's regardless.
