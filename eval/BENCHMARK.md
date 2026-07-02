# Video-Editing Benchmark — Design & Logic

A rebuilt benchmark for object-replacement video editing. It reuses **FiVE-Bench's
metric code** for measurement and wraps it in an **editing-specific 4-dimension
scoring framework with critical-failure caps**, replacing the previous CLIP-weighted
blend that was too CLIP-sensitive and hid obvious failures.

---

## 1. Why rebuild

The old benchmark scored `0.3·CLIP + 0.2·temporal + 0.2·DINO + 0.3·VLM`. Problems:

- **CLIP too sensitive**, and no **source-preservation** or **failure** signal, so
  scores didn't track visible quality — a video where the edit *didn't happen* could
  still score ~0.85 because the background looked fine and CLIP matched the target text.
- Everything clustered in **0.7–0.9** (std 0.127), giving little discriminative power.

The rebuilt benchmark adds **outside-mask preservation, motion fidelity, no-reference
quality, and hard caps for critical failures**, and organizes everything into four
dimensions aligned with what actually matters in video editing.

---

## 2. Four dimensions + final score

```
Final = 0.35 · Edit Success
      + 0.30 · Source Preservation
      + 0.20 · Temporal Consistency
      + 0.15 · Rendering Quality
      → then critical-failure caps (see §5)
```

| Dimension | Question it answers | Components (auto + VLM) |
| --- | --- | --- |
| **Edit Success** | Did the requested edit actually happen? | `0.7·`Gemini edit-success checklist `+ 0.3·`CLIP(edited↔target) |
| **Source Preservation** | Is everything *outside* the edit unchanged? | `0.5·`outside-mask PSNR/SSIM/LPIPS/structure `+ 0.5·`Gemini preservation checklist |
| **Temporal Consistency** | Is the edit stable over time? | `0.5·`CoTracker Motion-Fidelity-Score `+ 0.5·`Gemini temporal checklist |
| **Rendering Quality** | Does it look good? | NIQE (no-reference quality) |

Rendering is weighted lowest on purpose: *a video can look good while failing the edit*.

---

## 3. Metrics (all measurement reused from FiVE-Bench)

Automatic metrics come verbatim from `FiVE-Bench/evaluation/metrics_calculator.py`
(we only stub out its Qwen2.5-VL "FiVE-Acc" and substitute a Gemini judge):

| Metric | Meaning | Direction |
| --- | --- | --- |
| PSNR / SSIM / LPIPS / MSE (outside mask) | background/identity preservation in the *unedited* region | ↑ (LPIPS/MSE ↓) |
| structure_distance (DINO ViT self-similarity) | structural drift | ↓ |
| CLIP similarity (full + edit-region) | text–video alignment to the target | ↑ |
| Motion-Fidelity-Score (CoTracker) | edited-object motion vs source-object motion | ↑ |
| NIQE (pyiqa) | no-reference perceptual quality | ↓ |

The **VLM judge (Gemini)** answers a structured checklist (each item `1 / 0.5 / 0`)
across the four dimensions and returns boolean **critical flags** used by the caps.
Questions are specific ("was the target replaced?", "does the original reappear?"),
never "how good is it?".

---

## 4. Scoring pipeline (worked example: `cup0_easy`)

**Step 1 — raw metrics** (per frame, averaged over stride-8 samples):
`psnr 32.13 · ssim 0.984 · lpips 0.033 · structure 0.0014 · clip 23.86 · niqe 8.69 · mfs 0.948`

**Step 2 — normalize to [0,1]** via `scoring.yaml` (`invert` for lower-is-better):

| psnr | ssim | lpips | structure | clip | niqe | mfs |
|---|---|---|---|---|---|---|
| 0.685 | 0.977 | 0.945 | 0.986 | 0.591 | **0.187** | 0.948 |

**Step 3 — combine into dimensions** (Gemini checklist means all = 1.0 here):

```
edit_success        = 0.7·1.0  + 0.3·0.591                          = 0.877
source_preservation = 0.5·outside_mask_visual + 0.5·1.0             = 0.949
    outside_mask_visual = 0.25·(0.685+0.977+0.945+0.986)            = 0.898
temporal_consistency= 0.5·0.948 + 0.5·1.0                           = 0.974
rendering_quality   = niqe                                          = 0.187
```

**Step 4 — weighted final:**
```
0.35·0.877 + 0.30·0.949 + 0.20·0.974 + 0.15·0.187 = 0.815
```
(RQ=0.187 is NIQE penalizing a low-resolution source; because RQ weighs only 0.15 it
drags the final by just 0.028 — the intended behavior.)

All weights / normalization ranges / blend weights are in [`scoring.yaml`](scoring.yaml).

---

## 5. Critical-failure caps

Weighted averages can *average away* a fatal failure (high preservation + temporal +
quality can hide that the edit never happened). So after weighting, hard ceilings apply
(the **min** across all fired flags):

| Flag (from the Gemini judge) | Cap |
| --- | --- |
| `edit_not_completed` — edit not done / target not replaced | final ≤ 0.40 |
| `source_object_reappears` — original object returns | final ≤ 0.50 |
| `subject_identity_destroyed` | final ≤ 0.50 |
| `background_camera_changed` | final ≤ 0.60 |
| `severe_temporal_flicker` | temporal dim ≤ 0.40 |

Tier-specific caps for the internal stress-test: `medium` silhouette-not-different →
edit_success ≤ 0.50, final ≤ 0.60; `hard` material/style-not-visible → edit_success = 0,
final ≤ 0.45.

Example: `cafe_table_with_cameras_easy` — dims ES 0.18 / SP 0.91 / TC 0.98 / RQ 0.81 →
weighted **0.65**, but the judge flagged `edit_not_completed` → capped to **0.40**.

---

## 6. Unified case schema & masks

Each case is one JSONL line (`data/internal/cases.jsonl`), with `benchmark_source`
labelling the *dataset* (`internal` | `five` | `ive`) — the **metrics are the same**
regardless of source. Internal cases carry `replacement_transformation_tier`
(easy/medium/hard) and `failure_tags` (shadow / occlusion / reflection / motion / …).

**Edit-region mask = the region the model was allowed to change** (source∪target,
RoMa-warped per frame). The pipeline **already produces this at generation time**
(`outputs/<name>/mask/`), so the evaluator **reuses it** (`mask_source: pipeline`) and
only regenerates (SAM3+RoMa, `regen_masks.py`) as a fallback for pre-existing videos
whose masks weren't kept. MFS likewise reuses the pipeline's `frames_src`. In the
agentic loop, generation just made these → eval reads them, no recomputation.

---

## 7. Aggregation & reports

`aggregate_results.py` produces:
- `summary.json` — overall, per-dimension, **by tier**, **by failure_tag**, by edit_type, `critical_failure_rate`
- `summary.csv` — one row per case
- `failure_gallery.html` — worst cases + judge reasoning, grouped by tier

---

## 8. Results (internal 30×3 = 90 cases)

| | Old (CLIP-weighted) | New (this benchmark) |
| --- | --- | --- |
| Overall | 0.793 | **0.764** |
| Discriminative spread (std) | 0.127 | **0.183 (+44%)** |
| Score range | 0.44–0.97 | 0.35–0.94 |
| Critical-failure rate | — (no caps) | **23.3 %** |

**Dimensions:** Edit Success 0.84 · Source Preservation 0.89 · Temporal 0.77 · Rendering 0.61
**By tier:** easy 0.72 · medium 0.77 · hard 0.80
**By failure tag:** motion **0.54** · sky 0.70 · reflection 0.70 · large 0.73 · occlusion 0.79 · shadow 0.82
**Caps fired:** edit_not_completed ×15 · severe_temporal_flicker ×11 · source_object_reappears ×6 · background_camera_changed ×3

Key takeaways for the report:
- The score is **more discriminative** and **surfaces where the model fails** — worst on
  **motion** scenes (expected: the model targets static objects / minimal camera motion).
- It **exposed a data flaw**: several *easy*-tier prompts are near **no-ops** (e.g.
  `laptop → "a silver laptop"` when already silver), which the old CLIP score rated ~0.85
  but the new judge correctly flags as "no edit happened" → easy became the lowest tier.

---

## 9. Known caveats (and how to tune)

- **MFS is weak on near-static scenes** (≈0.4 for most cases) → drags Temporal uniformly.
  Consider down-weighting MFS or adding adjacent-frame consistency. Tune in `scoring.yaml`.
- **NIQE is source-resolution-dependent** → penalizes low-res sources regardless of edit.
- **Gemini judge is stochastic** on borderline `edit_not_completed` flags → pin with a
  stricter prompt or majority vote for release runs.

---

## 10. Using the score in the agent

- **Retry gate:** `final_score` vs a threshold (accept ≥ 0.7, else retry).
- **Tuning agent:** the fired `caps` map to which knob to turn —
  `edit_not_completed`/`source_object_reappears` → `dilate`/`region_shape`/`ref0`;
  `severe_temporal_flicker` → `segment_starts`/`interpolate`; `background_camera_changed`
  → composite/mask. The full FiVE metrics are heavy per-candidate; in-loop use the
  lightweight subset (Gemini judge + caps) and reserve full scoring for final reporting.
