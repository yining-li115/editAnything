"""Path registry for the FiVE-based benchmark eval (single source of truth).

The eval spans two roots: this repo (`editAnything/`, holds `components/`) and its
parent project dir (holds the vendored FiVE-Bench + co-tracker + the results pack +
runtime data/outputs). Every external/data path is resolved here, env-overridable,
so moving the checkout only touches this file. Mirrors contracts/layout.py's role.
"""
import os

_EVAL = os.path.dirname(os.path.abspath(__file__))     # editAnything/eval
EDITANYTHING = os.path.dirname(_EVAL)                    # editAnything repo root
PROJECT = os.path.dirname(EDITANYTHING)                  # parent (FiVE-Bench/, co-tracker/, data/, outputs/)


def _env(k, default):
    return os.environ.get(k, default)


# --- vendored external tools (siblings of the repo, like RIFE/ROSE) ---
FIVE_BENCH     = _env("FIVE_BENCH_ROOT", os.path.join(PROJECT, "FiVE-Bench"))
COTRACKER_CKPT = _env("COTRACKER_CKPT",  os.path.join(PROJECT, "co-tracker", "checkpoints", "scaled_offline.pth"))

# --- data + outputs (runtime artifacts, gitignored) ---
RESULTS_PACK = _env("RESULTS_PACK", os.path.join(PROJECT, "editAnything_results_20260624"))
# where the pipeline writes its own run artifacts (contracts.layout RunPaths default);
# eval prefers the pipeline's own edit mask (outputs/<name>/mask) over regenerating.
PIPELINE_OUTPUTS = _env("PIPELINE_OUTPUTS", os.path.join(EDITANYTHING, "outputs"))
DATA_ROOT    = _env("BENCH_DATA",   os.path.join(PROJECT, "data"))
OUT_ROOT     = _env("BENCH_OUT",    os.path.join(PROJECT, "outputs"))

SCORING_YAML = os.path.join(_EVAL, "scoring.yaml")
CASES_JSONL  = os.path.join(DATA_ROOT, "internal", "cases.jsonl")
MASK_ROOT    = os.path.join(DATA_ROOT, "internal", "masks")
CACHE_ROOT   = os.path.join(OUT_ROOT, "cache", "internal")
EVAL_RESULTS = os.path.join(OUT_ROOT, "eval_results")
REPORTS      = os.path.join(OUT_ROOT, "reports")
