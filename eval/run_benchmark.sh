#!/bin/bash
# Full internal-benchmark run: generate all configs -> manifest -> eval -> aggregate.
# Single-GPU, single-chunk (49f), source_mask=track, VP_OFFLOAD=none (fastest single
# run; 44GB resident so workers=1). Everything is resumable: outputs/<name>/final.mp4
# is the durable artifact, --skip-done skips finished generations, run_eval caches
# per-case results/<name>.json. If eval breaks, just re-run steps 2-4 — no regen.
set -u
cd /root/project/editAnything

export HF_HOME=/root/project/editAnything/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VP_OFFLOAD=none
export RIFE_BIN=/root/project/tools/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan
PY=/venv/editanything/bin/python

echo "===== BENCHMARK START $(date) ====="
echo "configs: $(ls eval/configs/*.yaml | wc -l)"

echo "[1/4] generation batch (workers=1, offload=none) $(date)"
$PY eval/run_batch.py --configs eval/configs --workers 1 --skip-done

echo "[2/4] build manifest $(date)"
$PY eval/build_manifest.py --configs eval/configs

echo "[3/4] run eval — CLIP/temporal/DINO/VLM $(date)"
$PY eval/run_eval.py --manifest eval/manifest.json

echo "[4/4] aggregate $(date)"
$PY eval/aggregate.py --results eval/results

echo "===== BENCHMARK DONE $(date) ====="
