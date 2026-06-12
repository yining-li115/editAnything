#!/bin/bash
# Headless VideoPainter: SAM3 text-prompt tracking + FLUX/CogVideoX inpainting.
# Usage:
#   FLUX path:   ./run_cli.sh <video> <object_prompt> <video_caption> <target_caption> [output]
#   Gemini path: ./run_cli.sh <video> <object_prompt> <video_caption> "" [output] <first_frame_image>
#   <video>          path to input video
#   <object_prompt>  SAM3 text prompt of the object to replace, e.g. "cup"
#   <video_caption>  global caption describing the desired result
#   <target_caption> caption for the masked region (FLUX first-frame inpainting);
#                    pass "" when using a supplied first frame
#   [output]         output path (default ./output.mp4)
#   <first_frame_image> optional: external first frame (e.g. Gemini); if set, FLUX is skipped
set -e

VIDEO="${1:?need video path}"
OBJECT_PROMPT="${2:?need object_prompt}"
VIDEO_CAPTION="${3:?need video_caption}"
TARGET_CAPTION="${4:-}"
OUTPUT="${5:-./output.mp4}"
FIRST_FRAME_IMAGE="${6:-}"

EXTRA=()
if [ -n "$FIRST_FRAME_IMAGE" ]; then
    EXTRA+=(--first_frame_image "$FIRST_FRAME_IMAGE")
fi

CUDA_VISIBLE_DEVICES=0 python run_cli.py \
    --video "$VIDEO" \
    --object_prompt "$OBJECT_PROMPT" \
    --video_caption "$VIDEO_CAPTION" \
    --target_caption "$TARGET_CAPTION" \
    --output "$OUTPUT" \
    --model_path ../ckpt/CogVideoX-5b-I2V \
    --inpainting_branch ../ckpt/VideoPainter/checkpoints/branch \
    --id_adapter ../ckpt/VideoPainterID/checkpoints \
    --img_inpainting_model ../ckpt/flux_inp \
    "${EXTRA[@]}"
