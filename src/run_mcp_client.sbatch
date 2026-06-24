#!/bin/bash
#SBATCH --job-name=mcp_client
#SBATCH --partition=PRACT
#SBATCH --qos=practical_course
#SBATCH --nodelist=node9
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/storage/slurm/s0037/logs/slurm-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate videopainter
export HF_HOME="/storage/slurm/s0037/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /usr/prakt/s0037/editAnything
python src/mcp/mcp_client.py \
  --prompt "replace the cup with a cyberpunk banana" \
  --frame /storage/slurm/s0037/outputs/orchestrator_test/frame_00001.png \
  --frames_dir /storage/slurm/s0037/input/frames_2chunk \
  --out_dir /storage/slurm/s0037/outputs/orchestrator_test \
  --existing_video /storage/slurm/s0037/outputs/cup2_output.mp4