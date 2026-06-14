"""VideoPainter multi-chunk generation — load models ONCE, loop segments in-process.

This is the only place we touch VideoPainter: its custom `diffusers` fork
(CogVideoXI2VDualInpaintAnyLPipeline + CogvideoXBranchModel + the id_pool
transformer) + the three checkpoints. SAM3 / Gemini / FLUX are NOT involved here
— masks and the per-segment first frame (anchor) are passed in as plain inputs,
so this stage is fully decoupled.

It is a refactor of the validated `run_replacement.py` driver (exp3 reanchor):
same 720x480 work res, same pipe() call and parameters (replace_gt / mask_add /
strength=1.0 / id_pool_resample_learnable=True / DPM trailing / sequential CPU
offload + VAE tiling). The only change is structural: `load_pipeline()` runs once
and `run_segment()` is called per segment, instead of spawning a fresh process
(and reloading ~10GB of weights) for each of the 7 segments.

Per-segment "reanchor": the long video is split into 49-frame segments whose
starts are derived from the clip length (`default_segments`, length-adaptive — no
hardcoded frame count). Each segment is generated as one 49-frame clip conditioned
on its own clean anchor frame, which is what stops the inserted object from
dissolving after the first clip.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import torch
import cv2
from PIL import Image

from diffusers import (
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintAnyLPipeline,
)

# Work resolution and clip length are fixed by the base model (CogVideoX-5B-I2V).
W, H, CLIP = 720, 480, 49

def default_segments(n, clip=CLIP, step=48):
    """Auto segment starts for an n-frame clip: every `step` frames + a tail window
    so the last `clip` frames are always covered. Length-adaptive — works for ANY
    video length (no hardcoded frame count)."""
    starts = list(range(0, max(1, n - clip + 1), step))
    tail = n - clip
    if tail > starts[-1]:
        starts.append(tail)
    return starts


def load_pipeline(model_path, branch, id_lora, dtype=torch.bfloat16, device=None):
    """Build the VideoPainter native long-video ID-resample pipeline (load once).

    Mirrors run_replacement.py exactly: id_pool transformer + VideoPainterID LoRA,
    DPM trailing scheduler, sequential CPU offload + VAE slicing/tiling.
    """
    branch_model = CogvideoXBranchModel.from_pretrained(branch, torch_dtype=dtype)
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype,
        id_pool_resample_learnable=True,
    )
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        model_path, branch=branch_model, transformer=transformer, torch_dtype=dtype,
    )
    pipe.load_lora_weights(
        id_lora, weight_name="pytorch_lora_weights.safetensors",
        adapter_name="test_1", target_modules=["transformer"],
    )
    for m in (pipe.text_encoder, pipe.transformer, pipe.vae, pipe.branch):
        m.requires_grad_(False)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    # Memory savers — same as the validated run; keeps peak VRAM ~22GB.
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def _frame_names(frames_dir):
    return [os.path.basename(f) for f in sorted(glob.glob(f"{frames_dir}/frame_*.png"))]


def load_segment_inputs(frames_dir, mask_dir, start, total, dilate):
    """Load `total` base frames + matching masks starting at index `start`.

    Returns (video_pil, masks_pil, names). Mirrors run_replacement.py:
    resize to 720x480, mask binarized + dilated, mask kept as RGB.
    """
    names = _frame_names(frames_dir)[start:start + total]
    video = [
        Image.fromarray(cv2.cvtColor(
            cv2.resize(cv2.imread(f"{frames_dir}/{n}"), (W, H)), cv2.COLOR_BGR2RGB))
        for n in names
    ]
    k = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    masks = []
    for n in names:
        m = cv2.resize(cv2.imread(f"{mask_dir}/{n}", 0), (W, H),
                       interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.uint8)
        if k is not None:
            m = cv2.dilate(m, k)
        masks.append(Image.fromarray(np.where(m > 0, 255, 0).astype(np.uint8)).convert("RGB"))
    return video, masks, names


def run_segment(pipe, video, masks, first_frame, prompt, *, steps=50, guidance=6.0,
                seed=42, overlap_frames=0, prev_clip_weight=0.5):
    """Generate one segment. video/masks are PIL lists (already 720x480).

    The first frame is replaced by the clean anchor and its mask zeroed (treated
    as the I2V ground-truth condition), exactly as in run_replacement.py. Returns
    a list of uint8 RGB frames (one per input frame of the segment).
    """
    ff = first_frame.convert("RGB").resize((W, H), Image.LANCZOS)
    video = list(video)
    masks = list(masks)
    video[0] = ff
    masks[0] = Image.fromarray(np.zeros((H, W, 3), np.uint8)).convert("RGB")

    # The base model always emits a full CLIP-length (49) clip. For a short video
    # (seg_len < CLIP) pad the context by repeating the last frame/mask so the clip
    # is full, then truncate the output back to the real length. For full segments
    # (seg_len == CLIP, the normal multi-chunk case) this is a no-op.
    seg_len = len(video)
    if seg_len < CLIP:
        video = video + [video[-1]] * (CLIP - seg_len)
        masks = masks + [masks[-1]] * (CLIP - seg_len)

    out = pipe(
        prompt=prompt, image=ff, height=H, width=W, num_videos_per_prompt=1,
        num_inference_steps=steps, num_frames=CLIP, use_dynamic_cfg=True,
        guidance_scale=guidance, generator=torch.Generator().manual_seed(seed),
        video=video, masks=masks,
        strength=1.0, replace_gt=True, mask_add=True,
        stride=int(CLIP - overlap_frames),
        prev_clip_weight=prev_clip_weight,
        id_pool_resample_learnable=True,
        output_type="np",
    ).frames[0]
    out = out[:seg_len]
    return [(np.array(f) * 255).astype(np.uint8) if f.dtype != np.uint8 else np.array(f)
            for f in out]


def generate(pipe, frames_dir, mask_dir, anchor_for_start, out_dir, *,
             segment_starts=None, total=CLIP, prompt="", dilate=12, steps=50,
             guidance=6.0, seed=42, overlap_frames=0, prev_clip_weight=0.5):
    """Run all segments with a single loaded pipeline.

    anchor_for_start: callable start -> PIL.Image (the segment's clean anchor).
    Frames are written to {out_dir}/frames/frame_{start+i+1:05d}.png; later
    segments overwrite the 1-frame boundary overlap, yielding a contiguous set.
    """
    if segment_starts is None:                       # derive from the actual clip length
        segment_starts = default_segments(len(_frame_names(frames_dir)))
    frames_out = os.path.join(out_dir, "frames")
    os.makedirs(frames_out, exist_ok=True)
    for s in segment_starts:
        video, masks, names = load_segment_inputs(frames_dir, mask_dir, s, total, dilate)
        anchor = anchor_for_start(s)
        print(f"[generate] segment start={s} frames={len(video)} anchor->{anchor.size}")
        out = run_segment(pipe, video, masks, anchor, prompt, steps=steps,
                          guidance=guidance, seed=seed, overlap_frames=overlap_frames,
                          prev_clip_weight=prev_clip_weight)
        for i, im in enumerate(out):
            cv2.imwrite(f"{frames_out}/frame_{s + i + 1:05d}.png",
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        print(f"[generate]   wrote {len(out)} frames; peak VRAM="
              f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    return frames_out


# --- CLI: single segment (back-compat with run_replacement.py) or all segments ---
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="VideoPainter multi-chunk generation (load-once)")
    ap.add_argument("--frames_dir", required=True, help="dir of base frames frame_*.png (with object)")
    ap.add_argument("--mask_dir", required=True, help="dir of per-frame masks frame_*.png")
    ap.add_argument("--anchor_dir", required=True, help="dir of per-segment anchors anchor_<start:04d>.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt", required=True, help="global generation prompt for the edited video")
    ap.add_argument("--segment_starts", default=None,
                    help="comma list, e.g. 0,48,96 or a single start for one chunk "
                         "(default: auto from frame count via default_segments)")
    ap.add_argument("--total", type=int, default=CLIP)
    ap.add_argument("--model_path", default="ckpt/CogVideoX-5b-I2V")
    ap.add_argument("--branch", default="ckpt/VideoPainter/checkpoints/branch")
    ap.add_argument("--id_lora", default="ckpt/VideoPainterID/checkpoints")
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    starts = ([int(x) for x in args.segment_starts.split(",")] if args.segment_starts
              else default_segments(len(_frame_names(args.frames_dir))))

    def anchor_for_start(s):
        return Image.open(f"{args.anchor_dir}/anchor_{s:04d}.png")

    pipe = load_pipeline(args.model_path, args.branch, args.id_lora)
    out = generate(pipe, args.frames_dir, args.mask_dir, anchor_for_start, args.out_dir,
                   segment_starts=starts, total=args.total, prompt=args.prompt,
                   dilate=args.dilate, steps=args.steps, guidance=args.guidance, seed=args.seed)
    print(f"DONE -> {out}")
