"""Command-line VideoPainter: SAM3 text-prompt tracking + FLUX/CogVideoX inpainting.

This is the headless (no-Gradio) entry point. It reuses the exact pure logic from
app.py (frame loading, SAM3 tracking, inpainting) but drives it straight from the
command line, so you can give a video + text prompts and get an output video.

It does NOT touch utils.py / generate_frames or the FLUX inpainting logic.

Example:
    CUDA_VISIBLE_DEVICES=0 python run_cli.py \
        --video ../../../cup2.mp4 \
        --object_prompt "cup" \
        --video_caption "A cartoon banana sits on a wooden table ..." \
        --target_caption "a cartoon banana" \
        --output ./output.mp4 \
        --model_path ../ckpt/CogVideoX-5b-I2V \
        --inpainting_branch ../ckpt/VideoPainter/checkpoints/branch \
        --id_adapter ../ckpt/VideoPainterID/checkpoints \
        --img_inpainting_model ../ckpt/flux_inp
"""
import os
# Match app.py: keep a writable temp dir for the intermediate artifacts that
# generate_frames() insists on saving (first_frame.png, first_mask.png, ...).
GRADIO_TEMP_DIR = "./tmp_gradio"
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.makedirs(f"{GRADIO_TEMP_DIR}/track", exist_ok=True)
os.makedirs(f"{GRADIO_TEMP_DIR}/inpaint", exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

import warnings
warnings.filterwarnings("ignore")

import argparse
import shutil

import cv2
import numpy as np
import scipy.ndimage
import torch
import torchvision
from PIL import Image
from decord import VideoReader

from utils import load_model, generate_frames
from sam3.model_builder import build_sam3_video_predictor


# ---------------------------------------------------------------------------
# Pure logic lifted from app.py (no Gradio).
# ---------------------------------------------------------------------------
def load_frames(video_path):
    """Same preprocessing as app.py:get_frames_from_video.

    decord read -> downsample to 8 fps -> first 49 frames -> resize to 720x480.
    Returns (frames, (orig_w, orig_h)):
      - frames: (N, 480, 720, 3) uint8 RGB numpy array (the 720x480 work resolution)
      - orig_w, orig_h: the native frame size, so the final output can be restored
        back to the original aspect (VideoPainter squishes everything to 720x480).
    """
    vr = VideoReader(video_path)
    original_fps = vr.get_avg_fps()

    if original_fps > 8:
        total_frames = len(vr)
        sample_interval = max(1, int(original_fps / 8))
        frame_indices = list(range(0, total_frames, sample_interval))
        frames = vr.get_batch(frame_indices).asnumpy()
    else:
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

    frames = frames[:49]

    orig_h, orig_w = frames[0].shape[0:2]
    resized_frames = [cv2.resize(frame, (720, 480)) for frame in frames]
    return np.array(resized_frames), (orig_w, orig_h)


def track_with_sam3(sam3_predictor, origin_images, object_prompt, work_dir):
    """Same SAM3 tracking core as app.py:auto_track_with_sam3.

    Dumps the (already resampled/resized) frames to a temp JPEG folder so the
    masks line up frame-for-frame, then runs start_session / add_prompt /
    propagate_in_video / close_session. Merges all matched instances per frame
    into one binary mask, resizes to frame size, dilates (iterations=6).

    Returns an (N, H, W, 1) uint8 array of 0/1 values, ready for inpainting.
    """
    object_prompt = str(object_prompt).strip()
    if not object_prompt:
        raise ValueError("--object_prompt must be a non-empty text prompt (e.g. 'cup').")

    num_frames = len(origin_images)
    height, width = origin_images[0].shape[0:2]

    frames_dir = os.path.join(work_dir, "sam3_frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(origin_images):
        # origin_images are RGB numpy; cv2.imwrite expects BGR
        cv2.imwrite(
            os.path.join(frames_dir, f"{i}.jpg"),
            cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR),
        )

    response = sam3_predictor.handle_request(
        request=dict(type="start_session", resource_path=frames_dir)
    )
    session_id = response["session_id"]
    outputs_per_frame = {}
    try:
        sam3_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=object_prompt,
            )
        )
        for resp in sam3_predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs_per_frame[resp["frame_index"]] = resp["outputs"]
    finally:
        sam3_predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )

    masks = []
    for frame_idx in range(num_frames):
        merged = np.zeros((height, width), dtype=np.uint8)
        out = outputs_per_frame.get(frame_idx)
        if out is not None and len(out["out_obj_ids"]) > 0:
            binary_masks = out["out_binary_masks"]
            if hasattr(binary_masks, "cpu"):
                binary_masks = binary_masks.cpu().numpy()
            binary_masks = np.asarray(binary_masks)  # (N_obj, h, w)
            union = np.any(binary_masks > 0.5, axis=0).astype(np.uint8)  # (h, w)
            if union.shape != (height, width):
                union = cv2.resize(
                    union, (width, height), interpolation=cv2.INTER_NEAREST
                )
            merged = (union > 0).astype(np.uint8)
        merged = scipy.ndimage.binary_dilation(merged, iterations=6).astype(np.uint8)
        masks.append(merged[:, :, None])
    masks = np.array(masks)  # (num_frames, H, W, 1)

    n_hit = sum(1 for f in range(num_frames) if masks[f].any())
    print(f"SAM3 tracked '{object_prompt}': masks={masks.shape}, frames with mask={n_hit}/{num_frames}")
    if n_hit == 0:
        print("WARNING: SAM3 matched the prompt on 0 frames. The output will be unchanged. "
              "Try a different / simpler object prompt.")
    return masks


def generate_video_from_frames(frames, output_path, fps=8):
    """Same as app.py:generate_video_from_frames."""
    frames = torch.from_numpy(np.asarray(frames)).to(torch.uint8)
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


def generate_frames_no_flux(pipe, images, masks, prompt, seed=42, cfg_scale=6.0, dilate_size=16):
    """Same as utils.generate_frames but WITHOUT the FLUX first-frame inpainting.

    Used when an externally edited first frame is supplied (e.g. a Gemini
    replacement): images[0] is taken as-is and fed straight to CogVideoX as the
    frame-0 condition, so the masked region's appearance comes from whatever you
    put in images[0] instead of from FLUX. Everything else (mask dilation, the
    CogVideoX call, return shape) mirrors generate_frames exactly. utils.py is
    not touched; this just omits the FLUX block.
    """
    os.makedirs(f"{GRADIO_TEMP_DIR}/inpaint", exist_ok=True)
    images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame.png")
    masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask.png")
    masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask.png")

    print(f"Dilating the mask with size {dilate_size}...")
    if dilate_size and int(dilate_size) > 0:
        for i in range(len(masks)):
            mask = cv2.dilate(np.array(masks[i]), np.ones((int(dilate_size), int(dilate_size))))
            masks[i] = Image.fromarray(mask.astype(np.uint8))
    masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask_dilate.png")
    masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask_dilate.png")

    # NO FLUX: keep the supplied first frame untouched.
    images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame_used.png")
    print("Skipping FLUX; using the supplied first frame as the CogVideoX condition.")

    # Clear the frame-0 mask so CogVideoX treats it as ground-truth condition (same as generate_frames).
    masks[0] = Image.fromarray(np.zeros_like(np.array(masks[0]))).convert("RGB")

    inpaint_outputs = pipe(
        prompt=prompt,
        image=images[0],
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=cfg_scale,
        generator=torch.Generator().manual_seed(seed),
        video=images,
        masks=masks,
        strength=1.0,
        replace_gt=True,
        mask_add=True,
        stride=int(49 - 0),
        prev_clip_weight=0.0,
        id_pool_resample_learnable=False,
        output_type="np",
    ).frames[0]
    inpaint_outputs = inpaint_outputs[1:]
    print(f"Video inpainting (no-FLUX) done! {np.array(inpaint_outputs).shape}")
    torch.cuda.empty_cache()
    return inpaint_outputs


def run_inpaint(pipe, pipe_img, origin_images, masks, video_caption,
                target_caption, seed, cfg_scale, dilate_size, first_frame_image=None):
    """Same image/mask prep as app.py:inpaint_video, then run inpainting.

    If first_frame_image is given, the supplied image replaces frame 0 and FLUX
    is skipped (generate_frames_no_flux); otherwise the normal FLUX path
    (utils.generate_frames) is used.
    """
    validation_images = origin_images[list(range(0, len(origin_images), 1))]
    validation_masks = masks[list(range(0, len(origin_images), 1))]

    validation_masks = [np.squeeze(mask) for mask in validation_masks]
    validation_masks = [(mask > 0).astype(np.uint8) * 255 for mask in validation_masks]
    validation_masks = [np.stack([m, m, m], axis=-1) for m in validation_masks]

    validation_images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in validation_images]
    validation_masks = [Image.fromarray(np.uint8(mask)).convert("RGB") for mask in validation_masks]

    validation_images = [img.resize((720, 480)) for img in validation_images]
    validation_masks = [mask.resize((720, 480)) for mask in validation_masks]

    if first_frame_image is not None:
        ff = Image.open(first_frame_image).convert("RGB")
        if ff.size != (720, 480):
            # The whole pipeline works at 720x480; the video frames are squished to
            # this size too, so an external first frame with the same aspect as the
            # video stays aligned after the same squish. (Restored to the original
            # resolution at the very end.)
            print(f"Resizing --first_frame_image from {ff.size} to (720, 480) (the work resolution).")
            ff = ff.resize((720, 480))
        validation_images[0] = ff
        # FLUX (pipe_img) is loaded on the GPU by load_model and normally moved to
        # CPU *inside* generate_frames after it runs. The no-FLUX path skips that
        # block, so offload it here or it keeps ~24GB resident and CogVideoX OOMs.
        if pipe_img is not None:
            pipe_img.to("cpu")
            torch.cuda.empty_cache()
        images = generate_frames_no_flux(
            pipe=pipe,
            images=validation_images,
            masks=validation_masks,
            prompt=str(video_caption),
            seed=seed,
            cfg_scale=float(cfg_scale),
            dilate_size=int(dilate_size),
        )
    else:
        images = generate_frames(
            images=validation_images,
            masks=validation_masks,
            pipe=pipe,
            pipe_img_inpainting=pipe_img,
            prompt=str(video_caption),
            image_inpainting_prompt=str(target_caption),
            seed=seed,
            cfg_scale=float(cfg_scale),
            dilate_size=int(dilate_size),
        )
    return (images * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Headless VideoPainter (SAM3 text prompt + FLUX/CogVideoX inpainting)")
    # inputs
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--object_prompt", required=True,
                        help="Object to segment/replace, SAM3 text prompt (e.g. 'cup').")
    parser.add_argument("--video_caption", required=True,
                        help="Global video caption describing the desired result.")
    parser.add_argument("--target_caption", default="",
                        help="Target object caption for the masked region (FLUX first-frame inpainting). "
                             "Required unless --first_frame_image is given.")
    parser.add_argument("--first_frame_image", default=None,
                        help="Path to an externally edited first frame (e.g. a Gemini replacement). "
                             "If set, FLUX is skipped and this image is used as the CogVideoX frame-0 "
                             "condition. Best results: an in-place edit of the real 720x480 first frame.")
    parser.add_argument("--output", default="./output.mp4", help="Output video path.")
    # output resolution
    parser.add_argument("--output_size", default=None,
                        help="Final output size as WxH (e.g. 540x1024). Default: restore the "
                             "original video resolution (VideoPainter works at 720x480 internally).")
    parser.add_argument("--no_restore", action="store_true",
                        help="Keep the 720x480 work resolution instead of restoring the original.")
    # sampling
    parser.add_argument("--seed", type=int, default=42, help="Inference seed (-1 for random).")
    parser.add_argument("--cfg_scale", type=float, default=6.0, help="Classifier-free guidance scale.")
    parser.add_argument("--dilate_size", type=int, default=16, help="Mask dilate size used inside generate_frames.")
    # model paths (mirror app.sh)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--inpainting_branch", required=True)
    parser.add_argument("--id_adapter", required=True)
    parser.add_argument("--img_inpainting_model", required=True)
    args = parser.parse_args()

    if args.first_frame_image is None and not str(args.target_caption).strip():
        parser.error("--target_caption is required when --first_frame_image is not given (FLUX path).")

    seed = int(args.seed) if int(args.seed) >= 0 else int(np.random.randint(0, 2 ** 32 - 1))

    # 1) Build models (same as app.py top-level).
    print("Building SAM3 video predictor...")
    sam3_predictor = build_sam3_video_predictor(
        gpus_to_use=range(torch.cuda.device_count()) if torch.cuda.is_available() else None
    )
    print("Build SAM3 video predictor done!")

    print("Loading VideoPainter + FLUX models...")
    pipe, pipe_img = load_model(
        model_path=args.model_path,
        inpainting_branch=args.inpainting_branch,
        id_adapter=args.id_adapter,
        img_inpainting_model=args.img_inpainting_model,
    )
    print("Load model done!")

    # 2) Frames -> SAM3 masks -> inpaint -> write video.
    print(f"Loading frames from {args.video} ...")
    origin_images, orig_size = load_frames(args.video)
    print(f"Loaded {len(origin_images)} frames, shape={origin_images.shape}, original size(WxH)={orig_size}")

    # Decide the final output size (VideoPainter processes at 720x480 internally).
    if args.no_restore:
        out_size = (720, 480)
    elif args.output_size:
        w, h = (int(v) for v in str(args.output_size).lower().split("x"))
        out_size = (w, h)
    else:
        out_size = orig_size

    masks = track_with_sam3(sam3_predictor, origin_images, args.object_prompt, GRADIO_TEMP_DIR)

    if args.first_frame_image is not None:
        print(f"Using supplied first frame (FLUX skipped): {args.first_frame_image}")
    print("Running inpainting (this may take a while)...")
    images = run_inpaint(
        pipe, pipe_img, origin_images, masks,
        video_caption=args.video_caption,
        target_caption=args.target_caption,
        seed=seed,
        cfg_scale=args.cfg_scale,
        dilate_size=args.dilate_size,
        first_frame_image=args.first_frame_image,
    )

    # Restore to the target output resolution (images are at the 720x480 work size).
    if out_size != (720, 480):
        print(f"Restoring output resolution to {out_size[0]}x{out_size[1]} ...")
        images = np.array([cv2.resize(f, out_size, interpolation=cv2.INTER_LANCZOS4) for f in images])

    out_path = generate_video_from_frames(images, output_path=args.output, fps=8)
    print(f"Done! Output written to: {out_path} ({images.shape[2]}x{images.shape[1]})")


if __name__ == "__main__":
    main()
