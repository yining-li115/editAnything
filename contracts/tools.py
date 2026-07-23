"""Tool schema registry — canonical input/output definitions for every MCP tool.

The MCP server reads these to register tools; callers (orchestrator, tests) read
them to know what to pass. Keeping schemas here prevents drift between the server
registration and the actual handler signatures.

Each entry is:
  "tool_name": {
      "description": str,
      "inputs":  { param: {"type": ..., "description": ..., "default": ...} },
      "outputs": { key:   {"type": ..., "description": ...} },
  }
Types use JSON-schema vocabulary (string, integer, number, boolean, array, object, null).
"""

TOOLS: dict = {

    # ------------------------------------------------------------------
    # T2 — extract frames from video
    # ------------------------------------------------------------------
    "extract_frames": {
        "description": (
            "Decode a video into individual PNG frames (frame_00001.png …). "
            "Also returns n_frames and native_size so the orchestrator can plan "
            "segment starts without a second call."
        ),
        "inputs": {
            "video_path":  {"type": "string",  "description": "Absolute path to the input video file."},
            "out_dir":     {"type": "string",  "description": "Directory where frame_*.png will be written."},
            "max_frames":  {"type": ["integer", "null"], "description": "Decode only the first N frames.", "default": None},
            "resume":      {"type": "boolean", "description": "Skip extraction if out_dir already has frames.", "default": False},
        },
        "outputs": {
            "frames_dir":  {"type": "string",  "description": "Absolute path to the frames directory."},
            "n_frames":    {"type": "integer", "description": "Total number of frames extracted."},
            "native_size": {"type": "array",   "description": "[W, H] of the source frames."},
        },
    },

    # ------------------------------------------------------------------
    # T3 — SAM3 text-prompt source mask
    # ------------------------------------------------------------------
    "sam3_mask": {
        "description": (
            "Run SAM3 text-prompt tracking across all frames to produce a "
            "per-frame 0/255 mask for the SOURCE object (the one being replaced). "
            "Returns immediately if resume=true and out_dir already has masks."
        ),
        "inputs": {
            "frames_dir":  {"type": "string",  "description": "Directory of frame_*.png."},
            "source_word": {"type": "string",  "description": "Object noun to segment, e.g. 'cup'."},
            "out_dir":     {"type": "string",  "description": "Directory to write per-frame masks."},
            "resume":      {"type": "boolean", "description": "Reuse existing masks if present.", "default": False},
        },
        "outputs": {
            "mask_dir":  {"type": "string",  "description": "Absolute path to the mask directory."},
            "n_masks":   {"type": "integer", "description": "Number of mask frames written."},
        },
    },

    # ------------------------------------------------------------------
    # T3b — camera-motion detector (routes generator/anchor choice)
    # ------------------------------------------------------------------
    "detect_camera_motion": {
        "description": (
            "Decide if the clip has CAMERA motion (pan/dolly/zoom/handheld) vs "
            "just OBJECT motion, via background-only ORB+RANSAC homography "
            "(excludes the SOURCE mask so the edited object is never mistaken "
            "for camera motion). Call right after sam3_mask, passing its "
            "mask_dir. camera_motion=false -> roma_anchors + "
            "videopainter_generate; camera_motion=true -> mvinpainter_anchors "
            "+ mvinpainter_generate."
        ),
        "inputs": {
            "frames_dir":    {"type": "string", "description": "ORIGINAL source frames directory (frame_*.png)."},
            "mask_dir":      {"type": ["string", "null"], "description": "SOURCE object mask dir (sam3_mask's output). Recommended.", "default": None},
            "stride":        {"type": "integer", "description": "Frame gap between sampled pairs.", "default": 6},
            "transform_frac_threshold":  {"type": "number",  "description": "Min displacement (as a fraction of frame diagonal) for a homography fit to count as coherent. Resolution-independent.", "default": 0.027},
            "inlier_ratio_thresh": {"type": "number", "description": "Min RANSAC inlier ratio to count as coherent; lower for handheld/parallax footage.", "default": 0.3},
            "coherent_fraction_thresh": {"type": "number", "description": "Min fraction of coherent pairs needed for camera_motion=true.", "default": 0.5},
            "raw_motion_frac_threshold": {"type": "number", "description": "Fallback: median displacement fraction alone triggers camera_motion=true even without a coherent homography (catches parallax).", "default": 0.045},
            "fps":           {"type": "number", "description": "Clip fps, used for the min_duration_sec gate.", "default": 25.0},
            "min_duration_sec": {"type": "number", "description": "Clips shorter than this are forced camera_motion=false, skipping detection. 0 disables.", "default": 3.0},
        },
        "outputs": {
            "camera_motion":        {"type": "boolean", "description": "True if the clip shows camera motion."},
            "coherent_fraction":    {"type": "number",  "description": "Fraction of sampled pairs with a coherent homography fit."},
            "median_transform_frac": {"type": "number",  "description": "Median implied background displacement across sampled pairs, as a fraction of frame diagonal."},
            "n_pairs_sampled":      {"type": "integer", "description": "Number of frame pairs evaluated."},
        },
    },


    # ------------------------------------------------------------------
    # T4 — RoMa edit-mask (frame-0 region warped per frame)
    # ------------------------------------------------------------------
    "roma_edit_mask": {
        "description": (
            "Build the frame-0 (target∪source) edit region and warp it to every "
            "frame with RoMa dense matching. The result is the per-frame mask "
            "that tells VideoPainter (or any inpainter) WHERE to repaint."
        ),
        "inputs": {
            "frames_dir":     {"type": "string",  "description": "Source frames directory."},
            "ref0_path":      {"type": "string",  "description": "Path to the edited frame-0 reference image (new object)."},
            "target_word":    {"type": "string",  "description": "Noun for SAM3 to segment the NEW object on ref0."},
            "source_word":    {"type": "string",  "description": "Noun for SAM3 to segment the OLD object on frame 0."},
            "work_dir":       {"type": "string",  "description": "Working directory for intermediate files (roma/)."},
            "ref0_mask_path": {"type": ["string", "null"], "description": "Optional pre-computed mask for the new object on ref0.", "default": None},
            "dilate":         {"type": "integer", "description": "Dilation in pixels for the edit region margin.", "default": 12},
            "region_shape":   {"type": "string",  "description": "hull | rect | bbox — how the frame-0 region is shaped.", "default": "hull"},
        },
        "outputs": {
            "mask_dir": {"type": "string", "description": "Directory of per-frame edit-region masks."},
        },
    },

    # ------------------------------------------------------------------
    # T5 — RoMa per-segment anchors
    # ------------------------------------------------------------------
    "roma_anchors": {
        "description": (
            "Warp the clean ref0 image into each segment-start viewpoint using "
            "RoMa dense matching. These anchors are the I2V conditioning frames "
            "that stop the inserted object dissolving after the first chunk."
        ),
        "inputs": {
            "frames_dir":      {"type": "string", "description": "Source frames directory."},
            "ref0_path":       {"type": "string", "description": "Clean first-frame reference (new object in scene)."},
            "work_dir":        {"type": "string", "description": "Working directory; anchors go into work_dir/anchors/."},
            "segment_starts":  {"type": "array", "items": {"type": "integer"},
                                 "description": "List of 0-indexed frame numbers that start each segment."},
        },
        "outputs": {
            "anchor_map": {
                "type": "object",
                "description": "Dict mapping str(start) -> absolute anchor image path.",
            },
        },
    },


    # ------------------------------------------------------------------
    # T5b — MVInpainter per-segment anchors (large camera motion)
    # ------------------------------------------------------------------
    "mvinpainter_anchors": {
        "description": (
            "Per-segment anchors via MVInpainter's own multi-view pass instead "
            "of RoMa warping — stays clean at large viewpoint changes where "
            "RoMa shears. Use with mvinpainter_generate when camera_motion=true. "
            "Slower (subprocess in its own env). Auto-adjusts segment_starts to match chunk size."
        ),
        "inputs": {
            "frames_dir":      {"type": "string", "description": "Source frames directory."},
            "ref0_path":       {"type": "string", "description": "Clean first-frame reference (new object in scene)."},
            "mask_dir":        {"type": "string", "description": "Edit-region mask (same one videopainter/mvinpainter generation uses)."},
            "work_dir":        {"type": "string", "description": "Working directory for the anchor pass outputs."},
            "segment_starts":  {"type": "array", "items": {"type": "integer"},
                                 "description": "0-indexed frame numbers that start each segment."},
            "n_views":         {"type": "integer", "description": "Sampled anchor views (<=32, model PE cap).", "default": 24},
            "prompt":          {"type": "string",  "description": "Generation prompt for the anchor pass.", "default": ""},
            "name":            {"type": "string",  "description": "Run name for the anchor pass's output folder.", "default": "mvi_anchor"},
            "steps":           {"type": ["integer", "null"], "description": "Diffusion inference steps (default 50); lower = faster, some quality cost. null = component default.", "default": None},
            "chunk":           {"type": "integer", "description": "Frame chunk size for mvinpainter_generate; auto-adjusts segment_starts if it detects 48-frame stride.", "default": 20},
        },
        "outputs": {
            "anchor_map": {
                "type": "object",
                "description": "Dict mapping str(start) -> absolute anchor image path.",
            },
            "segment_starts": {
                "type": "array",
                "description": "Auto-adjusted segment_starts if they were recomputed to match chunk size, otherwise echoes input.",
            },
        },
    },

    # ------------------------------------------------------------------
    # T6 — Gemini frame-0 edit (generate the ref0)
    # ------------------------------------------------------------------
    "gemini_edit": {
        "description": (
            "Call the Gemini image model to edit a single frame in-place, "
            "swapping the source object for the target. Returns the path to the "
            "edited reference image (ref0) that feeds into roma_anchors."
        ),
        "inputs": {
            "frame0_path": {"type": "string",          "description": "Path to the first video frame."},
            "out_path":    {"type": "string",          "description": "Where to save the edited reference image."},
            "source":      {"type": "string",          "description": "Object noun to remove, e.g. 'cup'."},
            "target":      {"type": ["string", "null"],"description": "Description of the new object (mode A).", "default": None},
            "ref_image":   {"type": ["string", "null"],"description": "Path to a reference image of the new object (mode B).", "default": None},
            "instruction": {"type": ["string", "null"],"description": "Full override prompt for Gemini.", "default": None},
            "model":       {"type": "string",          "description": "Gemini model id — leave as the default unless told otherwise.",
                            "default": "gemini-2.5-flash-image", "enum": ["gemini-2.5-flash-image"]},
        },
        "outputs": {
            "ref0_path": {"type": "string", "description": "Absolute path to the edited frame-0 reference."},
        },
    },

    # ------------------------------------------------------------------
    # T7 — VideoPainter generation
    # ------------------------------------------------------------------
    "videopainter_generate": {
        "description": (
            "Run CogVideoX-5B-I2V (VideoPainter) multi-chunk inpainting. "
            "Loads the pipeline once and loops over all segments. "
            "Returns immediately if resume=true and gen_dir already has frames."
        ),
        "inputs": {
            "frames_dir":     {"type": "string",  "description": "Source frames directory."},
            "mask_dir":       {"type": "string",  "description": "Per-frame edit-region masks."},
            "anchor_map":     {"type": "object",  "description": "Dict str(start)->anchor_path from roma_anchors."},
            "gen_dir":        {"type": "string",  "description": "Output directory; frames go into gen_dir/frames/."},
            "segment_starts": {"type": "array", "items": {"type": "integer"},
                                "description": "List of 0-indexed segment start frames."},
            "prompt":         {"type": "string",  "description": "Global generation prompt for the edited video."},
            "dilate":         {"type": "integer", "description": "Mask dilation applied inside the generator.", "default": 12},
            "steps":          {"type": "integer", "description": "Diffusion inference steps.", "default": 10},
            "guidance":       {"type": "number",  "description": "Classifier-free guidance scale.", "default": 6.0},
            "seed":           {"type": "integer", "description": "RNG seed for reproducibility.", "default": 42},
            "model_path":     {"type": "string",  "description": "Path to CogVideoX-5b-I2V checkpoint.",
                                "default": "/workspace/editAnything/ckpt/CogVideoX-5b-I2V"},
            "branch":         {"type": "string",  "description": "Path to VideoPainter branch checkpoint.",
                                "default": "/workspace/editAnything/ckpt/VideoPainter/checkpoints/branch"},
            "id_lora":        {"type": "string",  "description": "Path to VideoPainterID LoRA checkpoint.",
                                "default": "/workspace/editAnything/ckpt/VideoPainterID/checkpoints"},
            "resume":         {"type": "boolean", "description": "Skip generation if gen_dir already has frames.", "default": False},
        },
        "outputs": {
            "gen_frames_dir":    {"type": "string",  "description": "Absolute path to generated frames directory."},
            "n_frames_generated":{"type": "integer", "description": "Number of frames written."},
        },
    },

    # ------------------------------------------------------------------
    # T7b — MVInpainter generation (large camera motion)
    # ------------------------------------------------------------------
    "mvinpainter_generate": {
        "description": (
            "MVInpainter as generator, per-chunk reanchored (same shape as "
            "videopainter_generate, but a multi-view IMAGE model fills each "
            "chunk). Use with mvinpainter_anchors when camera_motion=true — "
            "stays consistent at large viewpoint changes where videopainter's "
            "fixed-background assumption breaks. Subprocess in its own env."
        ),
        "inputs": {
            "frames_dir":     {"type": "string",  "description": "Source frames directory."},
            "mask_dir":       {"type": "string",  "description": "Edit-region mask (same one videopainter_generate would use)."},
            "anchor_map":     {"type": "object",  "description": "Dict str(start)->anchor_path from mvinpainter_anchors."},
            "gen_dir":        {"type": "string",  "description": "Output directory; frames go into gen_dir/frames/."},
            "segment_starts": {"type": "array", "items": {"type": "integer"},
                                "description": "0-indexed segment start frames."},
            "prompt":         {"type": "string",  "description": "Global generation prompt for the edited video.", "default": ""},
            "chunk":          {"type": "integer", "description": "Consecutive frames per reanchor chunk (<= mvinpainter_anchors' n_views).", "default": 20},
            "steps":          {"type": "integer", "description": "Diffusion inference steps per chunk (default 50); lower = faster, some quality cost.", "default": 10},
            "resume":         {"type": "boolean", "description": "Skip generation if gen_dir already has frames.", "default": False},
        },
        "outputs": {
            "gen_frames_dir":     {"type": "string",  "description": "Absolute path to generated frames directory."},
            "n_frames_generated": {"type": "integer", "description": "Number of frames written."},
        },
    },
 


    # ------------------------------------------------------------------
    # T8 — ROSE removal (clean plate)
    # ------------------------------------------------------------------
    "rose_removal": {
        "description": (
            "Run ROSE video object removal to produce a clean plate: source object "
            "AND its shadow/reflection removed. Runs in the 'rose' conda env as a "
            "subprocess. Returns immediately if resume=true and out_dir already has frames."
        ),
        "inputs": {
            "frames_dir":    {"type": "string",          "description": "Source frames directory."},
            "mask_dir":      {"type": "string",          "description": "SAM3 source-object masks (white = remove)."},
            "out_dir":       {"type": "string",          "description": "Output root; clean plate written to out_dir/frames/."},
            "video_length":  {"type": ["integer", "null"],"description": "16n+1 clip length; auto if null.", "default": None},
            "prompt":        {"type": "string",          "description": "Optional text prompt for ROSE.", "default": ""},
            "resume":        {"type": "boolean",         "description": "Reuse existing clean plate if present.", "default": False},
        },
        "outputs": {
            "clean_plate_dir": {"type": "string",  "description": "Absolute path to the clean-plate frames directory."},
            "n_frames":        {"type": "integer", "description": "Number of clean-plate frames produced."},
        },
    },

    # ------------------------------------------------------------------
    # T9 — composite
    # ------------------------------------------------------------------
    "composite": {
        "description": (
            "Feather-composite the generated object onto a fixed background plate "
            "to kill chunk-boundary background drift. "
            "plate=original frames (exp3) or ROSE clean plate (exp1/2)."
        ),
        "inputs": {
            "bg_frames_dir":  {"type": "string",          "description": "Fixed background plate frames (original or ROSE)."},
            "gen_frames_dir": {"type": "string",          "description": "Generated frames containing the new object."},
            "mask_dir":       {"type": "string",          "description": "Per-frame edit-region masks."},
            "out_dir":        {"type": "string",          "description": "Output composite frames directory."},
            "total":          {"type": ["integer", "null"],"description": "Number of frames to composite; auto-counted if null.", "default": None},
        },
        "outputs": {
            "composite_dir": {"type": "string",  "description": "Absolute path to composite frames directory."},
            "n_frames":      {"type": "integer", "description": "Number of frames composited."},
        },
    },

    # ------------------------------------------------------------------
    # T10 — encode
    # ------------------------------------------------------------------
    "encode": {
        "description": (
            "Encode frames to a final .mp4, optionally RIFE-de-spiking the "
            "per-chunk anchor boundary frames first. Scales to the original "
            "portrait resolution."
        ),
        "inputs": {
            "frames_dir":    {"type": "string",          "description": "Input frames directory."},
            "out_path":      {"type": "string",          "description": "Output .mp4 path."},
            "source_frames_dir": {"type": "string", "description": "Original input frames directory, used to recover native resolution for encoding.", "default": None},
            "segment_starts": {"type": "array", "items": {"type": "integer"},
                                "description": "List of 0-indexed segment start frames."},
            "out_size":      {"type": ["array", "null"], "items": {"type": "integer"},
                               "description": "[W, H] output resolution; null = native.", "default": None},
            "fps":           {"type": "integer",         "description": "Output frame rate.", "default": 25},
            "interpolate":   {"type": "boolean",         "description": "RIFE-despike anchor boundary frames.", "default": False},
            "despike_dir":   {"type": ["string", "null"],"description": "Where to write de-spiked frames (required if interpolate=true).", "default": None},
        },
        "outputs": {
            "video_path":       {"type": "string", "description": "Absolute path to the encoded .mp4."},
            "duration_seconds": {"type": "number", "description": "Video duration in seconds."},
        },
    },

    # ------------------------------------------------------------------
    # T11 — union masks
    # ------------------------------------------------------------------
    "union_masks": {
        "description": (
            "Per-frame OR of two mask directories (source SAM3 mask ∪ target mask). "
            "Used by the assets/sam3 path when mask_mode=union."
        ),
        "inputs": {
            "source_mask_dir": {"type": "string", "description": "SAM3 source-object mask directory."},
            "target_mask_dir": {"type": "string", "description": "Target-object mask directory (from assets or edit_mask)."},
            "out_dir":         {"type": "string", "description": "Output union-mask directory."},
        },
        "outputs": {
            "mask_dir": {"type": "string",  "description": "Absolute path to the union-mask directory."},
            "n_frames": {"type": "integer", "description": "Number of mask frames written."},
        },
    },

    # ------------------------------------------------------------------
    # T12 — evaluate
    # ------------------------------------------------------------------

    "evaluate": {
        "description": (
            "Run the full 4-dimension benchmark (FiVE-Bench PSNR/SSIM/LPIPS/"
            "structure_distance/CLIP, CoTracker MFS, NIQE, Gemini judge, critical-"
            "failure caps) on one video via subprocess into the `five-bench` env. "
            "Defaults to mode='full' (MFS + NIQE on). Requires source_frames_dir "
            "when enable_mfs=true."
        ),
        "inputs": {
            "case_id":            {"type": "string", "description": "Unique id for this eval run, e.g. the job id."},
            "source_video_path":  {"type": "string", "description": "Original (pre-edit) video path."},
            "source_frames_dir":  {"type": ["string", "null"], "description": "Extracted source frames dir, required if enable_mfs=true.", "default": None},
            "edited_video_path":  {"type": "string", "description": "Final generated video path."},
            "mask_dir":           {"type": "string", "description": "Per-frame edit-region mask dir."},
            "target_prompt":      {"type": "string", "description": "Rich description of the new object/scene."},
            "source_object":      {"type": ["string", "null"], "description": "Source object noun.", "default": None},
            "target_object":      {"type": ["string", "null"], "description": "Target object description.", "default": None},
            "edit_instruction":   {"type": ["string", "null"], "description": "e.g. 'Replace the cup with a banana.'", "default": None},
            "tier":               {"type": ["string", "null"], "description": "easy | medium | hard, if known.", "default": None},
            "out_dir":            {"type": "string", "description": "Scratch dir for manifest, frames_src staging, and result json."},
            "mode":               {"type": "string", "description": "smoke | dev | full.", "default": "full"},
            "enable_mfs":         {"type": "boolean", "description": "CoTracker MFS; needs source_frames_dir and a downloaded co-tracker checkpoint.", "default": True},
            "enable_niqe":        {"type": "boolean", "description": "pyiqa NIQE.", "default": True},
        },
        "outputs": {
            "final_score":    {"type": ["number", "null"], "description": "Weighted 4-dim score with caps applied, 0-1."},
            "dimensions":     {"type": "object", "description": "edit_success, source_preservation, temporal_consistency, rendering_quality."},
            "critical_flags": {"type": "object", "description": "Boolean judge flags."},
            "caps_applied":   {"type": "array",  "description": "Cap strings that fired."},
            "raw_metrics":    {"type": "object", "description": "Unnormalized: psnr_unedit, ssim_unedit, lpips_unedit, mse_unedit, structure_distance, clip_target, clip_target_edit, niqe, mfs."},
            "judge":          {"type": ["object", "null"], "description": "Raw Gemini judge checklist."},
        },
    },

}


def validate_tool_input(tool_name: str, kwargs: dict) -> list[str]:
    """Return a list of validation error strings (empty = OK).

    Checks required inputs are present and non-null (inputs without a 'default'
    key are required). Does NOT do deep type checking — that would duplicate
    Pydantic's job; this is a fast guard for the orchestrator calling nonsense.
    """
    schema = TOOLS.get(tool_name)
    if schema is None:
        return [f"unknown tool: {tool_name!r}"]
    errors = []
    for param, spec in schema["inputs"].items():
        if "default" not in spec and param not in kwargs:
            errors.append(f"missing required input: {param!r}")
        val = kwargs.get(param, spec.get("default"))
        if "default" not in spec and val is None:
            errors.append(f"required input {param!r} must not be null")
    return errors