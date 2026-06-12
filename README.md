# editAnything 
Replace an object in a video with a described one (e.g. **cup → banana**), across
the whole clip, decoupling three independent components:

- **SAM3** — segments/tracks the object to remove → per-frame masks.
- **Gemini** — edits one frame in place → a clean reference of the new object.
- **VideoPainter** — generates the new object into the masked region and keeps it
  consistent across the video (CogVideoX-5B-I2V + branch + VideoPainterID LoRA).

These are **separate stages that exchange files**; VideoPainter never calls SAM3 or
Gemini (unlike the original `VideoPainter/app/app.py` Gradio demo). FLUX is not used.

## Environment setup (only what we need)

One conda env (`videopainter`, Python 3.10, torch 2.4 / cu121) runs everything we
use. We deliberately skip the heavy parts of a full VideoPainter install that our
pipeline does NOT use: **FLUX.1-Fill-dev (~24 GB)**, the SAM2 CUDA extension, and
the Gradio app (so no `OPENAI_API_KEY`).

```bash
# 0. clone (sam3 is a sibling clone, gitignored)
git clone https://github.com/yining-li115/editAnything.git && cd editAnything
git clone https://github.com/facebookresearch/sam3.git

# 1. env + the diffusers fork we actually use
conda create -n videopainter python=3.10 -y && conda activate videopainter
cd VideoPainter
pip install -r requirements.txt
pip install -e ./diffusers           # CogVideoX branch / id_pool pipeline lives here
conda install -c conda-forge ffmpeg -y
cd ..
#  Do NOT run `cd app && pip install -e .` — that builds the SAM2 ext we don't use.

# 2. SAM3 into the SAME env without disturbing torch 2.4
cd sam3
pip install -e . --no-deps --config-settings editable_mode=compat   # compat REQUIRED (else sam3.__file__ is None)
pip install timm ftfy==6.1.1 regex iopath typing_extensions "setuptools<81" pycocotools
cd ..
python -c "import sam3; print(sam3.__file__)"   # must print a real path, not None

# 3. HuggingFace + checkpoints — NO FLUX
pip install "huggingface_hub==0.24.1"   # keep <1.0 (transformers 4.42.2); use the old huggingface-cli
huggingface-cli login                    # first request access at https://huggingface.co/facebook/sam3
cd VideoPainter
huggingface-cli download TencentARC/VideoPainter --local-dir ckpt          # branch + VideoPainterID
huggingface-cli download THUDM/CogVideoX-5b-I2V  --local-dir ckpt/CogVideoX-5b-I2V
cd ..
#  We do NOT download black-forest-labs/FLUX.1-Fill-dev (flux_inp) — not used (~24 GB saved).
#  SAM3 weights auto-download on first build (gated facebook/sam3).

# 4. Gemini edit stage (optional)
pip install google-genai
cp .env.example .env && $EDITOR .env     # set GEMINI_API_KEY

# 5. RoMa anchor/mask propagation (--backend roma) — keep torch 2.4
pip install romatch --no-deps && pip install loguru einops   # do NOT let it pull torch>=2.5

# 6. RIFE smoothing (--interpolate) — prebuilt binary (Vulkan), incl. models
#    download from https://github.com/nihui/rife-ncnn-vulkan/releases, then:
export RIFE_BIN=/path/to/rife-ncnn-vulkan-*/rife-ncnn-vulkan   # default points under ../tools/
```

Checkpoint layout we rely on:
```
VideoPainter/ckpt/
├── VideoPainter/checkpoints/branch     # CogvideoXBranchModel
├── VideoPainterID/checkpoints          # VideoPainterID LoRA
└── CogVideoX-5b-I2V                     # base I2V DiT
```

Gotchas (learned the hard way):
- **diffusers `outputs.py` missing** → `ModuleNotFoundError: diffusers.utils.outputs`.
  Caused by VideoPainter/.gitignore's broad `output*`/`test*` rules (already fixed
  here to `/output*` / `/test*`). On a fresh checkout that still lacks the file,
  restore `src/diffusers/utils/outputs.py` from diffusers v0.31.0.
- Keep `huggingface_hub < 1.0` and `ftfy==6.1.1`, `setuptools<81` (sam3 uses
  `pkg_resources`). If `decord` import fails (non-x86_64): `pip install eva-decord`.
- Env = `/venv/videopainter` on this server (`/venv/videopainter/bin/python`).

Full original setup notes (incl. the parts we dropped) live in `../SERVER_SETUP.md`.

## What we actually use from `VideoPainter/`

Only two things — treat the rest of that vendored repo as untouched upstream:

- `VideoPainter/diffusers/` — the custom fork providing
  `CogVideoXI2VDualInpaintAnyLPipeline`, `CogvideoXBranchModel`, and the id_pool
  `CogVideoXTransformer3DModel`.
- `VideoPainter/ckpt/{CogVideoX-5b-I2V, VideoPainter/checkpoints/branch, VideoPainterID/checkpoints}`.

Not used: `app/` (Gradio), `app/sam2*`, `utils.py`'s FLUX path, `ckpt/flux_inp`,
`train/`, `evaluate/`, `infer/`, `data_utils/`.

## Modules (in `src/`)

| File | Stage |
|---|---|
| `sam3_track.py` | SAM3 text prompt → per-frame source mask |
| `gemini_edit.py` | Gemini API → in-place frame edit (the new-object reference) |
| `anchors.py` | per-segment anchors + target masks (pluggable; see gap below) |
| `generate.py` | VideoPainter multi-chunk reanchor generation (**models loaded once**) |
| `composite.py` | feather the object onto a fixed plate (kills chunk-boundary jumps) |
| `encode.py` | frames → portrait mp4 (+ optional interpolation) |
| `pipeline.py` | end-to-end orchestrator + CLI |

Outputs land in `outputs/<name>/`.

## Run (phase A — reproduces cup2 with prepared assets)

```bash
conda activate videopainter   # /venv/videopainter
cd editAnything
python src/pipeline.py \
  --frames_dir ../cup2_low_extract/cup2_low \
  --name cup2_run \
  --source cup --target "a ripe yellow banana" \
  --backend assets --assets ../exp3_bundle/inputs \
  --mask_mode target \
  --out_size 480x832 --fps 25 --interpolate
```

Single chunk (quick check): add `--segment_starts 0 --stop_after generate`.

**Length handling (automatic):** the base model does 49 frames per pass. If the
video is ≤49 frames the pipeline runs a single pass (`mode=single-pass`); only when
it exceeds 49 does multi-chunk reanchor kick in (`mode=multi-chunk (k segments)`),
with segment starts auto-derived from the frame count.

**Gemini reference (the edit stage), two modes:**
```bash
# A) describe the new object
python src/gemini_edit.py --image frame_00001.png --out ref.png --source cup --target "a ripe yellow banana"
# B) supply an image of the exact object you want
python src/gemini_edit.py --image frame_00001.png --out ref.png --source cup --ref_image my_banana.png
```
Needs `pip install google-genai` and `GEMINI_API_KEY` (see `.env.example`).

## Any-video path (RoMa backend)

The **reanchor** quality needs, per segment, a clean anchor at that viewpoint plus
a per-frame mask that follows the object. Two backends produce these
(`--backend`):

- `assets` — load pre-made anchors + masks (reproduces cup2 from the bundle).
- `roma` — **general**: warp a frame-0 reference (the Gemini edit) to every
  viewpoint with RoMa (`roma_propagate.py`). Needs only `--ref0` (frame-0 with the
  new object placed); it RoMa-matches frame0↔frame_k, warps the object RGB+mask,
  gates by match certainty → per-frame masks + per-segment anchors. The frame-0
  object mask is auto-segmented with SAM3 (`--target_word`) unless `--ref0_mask`
  is given.

```bash
# any video, from scratch: frame-0 reference -> RoMa -> multi-chunk -> RIFE
python src/pipeline.py --video my.mp4 --name my_run \
  --source cup --target "a ripe yellow banana" --target_word banana \
  --backend roma --ref0 ref0_banana.png \
  --mask_mode union --interpolate
```
(`ref0_banana.png` = a frame-0 edit from `gemini_edit.py`.)

## Smoothing (RIFE)

`--interpolate` writes a 2×-fps `final_interp.mp4`. Backend is **RIFE**
(rife-ncnn-vulkan, GPU/Vulkan) on the frames; it softens the 1-frame anchor "pop"
intrinsic to reanchor segment boundaries. Falls back to ffmpeg `minterpolate` if
the RIFE binary (`RIFE_BIN`) isn't found.

## Pipeline / params

Per-segment reanchor (validated exp3): 720×480 work res, 49-frame clips, starts
`0,48,96,144,192,240,251`, `id_pool_resample_learnable=True`, DPM trailing,
sequential CPU offload + VAE tiling, `steps=50 guidance=6.0 dilate=12 seed=42`.
Final video is unsquished back to portrait. ~11 min/segment, ~22 GB peak VRAM.
