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

## Run

All tunable parameters live in **`config.yaml`** — edit it, then:

```bash
conda activate videopainter   # /venv/videopainter
cd editAnything
HF_HOME=/workspace/.hf_home python src/pipeline.py --config config.yaml
```

`config.yaml` holds the source/target objects, prompt, backend, paths, segment/
sampling/output settings (see comments in the file). Any value can be overridden
on the CLI, e.g. `--name myrun --seed 7`. `HF_HOME` is needed so SAM3 (gated
`facebook/sam3`) can load.

The general (any-video) path uses `backend: roma` with a frame-0 reference
(`ref0`, a Gemini edit). RoMa propagates it to produce, from scratch, both the
per-frame edit masks (target∪source bbox) and the per-segment anchors — no
prepared assets needed. `backend: assets` instead loads pre-made anchors+masks.

Quick checks: `--stop_after extract|mask|generate` ; one chunk: `--segment_starts 0`.

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

`--backend roma` builds, from scratch, both the per-frame edit masks and the
per-segment anchors from one RoMa pass (`roma_propagate.py`):
- SAM3 segments the new object on `ref0` and the old object on frame 0 →
  frame-0 edit region; RoMa dense-warps it to every frame (follows motion).
- the whole `ref0` is warped to each segment start → the per-segment anchor.
Needs only `--ref0` (a frame-0 edit from `gemini_edit.py`). `--backend assets`
instead loads prepared anchors + masks.

## Smoothing (RIFE anchor de-spike)

`--interpolate` replaces each segment-boundary **anchor frame** n with
`RIFE(frame n-1, frame n+1)` — the warped anchor is a one-frame "pop", so we swap
it for the motion-midpoint of its neighbours. Native fps unchanged (NOT 2×).
RIFE = rife-ncnn-vulkan (`RIFE_BIN`).

## Pipeline / params

Per-segment reanchor: 720×480 work res, 49-frame clips, starts auto
(`0,48,96,…`, only past 49 frames), `id_pool_resample_learnable=True`, DPM
trailing, sequential CPU offload + VAE tiling, `steps=50 guidance=6.0 dilate=12
seed=42`. Final unsquished to portrait. ~11 min/segment, ~22 GB peak VRAM.
RoMa backend skips composite (warped-anchor + composite ghosts the hand).

## Known limitations

- **Source-object shadow can remain.** The old object's cast shadow lives outside
  the SAM3 mask, so it is neither in the edit region nor removed. The original
  pipeline killed it by compositing onto a **ROSE cup-removed plate** (a separate
  removal stage we don't have). Fixes: (a) grow the edit region (irregular hull +
  dilate) to cover the shadow so generation repaints it, or (b) add a ROSE removal
  stage. See Roadmap.
- Edit region is currently a **bbox** of (target∪source); an **irregular hull**
  (banana_no_bg-style alpha → RoMa-warp → dilate) hugs the object better and
  covers the shadow — to be switched in.

## Roadmap (agentic)

Goal: make this an **agentic** system — an orchestrator that calls each capability
on demand instead of one fixed script.
- Wrap the stages (SAM3 mask, Gemini edit, RoMa propagate, ROSE removal,
  VideoPainter generate, composite, encode) as **MCP tools**.
- A **tuning agent** that picks parameters (dilate, segment split, mask shape, …)
  from feedback on intermediate results.
- Add **ROSE removal** as a component (clean plate → fixes shadow, enables the
  original composite path).
- Orchestrator routes: choose backend, decide when to remove / re-anchor / smooth.
