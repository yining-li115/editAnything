# editAnything 

Replace an object in a video with a described one (e.g. **cup → banana**), across
the whole clip, decoupling three independent components:

- **SAM3** — segments/tracks the object to remove → per-frame masks.
- **Gemini** — edits one frame in place → a clean reference of the new object.
- **VideoPainter** — generates the new object into the masked region and keeps it
  consistent across the video (CogVideoX-5B-I2V + branch + VideoPainterID LoRA).

These are **separate stages that exchange files**; VideoPainter never calls SAM3 or
Gemini (unlike the original `submodules/VideoPainter/app/app.py` Gradio demo). FLUX is not used.

## Environment setup (only what we need)

One conda env (`editanything`, Python 3.10, torch 2.4 / cu121) runs everything we
use. We deliberately skip the heavy parts of a full VideoPainter install that our
pipeline does NOT use: **FLUX.1-Fill-dev (~24 GB)**, the SAM2 CUDA extension, and
the Gradio app (so no `OPENAI_API_KEY`).

```bash
# 0. clone WITH submodules (VideoPainter + sam3 are pinned git submodules, gitignored ckpt)
git clone --recurse-submodules https://github.com/yining-li115/editAnything.git && cd editAnything
#  (already cloned without --recurse-submodules? run:  git submodule update --init)

# 1. env + the diffusers fork we actually use (submodules/VideoPainter/ is the checkout)
conda create -n editanything python=3.10 -y && conda activate editanything
cd submodules/VideoPainter
pip install -r requirements.txt
pip install -e ./diffusers           # CogVideoX branch / id_pool pipeline lives here
conda install -c conda-forge ffmpeg -y
cd ../..
#  Do NOT run `cd app && pip install -e .` — that builds the SAM2 ext we don't use.

# 2. SAM3 (the submodules/sam3 submodule) into the SAME env without disturbing torch 2.4
cd submodules/sam3
pip install -e . --no-deps --config-settings editable_mode=compat   # compat REQUIRED (else sam3.__file__ is None)
pip install timm ftfy==6.1.1 regex iopath typing_extensions "setuptools<81" pycocotools
cd ../..
python -c "import sam3; print(sam3.__file__)"   # must print a real path, not None

# 3. HuggingFace + checkpoints into the top-level ckpt/ (gitignored) — NO FLUX
pip install "huggingface_hub==0.24.1"   # keep <1.0 (transformers 4.42.2); use the old huggingface-cli
huggingface-cli login                    # first request access at https://huggingface.co/facebook/sam3
huggingface-cli download TencentARC/VideoPainter --local-dir ckpt          # branch + VideoPainterID -> ckpt/
huggingface-cli download THUDM/CogVideoX-5b-I2V  --local-dir ckpt/CogVideoX-5b-I2V
#  We do NOT download black-forest-labs/FLUX.1-Fill-dev (flux_inp) — not used (~24 GB saved).
#  SAM3 weights auto-download on first build (gated facebook/sam3).

# 4. Gemini edit stage (optional)
pip install google-genai
cp .env.example .env && $EDITOR .env     # set GEMINI_API_KEY

# 5. RoMa anchor/mask propagation (--backend roma) — keep torch 2.4
pip install romatch --no-deps && pip install loguru einops   # do NOT let it pull torch>=2.5

# 6. RIFE smoothing (--interpolate) — prebuilt rife-ncnn-vulkan binary (Vulkan/GPU,
#    ships its own models). Needs a Vulkan loader (libvulkan.so.1 — provided by the
#    NVIDIA driver; check with `ldconfig -p | grep libvulkan`). Download + extract
#    under the repo parent's tools/ (commands run from editAnything/):
mkdir -p ../tools && cd ../tools
wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
unzip -q rife-ncnn-vulkan-20221029-ubuntu.zip
chmod +x rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan
cd ../editAnything
export RIFE_BIN="$(cd .. && pwd)/tools/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan"
#   sanity-check it runs (rife-v4.6 model ships inside the zip, next to the binary;
#   matches encode.py's RIFE_MODEL default):
#     "$RIFE_BIN" -0 a.png -1 b.png -o mid.png -m "$(dirname "$RIFE_BIN")/rife-v4.6"
#   NOTE: set RIFE_BIN explicitly as above — encode.py's built-in fallback path is a
#   stale absolute path (/root/project/tools/...) and won't match a fresh checkout.
```

Checkpoint layout we rely on (top-level `ckpt/`, gitignored; paths come from
`contracts/layout.py`'s model registry):

```
ckpt/
├── VideoPainter/checkpoints/branch     # CogvideoXBranchModel
├── VideoPainterID/checkpoints          # VideoPainterID LoRA
└── CogVideoX-5b-I2V                     # base I2V DiT
```

Gotchas (learned the hard way):

- Keep `huggingface_hub < 1.0` and `ftfy==6.1.1`, `setuptools<81` (sam3 uses
  `pkg_resources`). If `decord` import fails (non-x86_64): `pip install eva-decord`.
- Activate the `editanything` conda env before running (it holds the torch 2.4 /
  sam3 / diffusers-fork installs).

Full original setup notes (incl. the parts we dropped) live in `../SERVER_SETUP.md`.

## What we actually use from the `submodules/VideoPainter` submodule

`submodules/VideoPainter` is a pinned submodule of upstream `TencentARC/VideoPainter`
(no fork, no patches). We only use:

- `submodules/VideoPainter/diffusers/` — the custom fork providing
  `CogVideoXI2VDualInpaintAnyLPipeline`, `CogvideoXBranchModel`, and the id_pool
  `CogVideoXTransformer3DModel`.
- the checkpoints in top-level `ckpt/{CogVideoX-5b-I2V, VideoPainter/checkpoints/branch, VideoPainterID/checkpoints}`.

Not used: `app/` (Gradio), `app/sam2*`, `utils.py`'s FLUX path, `flux_inp`,
`train/`, `evaluate/`, `infer/`, `data_utils/`.

## Code layout

Capability logic, MCP layer, and agents are separated so each stage is an
isolated, independently-wrappable unit (dependency direction:
`contracts ← components ← (mcp) ← (agents)`).

| Path                       | Role                                                          |
| -------------------------- | ------------------------------------------------------------ |
| `components/extract.py`    | video → frames + frame-set queries                           |
| `components/sam3_mask.py`  | SAM3 text prompt → per-frame (or single-image) mask          |
| `components/gemini_edit.py`| Gemini API → in-place frame edit (the new-object reference)  |
| `components/roma_warp.py`  | low-level RoMa match + warp primitives (shared)              |
| `components/edit_mask.py`  | per-frame edit region — **generic, cross-candidate** (roma/assets) |
| `components/anchor.py`     | per-segment clean anchors — **VideoPainter-specific** (roma/assets) |
| `components/videopainter.py`| VideoPainter multi-chunk reanchor generation (**models loaded once**) |
| `components/composite.py`  | feather the object onto a fixed plate (kills chunk-boundary jumps) |
| `components/encode.py`     | frames → portrait mp4 (+ optional RIFE de-spike)             |
| `contracts/layout.py`      | run-dir layout + config-driven model registry (ckpt paths)  |
| `pipeline.py`              | end-to-end local runner (wires the components)              |

Outputs land in `outputs/<name>/`.

## Run

All tunable parameters live in **`config.yaml`** — edit it, then:

```bash
conda activate editanything
cd editAnything
HF_HOME=/workspace/.hf_home python pipeline.py --config config.yaml
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
python -m components.gemini_edit --image frame_00001.png --out ref.png --source cup --target "a ripe yellow banana"
# B) supply an image of the exact object you want
python -m components.gemini_edit --image frame_00001.png --out ref.png --source cup --ref_image my_banana.png
```

Needs `pip install google-genai` and `GEMINI_API_KEY` (see `.env.example`).

## Any-video path (RoMa backend)

`--backend roma` builds, from scratch, both the per-frame edit masks
(`components/edit_mask.py`) and the per-segment anchors (`components/anchor.py`),
sharing RoMa warp primitives (`components/roma_warp.py`):

- SAM3 segments the new object on `ref0` and the old object on frame 0 →
  frame-0 edit region; RoMa dense-warps it to every frame (follows motion).
- the whole `ref0` is warped to each segment start → the per-segment anchor.
  Needs only `--ref0` (a frame-0 edit from `gemini_edit`). `--backend assets`
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

**Goal:** turn today's fixed script into an **agentic** system — a Gemini
orchestrator that calls each capability on demand as an MCP tool, runs candidates
in parallel, and self-corrects from an LLM judge's feedback instead of us
hand-tuning params. Target architecture (`system_pipeline.png`):

![Target agentic architecture](system_pipeline.png)

What this requires, in build order:

1. **Wrap each stage as an MCP tool** — SAM3 mask, Gemini edit/Imagen asset, RoMa
   propagate, ROSE removal, VideoPainter generate, composite, encode. Today they
   are functions chained in `pipeline.py`; the orchestrator needs to call them
   individually.
2. **Gemini orchestrator agent** — parses the chat request ("cyberpunk banana" +
   video) into intent (source object, target object, style), then fans out:
   SAM tracker and Imagen/Gemini-edit run **in parallel**, and generation starts
   once both the mask and the asset are ready.
3. **Parallel model candidates** — run VideoPainter, a ROSE-removal+inpaint path,
   and future models on the same input concurrently, so the judge can pick the
   best result rather than us committing to one backend up front.
4. **LLM judge (Gemini)** — scores each candidate on quality, temporal
   consistency, and style match against the request; the best score above a
   threshold is returned, with the score and model surfaced to the user.
5. **Self-correcting retry loop** — on a below-threshold score, the orchestrator
   retries (2–3×) by **adjusting params and regenerating** — this is where the
   **tuning agent** lives, picking `dilate`, segment split, mask shape, etc. from
   the judge's feedback instead of us tuning them by hand.
6. **ROSE removal as a component** — a clean-plate removal stage that fixes the
   leftover source-object shadow (see Known limitations) and re-enables the
   original composite path.

The current `pipeline.py` is the single-backend, no-judge slice of this: it runs
the SAM3 → edit → VideoPainter chain end to end with params fixed in
`config.yaml`. The agentic version keeps these same stages but lets the
orchestrator choose, parallelize, score, and retry them.

## TODO

What's done vs. still open.

**Done**

- [x] SAM3 text-prompt source mask (`components/sam3_mask.py`)
- [x] Gemini frame-0 edit / reference (`components/gemini_edit.py`)
- [x] RoMa edit-mask + anchor propagation, any-length video (`components/roma_warp.py`, `edit_mask.py`, `anchor.py`)
- [x] VideoPainter inpainting-only generation, multi-chunk reanchor, models loaded once (`components/videopainter.py`)
- [x] Composite + portrait encode + RIFE anchor de-spike (`components/composite.py`, `encode.py`)
- [x] End-to-end fixed-param pipeline driven by `config.yaml` (`pipeline.py`)
- [x] Decoupled components + `contracts/` registry; VideoPainter & sam3 as git submodules

**Open — quality**

- [ ] **Source-object shadow removal** — shadow sits outside the SAM3 mask, so it stays. Add ROSE clean-plate removal, or grow the edit region to repaint it.
- [ ] **Irregular-hull edit region** — replace the current (target∪source) **bbox** with an alpha-based irregular hull (RoMa-warp → dilate) that hugs the object and covers the shadow.

**Open — agentic system** (see Roadmap)

- [ ] Wrap each stage as an **MCP tool** (SAM3, Gemini edit/Imagen, RoMa, ROSE, VideoPainter, composite, encode)
- [ ] **Gemini orchestrator agent** — parse chat intent, fan out SAM + asset generation in parallel
- [ ] **Parallel model candidates** — run VideoPainter / ROSE+inpaint / future models concurrently
- [ ] **LLM judge (Gemini)** — score quality / temporal consistency / style match, pick best above threshold
- [ ] **Self-correcting retry loop / tuning agent** — auto-pick `dilate`, segment split, mask shape from judge feedback (2–3× retries)
- [ ] **ROSE removal component** — clean plate, re-enables the original composite path
- [ ] **Web demo / API** — return video + score + which model was used