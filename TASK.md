# Task: Replace SAM2 with SAM3 in VideoPainter

## Background

This project is based on the VideoPainter repo (TencentARC/VideoPainter).
We already use SAM3 in our experiments. The goal is to replace the click-based SAM2 in VideoPainter's `app/app.py` with SAM3 text-prompt-based segmentation.

The core VideoPainter + FLUX logic in `app/utils.py` should NOT be changed.

---

## ✅ Task 1 (Do Now): Replace SAM2 with SAM3 in `app/app.py`

### What to REMOVE

1. **SAM2 imports and initialization** (top of file):
```python
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "../ckpt/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
```

2. **`sam_refine()` function** — handles user click events, calls `predictor.add_new_points()`

3. **`vos_tracking_video()` function** — propagates SAM2 mask across frames via `predictor.propagate_in_video()`

4. **`clear_click()` function** — resets click state

5. **`get_prompt()` function** — parses click coordinates into SAM2 prompt format

### What to ADD

A new function `auto_track_with_sam3()` that:
- Takes `video_state` (dict with `origin_images`: list of numpy frames) and `text_prompt` (str, e.g. `"cup"`)
- Runs SAM3 with the text prompt to generate per-frame masks automatically
- Applies binary dilation (same as original `vos_tracking_video`):
```python
import scipy.ndimage
mask = scipy.ndimage.binary_dilation(mask, iterations=6)
```
- Writes masks into `video_state["masks"]` in the required format
- Returns updated `video_state`

**Required mask format:**
```python
# numpy array, shape: (num_frames, height, width, 1)
# dtype: float or uint8
# values: 0 (background) or 1 (object to replace)
# height/width must match origin_images dimensions
```

### New flow

```
video + object_prompt ("cup") + replace_prompt ("a cartoon banana")
    ↓
Load video frames → video_state["origin_images"]
    ↓
auto_track_with_sam3(video_state, object_prompt)  ← replaces sam_refine + vos_tracking_video
    ↓
inpaint_video(...)  ← unchanged
    ↓
Output video
```

---

## ⏳ Task 2 (Pending): MCP Packaging

**To be decided with the team** — whether to package as:

- Option A: One single MCP (SAM3 + FLUX + VideoPainter together)
- Option B: Two separate MCPs (SAM3 | VideoPainter+FLUX)
- Option C: Three separate MCPs (SAM3 | FLUX | VideoPainter)

Do not implement this until the team agrees on the structure.

---

## Key constraints

- Do NOT modify `utils.py` or `generate_frames()`
- Do NOT modify the FLUX inpainting logic
- `video_state["masks"]` must be numpy array of shape `(N, H, W, 1)` with values 0/1
- SAM3 code is in the `sam3/` directory of this repo
