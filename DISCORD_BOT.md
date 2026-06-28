## Setup

### 1. Environment

```bash
conda activate editanything
pip install discord.py yt-dlp
```

### 2. `.env` file

```
GEMINI_API_KEY=...
DISCORD_BOT_TOKEN=...
```

## Running the bot

```bash
cd /workspace/editAnything
conda activate editanything
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VP_OFFLOAD=sequential #if on a 48GB VRAM GPU
python discord_bot.py
```

You should see: `[bot] connected as <bot name>`

> **Only one instance of the bot should run at a time.** If two people launch `discord_bot.py` simultaneously with the same token, behavior will be unpredictable.

---

## Usage

Go to the server on discord.
### With a video attachment
Type in the chat: 
```
!replace replace the cup with a ripe yellow banana, use segment_starts [0, 48]
```
+ attach a `.mp4`, `.mov`, `.avi`, or `.mkv` file.

### With a Google Drive or YouTube link
Type in the chat: 
```
!replace replace the cup with a ripe yellow banana, use segment_starts [0, 48] https://drive.google.com/...
```

> Google Drive links must be **publicly accessible**.

---

## Choosing `segment_starts`

Each segment covers 48 frames. Specify `segment_starts` based on your video length:

| Frames | `segment_starts` |
|---|---|
| ~48 | `[0]` |
| ~96 | `[0, 48]` |
| ~144 | `[0, 48, 96]` |
| ~192 | `[0, 48, 96, 144]` |
| ~240 | `[0, 48, 96, 144, 192]` |

---

## What the bot does

1. Downloads the video (attachment or Drive/YouTube link)
2. Extracts frames with ffmpeg → `jobs/<message_id>/frames/`
3. Runs the full pipeline via the Gemini orchestrator:
   - `sam3_mask` — source object mask
   - `gemini_edit` — reference frame generation
   - `roma_edit_mask` — per-frame edit masks
   - `roma_anchors` — per-segment anchors
   - `videopainter_generate` — video generation
   - `composite` — compositing
   - `encode` — final video
4. Sends the output video on Discord (direct if < 25MB, via file.io otherwise)

---

## Approximate durations (L40, VP_OFFLOAD=sequential)

| Video length | Estimated time |
|---|---|
| ~48 frames (2s) | ~20-30 min |
| ~96 frames (4s) | ~35-50 min |
| ~240 frames (10s) | ~90 min |

---

## Known issues

**`VP_OFFLOAD=sequential` is required** — running with `VP_OFFLOAD=none` may cause CUDA out of memory errors since SAM3 and VideoPainter together exceed available VRAM.

**`rose_removal` is disabled** — not configured, the bot explicitly skips it.

**Gemini may choose wrong `segment_starts`** — always specify them explicitly in your prompt until auto-injection is implemented.

###TODO
**Make concurrent jobs are sequential** — if two users send a request simultaneously, the second one will wait for the first to finish before starting.

---

## Output location

```
output/
└── <video_name>.mp4   ← final output video
```