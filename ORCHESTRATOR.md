# MCP server + Gemini orchestrator

Two pieces, layered on top of the existing `components/` pipeline stages:

- **`mcp_server.py`** — wraps each pipeline stage (`extract_frames`, `sam3_mask`,
  `roma_edit_mask`, `roma_anchors`, `gemini_edit`, `videopainter_generate`,
  `rose_removal`, `composite`, `encode`, `union_masks`) as an MCP tool via
  `fastmcp`. GPU models (SAM3, VideoPainter) are cached as module-level
  singletons so repeated calls in one process don't reload weights.
- **`orchestrator.py`** — a Gemini function-calling agent that drives those
  tools from a natural-language request (e.g. "replace the cup with a
  banana"). It imports the tool functions directly (no MCP transport) and
  builds Gemini's tool schemas straight from `contracts/tools.py`, so there's
  one schema to keep in sync, not two.

This is the "Gemini orchestrator agent" from the README roadmap — `mcp_server.py`
wraps the stages, `orchestrator.py` is the agent that calls them on demand
instead of running the fixed chain in `pipeline.py`.

## Setup

Same `.env` as `components/gemini_edit.py`:

```bash
cp .env.example .env   # set GEMINI_API_KEY (GOOGLE_API_KEY also accepted)
```

`fastmcp` is required for `mcp_server.py` (already in the `videopainter` conda
env on this cluster). `google-genai` is required for both `gemini_edit.py` and
`orchestrator.py`.

## Running the orchestrator

```bash
conda activate videopainter
python orchestrator.py "replace the cup with a ripe yellow banana in the \
frames at /path/to/frames, source word 'cup'"
```

It prints every tool call Gemini makes and the result, then a final summary.
Add `--model` to override the default (`gemini-2.5-pro`) and `--max-turns` to
raise the 20-turn cap on long chains.

**Be explicit about `segment_starts` in the prompt.** `videopainter_generate`
writes frames at their real global index (`frame_{start+i+1:05d}.png`), not a
contiguous 1..N range. With `CLIP=49`, contiguous coverage needs segments 48
apart (1-frame overlap), e.g. `[0,48,96,144]` for the first ~193 frames. Left
unconstrained, Gemini may instead space segments evenly across the whole
video (e.g. `[0,217,434,651]` for an 867-frame clip), which leaves large
unpainted gaps and breaks `composite`'s assumption of a contiguous frame
range. If you only want a short test clip, say the exact `segment_starts` and
`total` in your prompt rather than letting Gemini infer them from "N chunks".

## Running mcp_server.py directly (manual testing)

```bash
npx @modelcontextprotocol/inspector -- python mcp_server.py
```

Opens a browser UI to call each tool individually and inspect raw
inputs/outputs. Useful for debugging a single stage without the agent loop.
Bump the Inspector's Configuration → Request Timeout above the 10s default —
GPU-heavy tools (`sam3_mask`, `videopainter_generate`) take much longer than
that on first call (model load).

## Automatic defaults (no longer need to specify in prompt)
- **`segment_starts`** — computed automatically from `n_frames` by Gemini 
  (see system prompt); no need to specify in your prompt.
- **Output resolution** — always matches the original input frames; no need 
  to specify `out_size`.
- **RIFE de-spike** — always enabled at segment boundaries; no need to 
  specify `interpolate`.
  
## Known limitations

- **`rose_removal` needs its own `rose` conda env**, not yet set up on this
  cluster (see README.md's ROSE section). Skip it in orchestrator prompts —
  ask for the original frames to be used directly as the `composite`
  background plate instead.
- **One call at a time.** `mcp_server.py` runs synchronously in a single
  process — a long-running GPU call (model load, generation) blocks all other
  requests, including unrelated ones like `tools/list` in the Inspector.
- **GPU placement matters.** Run `mcp_server.py` / `orchestrator.py` on a node
  with a real GPU (e.g. `node21`), not a login node — `sam3_mask` and
  `videopainter_generate` will otherwise try to load onto whatever GPU is
  local to wherever the process runs.
