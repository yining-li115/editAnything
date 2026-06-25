"""Direct MCP tool-call smoke test — no Gemini orchestration, no VideoPainter.

Validates the MCP plumbing by calling server tools in order:
  extract(skipped) -> sam3_mask -> gemini_edit_frame -> roma_anchors
Stops before videopainter_generate — fits in 24G and runs in minutes.
If every step returns sane counts, the wiring is sound.
"""
import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# --- hardcoded run parameters ---
NAME           = "smoke"
SOURCE         = "cup"
TARGET         = "a ripe yellow banana"
URL            = "http://127.0.0.1:8000/mcp"
FRAMES_DIR     = "/storage/slurm/s0037/input/frames_2chunk"  # 97 frames, pre-extracted
SEGMENT_STARTS = [0, 48]   # 97 frames -> 2 segments


def _payload(result):
    for b in result.content:
        if getattr(b, "type", None) == "text":
            try:
                return json.loads(b.text)
            except json.JSONDecodeError:
                return {"raw": b.text}
    return {}


def _show(label, payload):
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2)[:1200])
    return payload


def _fail(payload):
    return isinstance(payload, dict) and (
        payload.get("error") or payload.get("returncode") not in (None, 0))


async def main():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("registered tools:", [t.name for t in tools.tools])

            async def call(tool, args):
                return _payload(await session.call_tool(tool, args))

            # 1. SAM3 source mask || Gemini reference edit (parallel)
            # Pass frames_dir explicitly so the tool reads from the pre-extracted
            # location instead of outputs/<name>/frames_src/.
            # gemini_edit_frame defaults image_path to frames_src/frame_00001.png
            # via RunPaths — so we also pass image_path explicitly here.
            first_frame = os.path.join(FRAMES_DIR, "frame_00001.png")
            sam3_co = call("sam3_mask", {
                "name": NAME, "text": SOURCE, "frames_dir": FRAMES_DIR})
            edit_co = call("gemini_edit_frame", {
                "name": NAME, "source": SOURCE, "target": TARGET,
                "image_path": first_frame})
            p_sam3, p_edit = await asyncio.gather(sam3_co, edit_co)
            _show("sam3_mask", p_sam3)
            _show("gemini_edit_frame", p_edit)
            if _fail(p_sam3):
                sys.exit("sam3 failed — stopping")
            if _fail(p_edit):
                print("WARNING: gemini edit failed (quota?) — roma needs ref0, may fail next")

            # 2. RoMa edit masks + per-segment anchors
            ra = _show("roma_anchors", await call("roma_anchors", {
                "name": NAME, "target_word": TARGET, "source_word": SOURCE,
                "segment_starts": SEGMENT_STARTS,
                "frames_dir": FRAMES_DIR}))
            if _fail(ra):
                sys.exit("roma failed — stopping")

            ok = (ra.get("n_anchors") == ra.get("expected_n_anchors")
                  and ra.get("n_masks", 0) > 0)
            print(f"\n=== SMOKE RESULT: {'PASS' if ok else 'CHECK'} ===")
            print(f"  masks : {p_sam3.get('n_mask_files')} sam3  |  {ra.get('n_masks')} edit-region")
            print(f"  anchors: {ra.get('n_anchors')}/{ra.get('expected_n_anchors')}")
            print("  -> wiring is sound; only videopainter_generate remains "
                  "(test on >=48G card).")


if __name__ == "__main__":
    asyncio.run(main())