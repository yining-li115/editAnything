"""Gemini orchestrator: drives the pipeline tools via Gemini function-calling.
 
videopainter_generate runs on a remote H100 (videopainter_mcp_server.py);
every other tool runs in-process as before. run() stays SYNCHRONOUS on
purpose — discord_bot.py calls it via
`asyncio.get_event_loop().run_in_executor(None, run, prompt)`, which requires
a plain blocking callable, not a coroutine. Only the one remote call opens
its own short-lived event loop internally (asyncio.run(), scoped to that
call) — safe here because run() itself executes in a worker thread (via
run_in_executor), never on discord.py's own event-loop thread.
 
Setup:
    cp .env.example .env   # set GEMINI_API_KEY (GOOGLE_API_KEY also accepted)
    export H100_HOST=user@h100-ip
    export VP_SHARED_FS=0   # 0 = no shared storage (rsync), 1 = SLURM shared mount
 
Usage (unchanged from before, and unchanged for discord_bot.py's import):
    python orchestrator.py "replace the cup with a ripe yellow banana, frames in \
/storage/slurm/sisi/AFM-dataset/cup/frames/cup2, source word 'cup'"
"""
from __future__ import annotations
import os
import sys
import time
import asyncio
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from contracts.tools import TOOLS  # noqa: E402
import mcp_server as tools_mod      # noqa: E402
from components.gemini_edit import load_dotenv  # noqa: E402
#from remote_videopainter import videopainter_generate_remote  # noqa: E402

DEFAULT_MODEL = "gemini-2.5-pro"

# ROSE removal (source shadow/reflection cleanup) is slow + VRAM-heavy. Disabled by
# default so it's never even offered to Gemini; set DISABLE_ROSE=0 to re-enable it.
DISABLE_ROSE = os.environ.get("DISABLE_ROSE", "1") == "1"

_TYPE_MAP = {
    "string": "STRING", "integer": "INTEGER", "number": "NUMBER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}


def _json_schema_to_gemini(spec: dict) -> dict:
    """Convert one contracts/tools.py input spec to a Gemini parameter schema."""
    raw_types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
    primary = next((t for t in raw_types if t != "null"), raw_types[0])
    schema = {"type": _TYPE_MAP.get(primary, "STRING"), "description": spec.get("description", "")}
    if "null" in raw_types:
        schema["nullable"] = True
    if primary == "array":
        item_type = spec.get("items", {}).get("type", "string")
        schema["items"] = {"type": _TYPE_MAP.get(item_type, "STRING")}
    if "enum" in spec:
        schema["enum"] = spec["enum"]
    if "default" in spec and spec["default"] is not None:
        schema["description"] = f"{schema['description']} Default: {spec['default']!r} — use this unless told otherwise."
    return schema


def build_tool() -> "types.Tool":
    from google.genai import types
    declarations = []
    for name, tool_spec in TOOLS.items():
        if DISABLE_ROSE and name == "rose_removal":
            continue  # not exposed to Gemini -> it cannot call ROSE at all
        properties, required = {}, []
        for pname, pspec in tool_spec["inputs"].items():
            properties[pname] = _json_schema_to_gemini(pspec)
            if "default" not in pspec:
                required.append(pname)
        declarations.append(types.FunctionDeclaration(
            name=name,
            description=tool_spec["description"],
            parameters={"type": "OBJECT", "properties": properties, "required": required},
        ))
    return types.Tool(function_declarations=declarations)


def call_tool(name: str, args: dict) -> dict:
    """Call tool from mcp_server (all local)."""
    fn = getattr(tools_mod, name, None)
    if fn is None:
        return {"error": f"unknown tool: {name!r}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"{name} raised: {e}"}

def _job_dir(args: dict):
    """Derive jobs/<id>/ from any path arg so timing lands next to the run's outputs."""
    for v in args.values():
        if isinstance(v, str) and "/jobs/" in v:
            head, tail = v.split("/jobs/", 1)
            return os.path.join(head, "jobs", tail.split("/", 1)[0])
    return None


def _log_timing(args: dict, name: str, begin: str, dt: float):
    """Append one clean timing line (times only) to jobs/<id>/timing.log."""
    jd = _job_dir(args)
    if not jd:
        return
    try:
        with open(os.path.join(jd, "timing.log"), "a") as fh:
            fh.write(f"{begin}  {name:<22} {dt:8.1f}s\n")
    except OSError:
        pass


def _client(api_key=None):
    from google import genai
    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) — see .env.example.")
    return genai.Client(api_key=key)


SYSTEM_PROMPT = (
    "You are an orchestrator for a video object-replacement pipeline. You have "
    "tools for each pipeline stage (sam3_mask, gemini_edit, roma_edit_mask, "
    "roma_anchors, videopainter_generate, rose_removal, composite, encode, "
    "union_masks, extract_frames, evaluate). Call them in the order needed to satisfy the "
    "user's request, using each tool's own output paths as inputs to the next "
    "tool — do not invent paths. "
    "The usual chain is: sam3_mask (SOURCE mask) + gemini_edit (ref0) in parallel "
    "-> detect_camera_motion (frames_dir + sam3_mask's mask_dir) -> roma_edit_mask "
    "(TARGET/edit mask) -> EITHER roma_anchors -> videopainter_generate "
    "(camera_motion=false, static/near-static camera) OR mvinpainter_anchors -> "
    "mvinpainter_generate (camera_motion=true, pan/dolly/zoom) -> rose_removal -> "
    "composite -> encode -> evaluate. mvinpainter_anchors needs ref0_path and the "
    "same edit-region mask_dir videopainter_generate would have used; "
    "mvinpainter_generate then consumes its anchor_map just like "
    "videopainter_generate consumes roma_anchors'. Never call both roma_anchors "
    "and mvinpainter_anchors, or both videopainter_generate and "
    "mvinpainter_generate — detect_camera_motion's camera_motion boolean picks "
    "exactly one route, and cite it (camera_motion / coherent_fraction / "
    "median_transform_px) when reporting which route was used. "
    "For evaluate, pass source_frames_dir (the ORIGINAL video's extracted frames) — "
    "required when enable_mfs is true (its own default). "
    "caps_applied non-empty means the score was capped by a hard failure detected "
    "by the judge, not just a low weighted average across dimensions — call this "
    "out explicitly when reporting results."
    "If a tool call fails, inspect the error and either fix the arguments and "
    "retry, or report the failure clearly. When the pipeline is done, report the "
    "final video path and evaluation scores to the user."
    "Compute segment_starts automatically as [0, 20, 40, 60, ...] every 20 frames "
    "when using mvinpainter_generate (camera_motion=True, pan/dolly/zoom detected), "
    "or [0, 48, 96, ...] every 48 frames when using videopainter_generate "
    "(camera_motion=False, static camera). detect_camera_motion returns camera_motion "
    "boolean — use it to pick the right stride."
    + ("ROSE removal is DISABLED — the rose_removal tool is unavailable, do not attempt it. "
       "For composite, use the ORIGINAL frames (frames_dir) as bg_frames_dir. " if DISABLE_ROSE else "")
)


def run(prompt: str, model: str = DEFAULT_MODEL, max_turns: int = 20, on_video=None) -> tuple[str, dict | None]:
    from google.genai import types

    client = _client()
    tool = build_tool()
    config = types.GenerateContentConfig(tools=[tool], system_instruction=SYSTEM_PROMPT)
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    eval_scores = None

    for turn in range(max_turns):
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        candidate = resp.candidates[0]
        contents.append(candidate.content)

        fn_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not fn_calls:
            text = resp.text or ""
            print(f"[orchestrator] final: {text}")
            return text, eval_scores

        response_parts = []
        for fc in fn_calls:
            args = dict(fc.args or {})
            _t0, _begin = time.time(), time.strftime('%H:%M:%S')
            print(f"[orchestrator] [{_begin}] turn {turn}: calling {fc.name}({args})")
            result = call_tool(fc.name, args)
            _dt = time.time() - _t0
            print(f"[orchestrator] [{time.strftime('%H:%M:%S')}] {fc.name} done in {_dt:.1f}s -> {result}")
            _log_timing(args, fc.name, _begin, _dt)
            if fc.name == "encode" and on_video and isinstance(result, dict) and result.get("video_path"):
                try:
                    on_video(result["video_path"])   # push the video to the user before eval runs
                except Exception as e:
                    print(f"[orchestrator] on_video callback failed: {e}")
            if fc.name == "evaluate":
                eval_scores = result
            response_parts.append(types.Part(function_response=types.FunctionResponse(
                name=fc.name,
                response=result if isinstance(result, dict) else {"result": result},
            )))
        contents.append(types.Content(role="user", parts=response_parts))

    raise RuntimeError(f"orchestrator hit max_turns={max_turns} without a final answer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gemini orchestrator for the video pipeline")
    ap.add_argument("prompt", help="natural-language editing request")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()
    result, eval_scores = run(args.prompt, model=args.model, max_turns=args.max_turns)
    print(result)
    if eval_scores:
        print(f"Eval scores: {eval_scores}")
