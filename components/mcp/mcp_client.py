"""Gemini-orchestrated MCP client over streamable-HTTP.

Gemini receives the tool list and decides WHICH tool to call and WHEN — but NOT
the arguments. Intent (source/target/style) is parsed ONCE up front and injected
by this client, so the canonical values never drift across calls. Every tool is
keyed off a single run `name`; the server resolves all paths via RunPaths, so the
client never passes on-disk layout.

Transport: streamable-HTTP (the current MCP standard; SSE deprecated). Local and
remote are the same code — only --server-url changes:
  local : http://127.0.0.1:8000/mcp          (default)
  remote: http://<gpu-host>:8000/mcp          (or an SSH tunnel to localhost)

High-level tools exposed to Gemini (the parallelism lives INSIDE prepare_inputs,
which fans sam3_mask ∥ gemini_edit_frame via asyncio.gather — Gemini does not
itself reason about concurrency; be accurate about this when reporting):
  prepare_inputs()        -> sam3_mask ∥ gemini_edit_frame (server tools, parallel)
  build_roma()            -> roma_anchors (edit masks + per-segment anchors)
  generate()              -> videopainter_generate
  encode()                -> encode_video
  judge_video()           -> VLM score on the final clip

A tool result carrying {"error": ...} or returncode != 0 blocks downstream stages
(the old client would judge a video even after the edit had failed on quota).
"""
import argparse
import asyncio
import json
import os
import sys

from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.gemini_edit import load_dotenv

# Orchestrator model — overridable via env so dev/orchestration can stay on a free
# key while the benchmark uses a separate paid key (keeps quota isolated).
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "gemini-2.5-flash")
INTENT_MODEL = os.environ.get("INTENT_MODEL", "gemini-2.5-flash-lite")

SYSTEM = """You orchestrate a video object-replacement pipeline. The user gives a video and a
prompt like 'replace the cup with a cyberpunk banana'. Intent is already parsed; you only
decide WHICH tool to call and WHEN, never its arguments. Normal order:
1. prepare_inputs  - segment the source object and generate the reference edit (parallel).
2. build_roma      - build per-frame edit masks + per-segment anchors.
3. generate        - run VideoPainter to produce the edited frames.
4. encode          - assemble the final video.
5. judge_video     - score the final video and report it.
If a tool result contains an "error" field or a non-zero return code, STOP and report the
failure — do NOT call later stages on a failed upstream. Call tools one at a time."""

INTENT_PROMPT = """You are a video editing assistant. Given a user prompt describing an object
replacement, return ONLY a JSON object with these fields:
- source: the object to remove (single noun, e.g. "cup")
- target: the new object description (e.g. "a cyberpunk banana with neon lights")
- style:  style keywords (e.g. "cyberpunk, neon, futuristic")
User prompt: {prompt}
Return ONLY the JSON, no markdown, no explanation."""

# Tool schemas Gemini sees — no content args; this client owns intent + name.
TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="prepare_inputs",
        description="Segment the source object (SAM3) and generate the Gemini reference edit, in parallel.",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
    types.FunctionDeclaration(
        name="build_roma",
        description="Build per-frame edit masks and per-segment anchors via RoMa (needs prepare_inputs first).",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
    types.FunctionDeclaration(
        name="generate",
        description="Run VideoPainter to generate the edited frames (needs build_roma first).",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
    types.FunctionDeclaration(
        name="encode",
        description="Assemble the generated frames into the final video (needs generate first).",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
    types.FunctionDeclaration(
        name="judge_video",
        description="Score the final edited video on style/blending/consistency/shadow (0-10).",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
])]


def parse_intent(prompt, client):
    resp = client.models.generate_content(
        model=INTENT_MODEL, contents=INTENT_PROMPT.format(prompt=prompt))
    text = resp.text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


def _payload(result):
    """Pull the first JSON/text block out of an MCP tool result."""
    for b in result.content:
        if getattr(b, "type", None) == "text":
            try:
                return json.loads(b.text)
            except json.JSONDecodeError:
                return {"raw": b.text}
    return {}


def _failed(payload):
    """True if a tool payload signals failure — used to block downstream stages."""
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return True
    if payload.get("returncode") not in (None, 0):
        return True
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="e.g. 'replace the cup with a cyberpunk banana'")
    ap.add_argument("--name", required=True, help="run name -> outputs/<name>/")
    ap.add_argument("--video", default="", help="input video (extracted if frames not present)")
    ap.add_argument("--server-url", default=os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp"),
                    help="MCP endpoint. localhost by default; point at a remote GPU host to run there.")
    ap.add_argument("--out-size", default="", help="final WxH for encode (default: native)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--offload", default="sequential", choices=["sequential", "model", "none"],
                    help="VideoPainter CPU-offload; 'none' on a >=48GB card for the ~2.6x speedup.")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    assert api_key, "set GEMINI_API_KEY (.env)"
    gem = genai.Client(api_key=api_key)

    intent = parse_intent(args.prompt, gem)
    print(f"[client] parsed intent: {json.dumps(intent)}")

    # Run-level state filled in as stages complete; injected into tool args so the
    # canonical values (name, intent, segment_starts) never drift.
    state = {"segment_starts": None, "out_size": args.out_size, "native_size": None}

    async with streamablehttp_client(args.server_url) as (read, write, _get_sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"[client] connected to {args.server_url} — server tools: "
                  f"{[t.name for t in tools.tools]}")

            async def call(tool, fnargs):
                return _payload(await session.call_tool(tool, fnargs))

            async def execute(name):
                """Map Gemini's chosen high-level tool to real MCP server call(s),
                injecting name/intent/state. Returns a compact result dict."""
                if name == "prepare_inputs":
                    # Frames must exist first; extract is cheap + idempotent.
                    if args.video:
                        ex = await call("extract_frames", {"name": args.name, "video": args.video})
                        if _failed(ex):
                            return {"stage": "extract", **ex}
                        state["segment_starts"] = ex.get("segment_starts")
                        state["native_size"] = ex.get("native_size")
                    # SAM3 source mask ∥ Gemini reference edit (real parallelism here).
                    sam3 = call("sam3_mask", {"name": args.name, "text": intent["source"]})
                    edit = call("gemini_edit_frame", {
                        "name": args.name, "source": intent["source"], "target": intent["target"]})
                    p_sam3, p_edit = await asyncio.gather(sam3, edit)
                    return {
                        "stage": "prepare_inputs",
                        "sam3": {"n_mask_files": p_sam3.get("n_mask_files"),
                                 "coverage_pct": p_sam3.get("sample_mask_coverage_pct"),
                                 "returncode": p_sam3.get("returncode")},
                        "edit": {"out_exists": p_edit.get("out_exists"),
                                 "returncode": p_edit.get("returncode")},
                        "_failed": _failed(p_sam3) or _failed(p_edit),
                        "segment_starts": state["segment_starts"],
                    }

                if name == "build_roma":
                    if not state["segment_starts"]:
                        return {"stage": "build_roma", "error": "segment_starts unknown — run prepare_inputs"}
                    r = await call("roma_anchors", {
                        "name": args.name, "target_word": intent["target"],
                        "source_word": intent["source"],
                        "segment_starts": state["segment_starts"]})
                    r["_failed"] = _failed(r) or (r.get("n_anchors") != r.get("expected_n_anchors"))
                    return {"stage": "build_roma", **r}

                if name == "generate":
                    if not state["segment_starts"]:
                        return {"stage": "generate", "error": "segment_starts unknown — run prepare_inputs"}
                    style = intent.get("style", "")
                    prompt = f"{intent['target']}, {style}".strip(", ")
                    r = await call("videopainter_generate", {
                        "name": args.name, "prompt": prompt,
                        "segment_starts": state["segment_starts"], "offload": args.offload})
                    r["_failed"] = _failed(r) or (r.get("n_frames_generated", 0) == 0)
                    return {"stage": "generate",
                            "n_frames_generated": r.get("n_frames_generated"),
                            "offload": r.get("offload"), "returncode": r.get("returncode"),
                            "_failed": r["_failed"]}

                if name == "encode":
                    nsz = state["native_size"]
                    size = state["out_size"] or (f"{nsz[0]}x{nsz[1]}" if nsz else "720x480")
                    r = await call("encode_video", {
                        "name": args.name, "size_wh": size, "fps": args.fps})
                    r["_failed"] = _failed(r) or not r.get("out_exists")
                    return {"stage": "encode", "out_path": r.get("out_path"),
                            "out_exists": r.get("out_exists"), "_failed": r["_failed"]}

                if name == "judge_video":
                    rp_final = os.path.join(HERE, "outputs", args.name, "final.mp4")
                    if not os.path.exists(rp_final):
                        return {"stage": "judge_video", "error": f"final video missing: {rp_final}"}
                    r = await call("judge_video", {
                        "video_path": rp_final,
                        "replace_prompt": intent["target"], "full_video": True})
                    return {"stage": "judge_video", **r}

                return {"error": f"unknown tool {name}"}

            # Orchestration loop.
            contents = [types.Content(role="user", parts=[types.Part(text=args.prompt)])]
            cfg = types.GenerateContentConfig(system_instruction=SYSTEM, tools=TOOLS)
            for _ in range(12):  # safety cap
                resp = gem.models.generate_content(
                    model=ORCHESTRATOR_MODEL, contents=contents, config=cfg)
                cand = resp.candidates[0]
                parts = (cand.content.parts if cand.content and cand.content.parts else []) or []
                calls = [p.function_call for p in parts if p.function_call]
                contents.append(cand.content)
                if not calls:
                    text = getattr(resp, "text", None)
                    print(f"\n[Gemini] {text}" if text else
                          f"\n[Gemini] (no text, finish_reason={getattr(cand, 'finish_reason', None)})")
                    break
                for fc in calls:
                    print(f"[Gemini decided] {fc.name}")
                    out = await execute(fc.name)
                    failed = bool(out.pop("_failed", False)) or bool(out.get("error"))
                    tag = "FAILED" if failed else "ok"
                    print(f"[tool result:{tag}] {json.dumps(out)[:220]}")
                    contents.append(types.Content(role="user", parts=[
                        types.Part(function_response=types.FunctionResponse(
                            name=fc.name,
                            response={**out, "ok": not failed}))]))


if __name__ == "__main__":
    asyncio.run(main())