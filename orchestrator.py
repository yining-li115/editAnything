"""Gemini orchestrator: drives the pipeline tools via Gemini function-calling.

Implements roadmap item #2 from README.md — a Gemini agent that parses a chat
request and calls the wrapped pipeline stages (mcp_server.py) on demand,
instead of a fixed chain (pipeline.py) or manual clicking (MCP Inspector).

This calls the tool functions directly in-process (no MCP transport) — Gemini's
function-calling only needs schemas + a way to execute them, both of which
mcp_server.py + contracts/tools.py already provide.

Setup:
    cp .env.example .env   # set GEMINI_API_KEY (GOOGLE_API_KEY also accepted)

Usage:
    python orchestrator.py "replace the cup with a ripe yellow banana, frames in \
/storage/slurm/sisi/AFM-dataset/cup/frames/cup2, source word 'cup'"
"""
from __future__ import annotations
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from contracts.tools import TOOLS  # noqa: E402
import mcp_server as tools_mod      # noqa: E402
from components.gemini_edit import load_dotenv  # noqa: E402

DEFAULT_MODEL = "gemini-2.5-pro"

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
    fn = getattr(tools_mod, name, None)
    if fn is None:
        return {"error": f"unknown tool: {name!r}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"{name} raised: {e}"}


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
    "tool — do not invent paths. The usual chain is: sam3_mask (SOURCE mask) + "
    "gemini_edit (ref0) in parallel -> roma_edit_mask (TARGET/edit mask) -> "
    "roma_anchors -> videopainter_generate -> rose_removal -> composite -> encode "
    "-> evaluate. "
    "If a tool call fails, inspect the error and either fix the arguments and "
    "retry, or report the failure clearly. When the pipeline is done, report the "
    "final video path and evaluation scores to the user."
    "Compute segment_starts automatically as [0, 48, 96, ...] every 48 frames "
    "up to n_frames, with a tail window so the last 49 frames are always covered. "
)


def run(prompt: str, model: str = DEFAULT_MODEL, max_turns: int = 20) -> tuple[str, dict | None]:
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
            print(f"[orchestrator] turn {turn}: calling {fc.name}({args})")
            result = call_tool(fc.name, args)
            print(f"[orchestrator] -> {result}")
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
