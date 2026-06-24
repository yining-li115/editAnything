"""Gemini-orchestrated MCP client. Gemini receives the tool list and decides
which MCP tools to call (which, and in what order) — but NOT with what arguments.
Intent (source/target/style) is parsed ONCE up front and injected by this client,
so the canonical values never drift across tool calls.

Tools exposed to Gemini:
  prepare_inputs()         -> runs sam3_mask + gemini_edit_frame in parallel (server tools)
  check_existing_video()   -> {exists: bool}  (client-local filesystem check)
  judge_video()            -> score dict       (server tool)
"""
import os, sys, json, asyncio, argparse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "mcp"))
from gemini_edit import load_dotenv
from intent import parse_intent
from models import ORCHESTRATOR_MODEL

SYSTEM = """You orchestrate a video object-replacement pipeline. The user gives a video and a
prompt like 'replace the cup with a cyberpunk banana'. Intent has already been parsed
for you; you only decide WHICH tool to call and WHEN, not its arguments. Your job:
1. Call prepare_inputs to run segmentation + reference generation.
2. Call check_existing_video to see if a generated result already exists.
3. If it exists, call judge_video to score it and report the score to the user.
Call tools one at a time. When done, summarize the result and the score."""

# Tool schemas Gemini sees. No source/target params — this client owns those values.
TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="prepare_inputs",
        description="Run SAM3 segmentation of the source object and the Gemini reference edit, in parallel.",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
    types.FunctionDeclaration(
        name="check_existing_video",
        description="Check whether a generated output video already exists.",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
    types.FunctionDeclaration(
        name="judge_video",
        description="Score the existing edited video on quality, consistency, style match (1-10).",
        parameters=types.Schema(type="OBJECT", properties={}, required=[])),
])]


def _payload(result):
    for b in result.content:
        if getattr(b, "type", None) == "text":
            try: return json.loads(b.text)
            except json.JSONDecodeError: return {"raw": b.text}
    return {}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--existing_video", required=True)
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    assert api_key
    os.makedirs(args.out_dir, exist_ok=True)
    gem = genai.Client(api_key=api_key)

    # ---- parse intent ONCE; these values are canonical for the whole run ----
    intent = parse_intent(args.prompt, gem)
    print(f"[client] parsed intent: {json.dumps(intent)}")

    server = StdioServerParameters(command=sys.executable,
                                   args=[os.path.join(HERE, "mcp", "mcp_server.py")])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"[client] MCP server connected")

            # local executor: maps Gemini's chosen tool -> real MCP call(s),
            # injecting the canonical intent (Gemini does NOT supply arguments).
            async def execute(name, fnargs):
                if name == "prepare_inputs":
                    mask_dir = os.path.join(args.out_dir, "masks")
                    ref0 = os.path.join(args.out_dir, "ref0.png")
                    print(f"[client] prepare_inputs -> sam3_mask ∥ gemini_edit_frame")
                    sam3 = session.call_tool("sam3_mask", {
                        "frames_dir": args.frames_dir, "text": intent["source"],
                        "out_mask_dir": mask_dir})
                    edit = session.call_tool("gemini_edit_frame", {
                        "image_path": args.frame, "out_path": ref0,
                        "source": intent["source"], "target": intent["target"]})
                    r1, r2 = await asyncio.gather(sam3, edit)
                    p1, p2 = _payload(r1), _payload(r2)
                    return {
                        "sam3": {"returncode": p1.get("returncode"),
                                "n_mask_files": p1.get("n_mask_files"),
                                "coverage_pct": p1.get("sample_mask_coverage_pct")},
                        "edit": {"out_exists": p2.get("out_exists"),
                                "returncode": p2.get("returncode"),
                                "stderr": (p2.get("stderr") or "")[:300]},   # temporary, for debugging
                    }
                if name == "check_existing_video":
                    exists = os.path.exists(args.existing_video)
                    if not exists:
                        print(f"[client] WARNING: existing video not found at {args.existing_video}")
                    return {"exists": exists, "path": args.existing_video}
                if name == "judge_video":
                    if not os.path.exists(args.existing_video):
                        return {"error": f"cannot judge — video missing: {args.existing_video}"}
                    r = await session.call_tool("judge_video", {
                        "video_path": args.existing_video,
                        "source": intent["source"], "target": intent["target"],
                        "n_frames": 3})
                    return _payload(r)
                return {"error": f"unknown tool {name}"}

            # Gemini orchestration loop
            contents = [types.Content(role="user", parts=[types.Part(text=args.prompt)])]
            cfg = types.GenerateContentConfig(system_instruction=SYSTEM, tools=TOOLS)
            for step in range(8):  # safety cap
                resp = gem.models.generate_content(
                    model=ORCHESTRATOR_MODEL, contents=contents, config=cfg)
                cand = resp.candidates[0]
                parts = (cand.content.parts if cand.content and cand.content.parts else []) or []
                calls = [p.function_call for p in parts if p.function_call]
                contents.append(cand.content)
                if not calls:
                    # no tool calls this turn — print whatever text came back and stop
                    finish = getattr(cand, "finish_reason", None)
                    text = getattr(resp, "text", None)
                    if text:
                        print(f"\n[Gemini] {text}")
                    else:
                        print(f"\n[Gemini] (no text, finish_reason={finish})")
                    break
                for fc in calls:
                    print(f"[Gemini decided] call {fc.name}")
                    out = await execute(fc.name, dict(fc.args))
                    print(f"[tool result] {json.dumps(out)[:200]}")
                    contents.append(types.Content(role="user", parts=[
                        types.Part(function_response=types.FunctionResponse(
                            name=fc.name, response=out))]))


if __name__ == "__main__":
    asyncio.run(main())