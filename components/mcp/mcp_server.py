"""MCP server (HTTP/SSE) exposing each editAnything stage as a tool.

Transport: SSE — same-machine for now (client connects to localhost), but the
same code reaches a remote GPU node later by binding a routable host/port, with
no redesign (this is why we don't use stdio).

Fixes vs. the old server:
  - judge_video called a phantom `judge` module; it now drives the real eval
    harness (eval/metrics/vlm_judge.py -> GeminiJudge), matching what run_eval.py
    actually uses, and accepts full_video=True to upload the whole clip via the
    Files API instead of sampling keyframes.
  - tools come from the repointed pipeline_tools (components/ + RunPaths).
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_tools as pt
from mcp.server.fastmcp import FastMCP, Image

from components.gemini_edit import load_dotenv

load_dotenv(os.path.join(HERE, ".env"))

mcp = FastMCP("editAnything-pipeline")

# Stage tools (each takes a run `name`; paths resolved via RunPaths).
mcp.tool()(pt.extract_frames)
mcp.tool()(pt.sam3_mask)
mcp.tool()(pt.gemini_edit_frame)
mcp.tool()(pt.roma_anchors)
mcp.tool()(pt.videopainter_generate)
mcp.tool()(pt.composite_frames)
mcp.tool()(pt.encode_video)
mcp.tool()(pt.list_outputs)


@mcp.tool()
def inspect_frame(path: str) -> Image:
    """Return a single frame/mask/anchor as an image for a visual check."""
    return Image(path=path, format="png")


@mcp.tool()
def judge_video(video_path: str, replace_prompt: str,
                n_keyframes: int = 8, full_video: bool = True) -> dict:
    """Score an edited video with the Gemini VLM judge (style_match, edge_blending,
    physical_consistency, shadow, overall; 0-10). full_video=True uploads the whole
    clip via the Files API (better for temporal dims); False samples n_keyframes."""
    sys.path.insert(0, os.path.join(HERE, "eval"))
    from metrics.vlm_judge import GeminiJudge, VlmUnavailable
    try:
        judge = GeminiJudge()
    except VlmUnavailable as e:
        return {"error": str(e)}
    if not os.path.exists(video_path):
        return {"error": f"video missing: {video_path}"}
    try:
        if full_video and hasattr(judge, "judge_video"):
            return judge.judge_video(video_path, replace_prompt)
        from metrics import frames as fr
        all_frames = fr.load_video(video_path)
        return judge.judge(all_frames, replace_prompt, n_keyframes)
    except Exception as e:  # noqa: BLE001
        return {"error": f"judge failed: {e}"}


if __name__ == "__main__":
    # Streamable-HTTP transport (the current MCP standard; SSE is deprecated).
    # Host/port via env so the same script serves localhost now and a routable
    # address (remote GPU) later with no code change. Endpoint: http://<host>:<port>/mcp
    # To fall back to SSE on an older SDK: change the transport string to "sse".
    mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
    mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
    mcp.run(transport="streamable-http")