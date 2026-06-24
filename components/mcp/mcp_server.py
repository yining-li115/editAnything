import os
import pipeline_tools as pt
from mcp.server.fastmcp import FastMCP, Image
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from gemini_edit import load_dotenv
load_dotenv()

mcp = FastMCP("editAnything-pipeline")

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
    """Return a single frame/mask/anchor as an image for visual check."""
    return Image(path=path, format="png")


@mcp.tool()
def judge_video(video_path: str, source: str, target: str, n_frames: int = 3) -> dict:
    """Score an edited video with Gemini: quality, consistency, style_match (1-10)."""
    from judge import judge as run_judge
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set in server environment"}
    return run_judge(video_path, source, target, api_key, n_frames=n_frames)


if __name__ == "__main__":
    mcp.run()