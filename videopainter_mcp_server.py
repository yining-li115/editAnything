"""videopainter_mcp_server.py — H100-only MCP server.

Exposes ONLY videopainter_generate, not the full pipeline. Everything else
(sam3_mask, gemini_edit, roma_*, rose_removal, composite, encode, evaluate)
keeps running wherever the orchestrator runs, in-process, unchanged.

Run:
    python videopainter_mcp_server.py --http --host 127.0.0.1 --port 8100

Port 8100, not 8000 — deliberately different from the full mcp_server.py's
default, in case both ever run side by side (e.g. testing this against a
machine that also runs the full server).
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import traceback
from typing import Optional

import fastmcp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from contracts.tools import validate_tool_input, TOOLS  # noqa: E402

mcp = fastmcp.FastMCP("videopainter-only")

_vp_pipeline: dict = {}  # keyed by (model_path, branch, id_lora)


def _abs(p: str) -> str:
    return os.path.abspath(p)


def _ok(**kwargs) -> dict:
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, str) and (k.endswith("_dir") or k.endswith("_path")):
            out[k] = _abs(v)
        elif isinstance(v, dict):
            out[k] = {kk: _abs(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _err(msg: str, exc: BaseException | None = None) -> dict:
    detail = traceback.format_exc() if exc else ""
    return {"error": msg, "detail": detail}


def _guard(tool_name: str, kwargs: dict):
    errs = validate_tool_input(tool_name, kwargs)
    if errs:
        raise ValueError(f"[{tool_name}] bad inputs: {errs}")


def _count_frames(d: str) -> int:
    return len(glob.glob(os.path.join(d, "frame_*.png")))


@mcp.tool(description=TOOLS["videopainter_generate"]["description"])
def videopainter_generate(
    frames_dir: str,
    mask_dir: str,
    anchor_map: dict,
    gen_dir: str,
    segment_starts: list,
    prompt: str,
    model_path: str,
    branch: str,
    id_lora: str,
    dilate: int = 12,
    steps: int = 10,
    guidance: float = 6.0,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    global _vp_pipeline
    try:
        _guard("videopainter_generate", {
            "frames_dir": frames_dir, "mask_dir": mask_dir, "anchor_map": anchor_map,
            "gen_dir": gen_dir, "segment_starts": segment_starts, "prompt": prompt,
            "model_path": model_path, "branch": branch, "id_lora": id_lora,
        })
        from components import extract
        import components.videopainter as vp
        from PIL import Image

        gen_frames_dir = os.path.join(gen_dir, "frames")
        if resume and extract.has_frames(gen_frames_dir):
            n = _count_frames(gen_frames_dir)
            print(f"[videopainter_generate] resume: reusing {n} frames in {gen_frames_dir}")
            return _ok(gen_frames_dir=gen_frames_dir, n_frames_generated=n)

        cache_key = (model_path, branch, id_lora)
        if cache_key not in _vp_pipeline:
            print("[videopainter_generate] loading pipeline (first call)…")
            _vp_pipeline[cache_key] = vp.load_pipeline(model_path, branch, id_lora)
        pipe = _vp_pipeline[cache_key]

        def anchor_for_start(s):
            p = anchor_map.get(str(s)) or anchor_map.get(s)
            if p is None:
                raise KeyError(f"no anchor for segment start={s} in anchor_map")
            return Image.open(p).convert("RGB")

        vp.generate(
            pipe, frames_dir, mask_dir, anchor_for_start, gen_dir,
            segment_starts=segment_starts, total=vp.CLIP,
            prompt=prompt, dilate=dilate, steps=steps,
            guidance=guidance, seed=seed,
        )
        n = _count_frames(gen_frames_dir)
        return _ok(gen_frames_dir=gen_frames_dir, n_frames_generated=n)
    except Exception as e:
        return _err(f"videopainter_generate failed: {e}", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VideoPainter-only MCP server (H100)")
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if args.http:
        print(f"[videopainter_mcp_server] HTTP/SSE on {args.host}:{args.port}", flush=True)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("[videopainter_mcp_server] stdio transport", flush=True)
        mcp.run(transport="stdio")