"""Video pipeline MCP server.

Exposes every pipeline stage as an MCP tool so the Gemini orchestrator
can call them individually over stdio. Each tool is a thin wrapper around
the existing component functions — no logic is duplicated here.

Key design rules (see backlog T1–T13):
  - GPU models (SAM3, RoMa, VideoPainter) are cached as module-level
    singletons so repeated orchestrator calls don't reload weights.
  - Every tool returns a plain dict; failures surface as {"error": ...,
    "detail": ...} rather than raising unhandled exceptions.
  - All paths in inputs and outputs are absolute.
  - resume guards prevent re-running expensive stages.

Run (stdio transport, for the Gemini orchestrator):
    python mcp_server.py

Run (HTTP/SSE transport, for browser / remote testing):
    python mcp_server.py --http --port 8000
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import traceback
from typing import Optional

import fastmcp

# ── repo root on sys.path so component imports work ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from contracts.tools import validate_tool_input, TOOLS  # noqa: E402
from contracts import layout                             # noqa: E402

mcp = fastmcp.FastMCP("video-pipeline")

# ── Singleton caches (populated lazily on first call) ─────────────────────────
_sam3_predictor = None
_vp_pipeline: dict = {}   # keyed by (model_path, branch, id_lora)
_clip_scorer = None      # <-- add
_dino_scorer = None      # <-- add
_vlm_judge = None        # <-- add


# ── helpers ───────────────────────────────────────────────────────────────────

def _abs(p: str) -> str:
    return os.path.abspath(p)


def _ok(**kwargs) -> dict:
    """Wrap successful output, absolutifying any *_path/*_dir/*_map values."""
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, str) and (k.endswith("_dir") or k.endswith("_path")):
            out[k] = _abs(v)
        elif isinstance(v, dict):          # anchor_map: values are paths
            out[k] = {kk: _abs(vv) if isinstance(vv, str) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _err(msg: str, exc: BaseException | None = None) -> dict:
    detail = traceback.format_exc() if exc else ""
    return {"error": msg, "detail": detail}


def _guard(tool_name: str, kwargs: dict):
    """Raise ValueError with readable message if required inputs are missing."""
    errs = validate_tool_input(tool_name, kwargs)
    if errs:
        raise ValueError(f"[{tool_name}] bad inputs: {errs}")


def _count_frames(d: str) -> int:
    return len(glob.glob(os.path.join(d, "frame_*.png")))


# ─────────────────────────────────────────────────────────────────────────────
# T2 — extract_frames
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["extract_frames"]["description"])
def extract_frames(
    video_path: str,
    out_dir: str,
    max_frames: Optional[int] = None,
    resume: bool = False,
) -> dict:
    try:
        _guard("extract_frames", {"video_path": video_path, "out_dir": out_dir})
        from components import extract
        extract.extract_frames(video_path, out_dir, resume=resume, max_frames=max_frames)
        n_frames, native_size = extract.video_meta(out_dir)
        return _ok(frames_dir=out_dir, n_frames=n_frames, native_size=list(native_size))
    except Exception as e:
        return _err(f"extract_frames failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T3 — sam3_mask
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["sam3_mask"]["description"])
def sam3_mask(
    frames_dir: str,
    source_word: str,
    out_dir: str,
    resume: bool = False,
) -> dict:
    global _sam3_predictor
    try:
        _guard("sam3_mask", {"frames_dir": frames_dir, "source_word": source_word, "out_dir": out_dir})
        from components import extract, sam3_mask as sam3
        if resume and extract.has_frames(out_dir):
            n = _count_frames(out_dir)
            print(f"[sam3_mask] resume: reusing {n} masks in {out_dir}")
            return _ok(mask_dir=out_dir, n_masks=n)
        if _sam3_predictor is None:
            print("[sam3_mask] loading SAM3 predictor (first call)…")
            _sam3_predictor = sam3.build_predictor()
        sam3.track(_sam3_predictor, frames_dir, source_word, out_dir)
        n = _count_frames(out_dir)
        return _ok(mask_dir=out_dir, n_masks=n)
    except Exception as e:
        return _err(f"sam3_mask failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T4 — roma_edit_mask
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["roma_edit_mask"]["description"])
def roma_edit_mask(
    frames_dir: str,
    ref0_path: str,
    target_word: str,
    source_word: str,
    work_dir: str,
    ref0_mask_path: Optional[str] = None,
    dilate: int = 12,
    region_shape: str = "hull",
) -> dict:
    try:
        _guard("roma_edit_mask", {
            "frames_dir": frames_dir, "ref0_path": ref0_path,
            "target_word": target_word, "source_word": source_word, "work_dir": work_dir,
        })
        from components.edit_mask import get_edit_mask
        em = get_edit_mask(
            "roma",
            frames_dir=frames_dir,
            ref0_path=ref0_path,
            target_word=target_word,
            source_word=source_word,
            ref0_mask_path=ref0_mask_path,
            work_dir=work_dir,
            dilate=dilate,
            region_shape=region_shape,
        )
        mask_dir = em.mask_dir   # triggers lazy computation
        return _ok(mask_dir=mask_dir)
    except Exception as e:
        return _err(f"roma_edit_mask failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T5 — roma_anchors
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["roma_anchors"]["description"])
def roma_anchors(
    frames_dir: str,
    ref0_path: str,
    work_dir: str,
    segment_starts: list,
) -> dict:
    try:
        _guard("roma_anchors", {
            "frames_dir": frames_dir, "ref0_path": ref0_path,
            "work_dir": work_dir, "segment_starts": segment_starts,
        })
        from components.anchor import get_anchor
        an = get_anchor(
            "roma",
            frames_dir=frames_dir,
            ref0_path=ref0_path,
            work_dir=work_dir,
            segment_starts=segment_starts,
        )
        # Eagerly build all anchors and return the map
        an.prepare()
        anchor_map = {str(s): an._path(s) for s in segment_starts}
        return _ok(anchor_map=anchor_map)
    except Exception as e:
        return _err(f"roma_anchors failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T6 — gemini_edit
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["gemini_edit"]["description"])
def gemini_edit(
    frame0_path: str,
    out_path: str,
    source: str,
    target: Optional[str] = None,
    ref_image: Optional[str] = None,
    instruction: Optional[str] = None,
    model: str = "gemini-2.5-flash-image",
) -> dict:
    try:
        _guard("gemini_edit", {"frame0_path": frame0_path, "out_path": out_path, "source": source})
        if not target and not ref_image and not instruction:
            return _err("gemini_edit: provide at least one of target, ref_image, or instruction")
        from components.gemini_edit import edit_image
        result = edit_image(
            frame0_path, out_path,
            source=source, target=target,
            ref_image=ref_image, instruction=instruction,
            model=model,
        )
        return _ok(ref0_path=result)
    except Exception as e:
        return _err(f"gemini_edit failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T7 — videopainter_generate
# ─────────────────────────────────────────────────────────────────────────────

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
    steps: int = 50,
    guidance: float = 6.0,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    global _vp_pipeline
    global _sam3_predictor
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
        if _sam3_predictor is not None:
            print("[videopainter_generate] evicting SAM3 from VRAM before loading VP...")
            _sam3_predictor = None
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
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


# ─────────────────────────────────────────────────────────────────────────────
# T8 — rose_removal
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["rose_removal"]["description"])
def rose_removal(
    frames_dir: str,
    mask_dir: str,
    out_dir: str,
    video_length: Optional[int] = None,
    prompt: str = "",
    resume: bool = False,
) -> dict:
    try:
        _guard("rose_removal", {"frames_dir": frames_dir, "mask_dir": mask_dir, "out_dir": out_dir})
        from components import extract
        clean_plate_dir = os.path.join(out_dir, "frames")
        if resume and extract.has_frames(clean_plate_dir):
            n = _count_frames(clean_plate_dir)
            print(f"[rose_removal] resume: reusing {n} clean-plate frames")
            return _ok(clean_plate_dir=clean_plate_dir, n_frames=n)
        from components.removal import remove
        remove(frames_dir, mask_dir, out_dir, video_length=video_length, prompt=prompt)
        n = _count_frames(clean_plate_dir)
        return _ok(clean_plate_dir=clean_plate_dir, n_frames=n)
    except Exception as e:
        return _err(f"rose_removal failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T9 — composite
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["composite"]["description"])
def composite(
    bg_frames_dir: str,
    gen_frames_dir: str,
    mask_dir: str,
    out_dir: str,
    total: Optional[int] = None,
) -> dict:
    try:
        _guard("composite", {
            "bg_frames_dir": bg_frames_dir, "gen_frames_dir": gen_frames_dir,
            "mask_dir": mask_dir, "out_dir": out_dir,
        })
        from components.composite import composite as do_composite
        if total is None:
            total = _count_frames(bg_frames_dir)
        do_composite(bg_frames_dir, gen_frames_dir, mask_dir, out_dir, total=total)
        n = _count_frames(out_dir)
        return _ok(composite_dir=out_dir, n_frames=n)
    except Exception as e:
        return _err(f"composite failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T10 — encode
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["encode"]["description"])
def encode(
    frames_dir: str,
    out_path: str,
    source_frames_dir: str,
    segment_starts: list,
    out_size: Optional[list] = None,
    fps: int = 25,
    interpolate: bool = True,
    despike_dir: Optional[str] = None,
) -> dict:
    try:
        _guard("encode", {"frames_dir": frames_dir, "out_path": out_path, "source_frames_dir": source_frames_dir,"segment_starts": segment_starts,})
        from components import extract, encode as enc

        if out_size is None:
            _, native = extract.video_meta(source_frames_dir)
            out_size = list(native)

        anchor_frames = [s + 1 for s in segment_starts if s > 0]

        src = frames_dir
        if interpolate and anchor_frames:
            if not despike_dir:
                despike_dir = frames_dir.rstrip("/") + "_despike"
            src = enc.despike_anchors(frames_dir, despike_dir, anchor_frames)

        enc.encode(src, out_path, tuple(out_size), fps=fps)

        import subprocess, json as _json
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", out_path],
            capture_output=True, text=True
        )
        duration = 0.0
        if probe.returncode == 0:
            try:
                duration = float(_json.loads(probe.stdout)["format"]["duration"])
            except Exception:
                pass

        return _ok(video_path=out_path, duration_seconds=duration)
    except Exception as e:
        return _err(f"encode failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T11 — union_masks
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["union_masks"]["description"])
def union_masks(
    source_mask_dir: str,
    target_mask_dir: str,
    out_dir: str,
) -> dict:
    try:
        _guard("union_masks", {
            "source_mask_dir": source_mask_dir,
            "target_mask_dir": target_mask_dir,
            "out_dir": out_dir,
        })
        from components.edit_mask import union_masks as do_union
        do_union(source_mask_dir, target_mask_dir, out_dir)
        n = _count_frames(out_dir)
        return _ok(mask_dir=out_dir, n_frames=n)
    except Exception as e:
        return _err(f"union_masks failed: {e}", e)


# ─────────────────────────────────────────────────────────────────────────────
# T12 — evaluate
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(description=TOOLS["evaluate"]["description"])
def evaluate(
    video_path: str,
    mask_dir: str,
    replace_prompt: str,
    object_prompt: str,
) -> dict:
    global _clip_scorer, _dino_scorer, _vlm_judge
    try:
        _guard("evaluate", {
            "video_path": video_path,
            "mask_dir": mask_dir,
            "replace_prompt": replace_prompt,
            "object_prompt": object_prompt,
        })

        # hardcoded config
        clip_model    = "ViT-B/32"
        dino_model    = "dinov2_vits14"
        vlm_model     = "gemini-2.5-flash"
        sample_every  = 10
        vlm_keyframes = 8
        weights       = {"clip": 0.3, "temporal": 0.2, "dino": 0.2, "vlm": 0.3}
        clip_norm     = {"lo": 0.15, "hi": 0.35}

        import sys
        sys.path.insert(0, os.path.join(_HERE, "eval"))
        from metrics.frames import (
            load_video, load_masks, sample_indices, align_masks_to_frames
        )
        from metrics.clip_metrics import ClipScorer
        from metrics.dino_metrics import DinoScorer
        from metrics.vlm_judge import GeminiJudge, VlmUnavailable

        # load models once
        if _clip_scorer is None:
            print("[evaluate] loading CLIP...")
            _clip_scorer = ClipScorer(clip_model)
        if _dino_scorer is None:
            print("[evaluate] loading DINOv2...")
            _dino_scorer = DinoScorer(dino_model)
        if _vlm_judge is None:
            try:
                print("[evaluate] loading Gemini VLM judge...")
                _vlm_judge = GeminiJudge(vlm_model)
            except VlmUnavailable as e:
                print(f"[evaluate] VLM judge disabled: {e}")

        # load frames
        all_frames = load_video(video_path)
        idx        = sample_indices(len(all_frames), every=sample_every)
        frames     = [all_frames[i] for i in idx]

        rec = {
            "clip_score":           None,
            "temporal_consistency": None,
            "dino_score":           None,
            "vlm_judge":            None,
        }

        # CLIP + temporal
        cs, tc = _clip_scorer.score(frames, replace_prompt)
        rec["clip_score"]           = round(cs, 4)
        rec["temporal_consistency"] = round(tc, 4)

        # DINO
        masks = align_masks_to_frames(
            load_masks(mask_dir), idx, len(all_frames))
        if all(m is None for m in masks):
            print("[evaluate] no masks found; DINO using full frames")
        rec["dino_score"] = round(_dino_scorer.score(frames, masks), 4)

        # VLM judge
        if _vlm_judge is not None:
            try:
                rec["vlm_judge"] = _vlm_judge.judge(
                    all_frames, replace_prompt, vlm_keyframes)
            except Exception as e:
                print(f"[evaluate] VLM judge failed: {e}")

        # final score
        def normalize_clip(raw):
            return max(0.0, min(1.0, (raw - clip_norm["lo"]) /
                                     (clip_norm["hi"] - clip_norm["lo"])))

        parts = {}
        if rec["clip_score"]           is not None: parts["clip"]     = normalize_clip(rec["clip_score"])
        if rec["temporal_consistency"] is not None: parts["temporal"] = rec["temporal_consistency"]
        if rec["dino_score"]           is not None: parts["dino"]     = rec["dino_score"]
        if rec["vlm_judge"] and rec["vlm_judge"].get("overall") is not None:
            parts["vlm"] = float(rec["vlm_judge"]["overall"]) / 10.0

        wsum = sum(weights[k] for k in parts)
        rec["final_score"] = round(
            sum(weights[k] * v for k, v in parts.items()) / wsum, 4
        ) if wsum > 0 else None

        return _ok(**rec)
    except Exception as e:
        return _err(f"evaluate failed: {e}", e)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Video pipeline MCP server")
    ap.add_argument("--http", action="store_true", help="HTTP/SSE transport instead of stdio")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    if args.http:
        print(f"[mcp_server] HTTP/SSE on {args.host}:{args.port}", flush=True)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("[mcp_server] stdio transport", flush=True)
        mcp.run(transport="stdio")