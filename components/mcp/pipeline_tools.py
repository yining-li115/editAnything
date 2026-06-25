"""Plain Python functions wrapping each pipeline stage as a subprocess.
Imported by mcp_server.py and registered as MCP tools.

Each function returns structured metadata (returncode, counts, sample coverage)
for the orchestrator / judge to reason over. Subprocess (not in-process import) is
deliberate: clean VRAM on exit between SAM3 / RoMa / VideoPainter, and crash
isolation (a model OOM kills one subprocess, not the server).
"""
import glob
import os
import subprocess
import sys

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of mcp/)
sys.path.insert(0, HERE)
PY = sys.executable

from contracts import layout  # noqa: E402

_VP = layout.MODELS["videopainter"]


def _run(args, timeout=None, env=None):
    """Run `python <args>` from the repo root; capture output. `env` overlays extra
    vars (e.g. VP_OFFLOAD) on top of the inherited environment."""
    run_env = None
    if env:
        run_env = dict(os.environ)
        run_env.update(env)
    proc = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True,
                          timeout=timeout, env=run_env)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()}


def _component(name):
    return os.path.join("components", name)


def _count(d, pattern="frame_*.png"):
    return len(glob.glob(os.path.join(d, pattern))) if os.path.isdir(d) else 0


def _parse_kv(stdout):
    """Pull KEY=VALUE lines (emitted by run_roma.py) into a dict."""
    kv = {}
    for line in stdout.splitlines():
        if "=" in line and line.split("=", 1)[0].isupper():
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


# --------------------------------------------------------------------------- #
# Verification / inspection
# --------------------------------------------------------------------------- #
def list_outputs(name: str, sub: str = "frames_src", pattern: str = "frame_*.png") -> dict:
    """List artifacts under outputs/<name>/<sub>/ — for the orchestrator to verify a
    stage produced files before advancing."""
    rp = layout.RunPaths(name)
    d = os.path.join(rp.root, sub)
    files = sorted(glob.glob(os.path.join(d, pattern)))
    return {"dir": d, "pattern": pattern, "count": len(files),
            "first": files[0] if files else None, "last": files[-1] if files else None}


# --------------------------------------------------------------------------- #
# 1. extract
# --------------------------------------------------------------------------- #
def extract_frames(name: str, video: str, max_frames: int = 0) -> dict:
    """Decode `video` -> outputs/<name>/frames_src/. Returns frame count, native
    size, and the auto-derived segment starts (so the orchestrator knows the run
    shape without re-deriving it)."""
    rp = layout.RunPaths(name)
    out_dir = rp.frames_src
    os.makedirs(out_dir, exist_ok=True)
    if _count(out_dir) == 0:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video]
        if max_frames:
            cmd += ["-frames:v", str(int(max_frames))]
        cmd += ["-start_number", "1", os.path.join(out_dir, "frame_%05d.png")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip()}
    paths = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    if not paths:
        return {"error": "no frames extracted"}
    n = len(paths)
    h, w = cv2.imread(paths[0]).shape[:2]
    CLIP, STEP = 49, 48
    starts = list(range(0, max(1, n - CLIP + 1), STEP))
    tail = n - CLIP
    if tail > starts[-1]:
        starts.append(tail)
    mode = "single-pass" if n <= CLIP else f"multi-chunk ({len(starts)} segments)"
    return {"out_dir": out_dir, "n_frames": n, "native_size": [w, h],
            "mode": mode, "segment_starts": starts}


# --------------------------------------------------------------------------- #
# 2. SAM3 source mask
# --------------------------------------------------------------------------- #
def sam3_mask(name: str, text: str, frames_dir: str = "") -> dict:
    """Per-frame SAM3 mask of `text` (the source object) -> outputs/<name>/mask_src/."""
    rp = layout.RunPaths(name)
    frames_dir = frames_dir or rp.frames_src
    out_mask_dir = rp.mask_src
    res = _run([_component("sam3_mask.py"), "--frames_dir", frames_dir,
                "--text", text, "--out_mask_dir", out_mask_dir])
    res["out_mask_dir"] = out_mask_dir
    res["n_mask_files"] = _count(out_mask_dir)
    sample = sorted(glob.glob(os.path.join(out_mask_dir, "frame_*.png")))
    if sample:
        m = cv2.imread(sample[0], 0)
        res["sample_mask_coverage_pct"] = round(100 * (m > 127).mean(), 2)
        res["sample_mask_file"] = sample[0]
    return res


# --------------------------------------------------------------------------- #
# 3. Gemini reference edit (frame 0)
# --------------------------------------------------------------------------- #
def gemini_edit_frame(name: str, source: str, target: str = "",
                      image_path: str = "", ref_image: str = "") -> dict:
    """Edit frame 0 -> outputs/<name>/ref0.png (the new-object reference). Mode A
    (target text) or mode B (ref_image). Defaults image_path to frame_00001.png."""
    rp = layout.RunPaths(name)
    image_path = image_path or os.path.join(rp.frames_src, "frame_00001.png")
    out_path = os.path.join(rp.root, "ref0.png")
    args = [_component("gemini_edit.py"), "--image", image_path,
            "--out", out_path, "--source", source]
    if ref_image:
        args += ["--ref_image", ref_image]
    elif target:
        args += ["--target", target]
    else:
        return {"error": "must provide either target or ref_image"}
    res = _run(args)
    res["out_path"] = out_path
    res["out_exists"] = os.path.exists(out_path)
    return res


# --------------------------------------------------------------------------- #
# 4. RoMa edit masks + per-segment anchors (source_mask=track)
# --------------------------------------------------------------------------- #
def roma_anchors(name: str, target_word: str, source_word: str, segment_starts: list,
                 ref0_path: str = "", frames_dir: str = "", ref0_mask: str = "",
                 dilate: int = 12, region_shape: str = "rect",
                 source_mask: str = "track", timeout_s: int = 1800) -> dict:
    """Build RoMa per-frame edit masks + per-segment anchors, then (source_mask=track)
    union the per-frame SAM3 source mask on top to close the moving-source leak.
    Returns the resolved anchors_dir and mask_dir parsed from the CLI."""
    rp = layout.RunPaths(name)
    frames_dir = frames_dir or rp.frames_src
    ref0_path = ref0_path or os.path.join(rp.root, "ref0.png")
    starts_str = ",".join(str(s) for s in segment_starts)
    args = [_component("run_roma.py"), "--frames_dir", frames_dir, "--ref0", ref0_path,
            "--target_word", target_word, "--source_word", source_word,
            "--work_dir", rp.roma, "--mask_src", rp.mask_src, "--union_dir", rp.mask,
            "--segment_starts", starts_str, "--dilate", str(dilate),
            "--region_shape", region_shape, "--source_mask", source_mask]
    if ref0_mask:
        args += ["--ref0_mask", ref0_mask]
    res = _run(args, timeout=timeout_s)
    kv = _parse_kv(res["stdout"])
    anchors_dir = kv.get("ANCHORS_DIR", os.path.join(rp.roma, "anchors"))
    mask_dir = kv.get("MASK_DIR", rp.mask)
    res.update(anchors_dir=anchors_dir, mask_dir=mask_dir,
               n_anchors=_count(anchors_dir, "anchor_*.png"),
               n_masks=_count(mask_dir),
               expected_n_anchors=len(segment_starts),
               n_frames_in=_count(frames_dir))
    return res


# --------------------------------------------------------------------------- #
# 5. VideoPainter generation
# --------------------------------------------------------------------------- #
def videopainter_generate(name: str, prompt: str, segment_starts: list,
                          frames_dir: str = "", mask_dir: str = "", anchor_dir: str = "",
                          dilate: int = 12, steps: int = 50, guidance: float = 6.0,
                          seed: int = 42, offload: str = "sequential",
                          timeout_s: int = 6000) -> dict:
    """VideoPainter multi-chunk generation -> outputs/<name>/gen/frames/.
    `offload` (sequential|model|none): pass 'none' on a >=48GB card for the ~2.6x
    speedup; 'sequential' (default) is the safe small-card path."""
    rp = layout.RunPaths(name)
    frames_dir = frames_dir or rp.frames_src
    mask_dir = mask_dir or rp.mask
    anchor_dir = anchor_dir or os.path.join(rp.roma, "anchors")
    starts_str = ",".join(str(s) for s in segment_starts)
    res = _run([
        _component("videopainter.py"), "--frames_dir", frames_dir, "--mask_dir", mask_dir,
        "--anchor_dir", anchor_dir, "--out_dir", rp.gen, "--prompt", prompt,
        "--segment_starts", starts_str, "--model_path", _VP["model_path"],
        "--branch", _VP["branch"], "--id_lora", _VP["id_lora"], "--dilate", str(dilate),
        "--steps", str(steps), "--guidance", str(guidance), "--seed", str(seed),
    ], timeout=timeout_s, env={"VP_OFFLOAD": offload})
    frames_out = rp.gen_frames
    res.update(frames_out=frames_out, n_frames_generated=_count(frames_out),
               n_frames_source=_count(frames_dir), offload=offload,
               segment_lines=[l for l in res["stdout"].splitlines()
                              if l.startswith("[generate]")])
    return res


# --------------------------------------------------------------------------- #
# 6. composite
# --------------------------------------------------------------------------- #
def composite_frames(name: str, total: int, plate_dir: str = "", object_dir: str = "",
                     mask_dir: str = "", maxfilter: int = 9, feather: int = 4) -> dict:
    rp = layout.RunPaths(name)
    plate_dir = plate_dir or rp.frames_src
    object_dir = object_dir or rp.gen_frames
    mask_dir = mask_dir or rp.mask
    res = _run([_component("composite.py"), "--plate_dir", plate_dir,
                "--object_dir", object_dir, "--mask_dir", mask_dir,
                "--out_dir", rp.composite, "--total", str(total),
                "--maxfilter", str(maxfilter), "--feather", str(feather)])
    res.update(out_dir=rp.composite, n_frames=_count(rp.composite), expected=total)
    return res


# --------------------------------------------------------------------------- #
# 7. encode
# --------------------------------------------------------------------------- #
def encode_video(name: str, size_wh: str, frames_dir: str = "", fps: int = 25,
                 despike_frames: list = None) -> dict:
    rp = layout.RunPaths(name)
    frames_dir = frames_dir or rp.gen_frames
    args = [_component("encode.py"), "--frames_dir", frames_dir, "--out", rp.final,
            "--size", size_wh, "--fps", str(fps)]
    if despike_frames:
        args += ["--despike", ",".join(str(f) for f in despike_frames)]
    res = _run(args)
    res["out_path"] = rp.final
    res["out_exists"] = os.path.exists(rp.final)
    if res["out_exists"]:
        res["out_size_bytes"] = os.path.getsize(rp.final)
    return res