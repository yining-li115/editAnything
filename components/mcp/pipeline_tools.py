"""Plain Python functions wrapping each pipeline stage CLI as a subprocess.
Imported by mcp_server.py (as MCP tools).
"""
import os
import sys
import glob
import subprocess
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _run(args, timeout=None):
    proc = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def _count(d, pattern="frame_*.png"):
    return len(glob.glob(os.path.join(d, pattern))) if os.path.isdir(d) else 0


def list_outputs(dir_path, pattern="frame_*.png"):
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    return {"dir": dir_path, "pattern": pattern, "count": len(files),
            "first": files[0] if files else None, "last": files[-1] if files else None}


def extract_frames(video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if _count(out_dir) == 0:
        proc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                               "-start_number", "1", os.path.join(out_dir, "frame_%05d.png")],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": proc.stderr.strip()}
    paths = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
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


def sam3_mask(frames_dir, text, out_mask_dir):
    res = _run(["sam3_track.py", "--frames_dir", frames_dir, "--text", text,
                "--out_mask_dir", out_mask_dir])
    res["n_mask_files"] = _count(out_mask_dir)
    sample = sorted(glob.glob(os.path.join(out_mask_dir, "frame_*.png")))
    if sample:
        m = cv2.imread(sample[0], 0)
        res["sample_mask_coverage_pct"] = round(100 * (m > 127).mean(), 2)
        res["sample_mask_file"] = sample[0]
    return res


def gemini_edit_frame(image_path, out_path, source, target=None, ref_image=None):
    args = ["gemini_edit.py", "--image", image_path, "--out", out_path, "--source", source]
    if ref_image:
        args += ["--ref_image", ref_image]
    elif target:
        args += ["--target", target]
    else:
        return {"error": "must provide either target or ref_image"}
    res = _run(args)
    res["out_exists"] = os.path.exists(out_path)
    return res


def roma_anchors(frames_dir, ref0_path, target_word, source_word, work_dir,
                segment_starts, ref0_mask=None, dilate=12, timeout_s=1800):
    starts_str = ",".join(str(s) for s in segment_starts)
    args = ["run_roma.py", "--frames_dir", frames_dir, "--ref0", ref0_path,
            "--target_word", target_word, "--source_word", source_word,
            "--work_dir", work_dir, "--segment_starts", starts_str, "--dilate", str(dilate)]
    if ref0_mask:
        args += ["--ref0_mask", ref0_mask]
    res = _run(args, timeout=timeout_s)
    anchors_dir = os.path.join(work_dir, "anchors")
    masks_dir = os.path.join(work_dir, "masks")
    res.update(anchors_dir=anchors_dir, masks_dir=masks_dir,
               n_anchors=_count(anchors_dir, "anchor_*.png"),
               n_masks=_count(masks_dir, "frame_*.png"),
               expected_n_anchors=len(segment_starts),
               n_frames_in=_count(frames_dir))
    return res


def videopainter_generate(frames_dir, mask_dir, anchor_dir, out_dir, prompt, segment_starts,
                          model_path, branch, id_lora, dilate=12, steps=50, guidance=6.0,
                          seed=42, timeout_s=6000):
    starts_str = ",".join(str(s) for s in segment_starts)
    res = _run([
        "generate.py", "--frames_dir", frames_dir, "--mask_dir", mask_dir,
        "--anchor_dir", anchor_dir, "--out_dir", out_dir, "--prompt", prompt,
        "--segment_starts", starts_str, "--model_path", model_path,
        "--branch", branch, "--id_lora", id_lora, "--dilate", str(dilate),
        "--steps", str(steps), "--guidance", str(guidance), "--seed", str(seed),
    ], timeout=timeout_s)
    frames_out = os.path.join(out_dir, "frames")
    res.update(frames_out=frames_out, n_frames_generated=_count(frames_out),
               n_frames_source=_count(frames_dir),
               segment_lines=[l for l in res["stdout"].splitlines() if l.startswith("[generate]")])
    return res


def composite_frames(plate_dir, object_dir, mask_dir, out_dir, total, maxfilter=9, feather=4):
    res = _run(["composite.py", "--plate_dir", plate_dir, "--object_dir", object_dir,
               "--mask_dir", mask_dir, "--out_dir", out_dir, "--total", str(total),
               "--maxfilter", str(maxfilter), "--feather", str(feather)])
    res.update(out_dir=out_dir, n_frames=_count(out_dir), expected=total)
    return res


def encode_video(frames_dir, out_path, size_wh, fps=25, despike_frames=None):
    args = ["encode.py", "--frames_dir", frames_dir, "--out", out_path,
            "--size", size_wh, "--fps", str(fps)]
    if despike_frames:
        args += ["--despike", ",".join(str(f) for f in despike_frames)]
    res = _run(args)
    res["out_path"] = out_path
    res["out_exists"] = os.path.exists(out_path)
    if res["out_exists"]:
        res["out_size_bytes"] = os.path.getsize(out_path)
    return res