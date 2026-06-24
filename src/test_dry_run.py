"""Dry-run: calls pipeline_tools functions directly (no MCP), driven by
config.yaml (the cup2_2chunk / roma run, 97 frames -> segments [0,48]).

    python test_dry_run.py [--stop_after extract|mask|roma|generate|encode]
"""
import os, sys, argparse, yaml
import mcp.pipeline_tools as pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
ap.add_argument("--stop_after", default=None,
                choices=["extract", "mask", "roma", "generate", "encode"])
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))

def p(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)

out_root = p(cfg.get("out_root", "outputs"))
run_dir = os.path.join(out_root, cfg["name"])
os.makedirs(run_dir, exist_ok=True)

# 0. frames: extract from video if given, else use pre-extracted frames_dir
if cfg.get("frames_dir"):
    frames_dir = p(cfg["frames_dir"])
    meta = pt.list_outputs(frames_dir)
    print("frames (pre-extracted):", meta)
    n_frames = meta["count"]
else:
    assert cfg.get("video"), "config needs either 'frames_dir' or 'video'"
    frames_dir = os.path.join(run_dir, "frames_src")
    meta = pt.extract_frames(p(cfg["video"]), frames_dir)
    print("frames (extracted):", meta)
    n_frames = meta["n_frames"]

CLIP, STEP = 49, 48
starts = list(range(0, max(1, n_frames - CLIP + 1), STEP))
tail = n_frames - CLIP
if tail > starts[-1]:
    starts.append(tail)
print("segment_starts:", starts)
if args.stop_after == "extract":
    sys.exit(0)

# 1. SAM3 source mask
mask_src = os.path.join(run_dir, "mask_src")
r1 = pt.sam3_mask(frames_dir, cfg["source"], mask_src)
print("sam3_mask:", r1)
if args.stop_after == "mask":
    sys.exit(0)

# 2. RoMa anchors + edit masks
roma_dir = os.path.join(run_dir, "roma")
r2 = pt.roma_anchors(frames_dir, p(cfg["ref0"]), cfg.get("target_word") or cfg["target"],
                     cfg["source"], roma_dir, starts, dilate=cfg.get("dilate", 12))
print("roma_anchors:", r2)
if args.stop_after == "roma":
    sys.exit(0)

# 3. VideoPainter generate
gen_dir = os.path.join(run_dir, "gen")
model_path = p(cfg.get("model_path") or "VideoPainter/ckpt/CogVideoX-5b-I2V")
branch = p(cfg.get("branch") or "VideoPainter/ckpt/VideoPainter/checkpoints/branch")
id_lora = p(cfg.get("id_lora") or "VideoPainter/ckpt/VideoPainterID/checkpoints")
r3 = pt.videopainter_generate(frames_dir, r2["masks_dir"], r2["anchors_dir"], gen_dir,
                              cfg["prompt"], starts, model_path, branch, id_lora,
                              dilate=cfg.get("dilate", 12), steps=cfg.get("steps", 50),
                              guidance=cfg.get("guidance", 6.0), seed=cfg.get("seed", 42))
print("videopainter_generate:", r3)
if args.stop_after == "generate":
    sys.exit(0)

# 4. encode (roma backend -> composite skipped, per pipeline.py)
w, h = (int(v) for v in cfg["out_size"].lower().split("x"))
final = os.path.join(run_dir, "final.mp4")
despike = [s + 1 for s in starts if s > 0] if cfg.get("interpolate") else None
r4 = pt.encode_video(r3["frames_out"], final, f"{w}x{h}", fps=cfg.get("fps", 25),
                     despike_frames=despike)
print("encode_video:", r4)