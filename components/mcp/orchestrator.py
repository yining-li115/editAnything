"""Minimal orchestrator: text prompt -> Gemini parses intent -> parallel SAM3 + Gemini edit.
Usage: python src/orchestrator.py --prompt "replace the cup with a cyberpunk banana" \
           --frame /storage/slurm/s0037/input/frame_00001.png \
           --frames_dir /storage/slurm/s0037/input/frames_2chunk \
           --out_dir /storage/slurm/s0037/outputs/orchestrator_test
"""
import os
import sys
import json
import asyncio
import argparse
import time
from google import genai

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pipeline_tools as pt

PARSE_PROMPT = """You are a video editing assistant. Given a user prompt describing an object replacement,
return ONLY a JSON object with these fields:
- source: the object to remove (single noun, e.g. "cup")
- target: the new object description (e.g. "a cyberpunk banana with neon lights")
- style: style keywords (e.g. "cyberpunk, neon, futuristic")

User prompt: {prompt}

Return ONLY the JSON, no markdown, no explanation."""


def parse_intent(prompt, api_key):
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=PARSE_PROMPT.format(prompt=prompt)
    )
    text = resp.text.strip().strip("```json").strip("```").strip()
    return json.loads(text)


async def run_parallel(frames_dir, frame_path, intent, out_dir, api_key):
    """Fan out SAM3 mask + Gemini edit in parallel."""
    mask_dir = os.path.join(out_dir, "masks")
    ref0_path = os.path.join(out_dir, "ref0.png")
    loop = asyncio.get_event_loop()

    # Both are CPU/API calls — safe to run in parallel via threads
    t0 = time.time()
    print(f"[orchestrator] launching SAM3 + Gemini edit in parallel at t=0.0s")

    sam3_future = loop.run_in_executor(None, pt.sam3_mask, frames_dir, intent["source"], mask_dir)
    gemini_future = loop.run_in_executor(None, pt.gemini_edit_frame, frame_path, ref0_path,
                                         intent["source"], intent["target"], None)

    sam3_result, gemini_result = await asyncio.gather(sam3_future, gemini_future)

    print(f"[orchestrator] SAM3 done at t={time.time()-t0:.1f}s — {sam3_result['n_mask_files']} masks, "
          f"{sam3_result.get('sample_mask_coverage_pct', '?')}% coverage")
    print(f"[orchestrator] Gemini edit done at t={time.time()-t0:.1f}s — ref0 exists: {gemini_result['out_exists']}")

    return {"sam3": sam3_result, "gemini_edit": gemini_result,
            "mask_dir": mask_dir, "ref0_path": ref0_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="e.g. 'replace the cup with a cyberpunk banana'")
    ap.add_argument("--frame", required=True, help="first frame of the video (for Gemini edit)")
    ap.add_argument("--frames_dir", required=True, help="all frames (for SAM3)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    from gemini_edit import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    assert api_key, "set GEMINI_API_KEY in .env"

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: parse intent
    print(f"[orchestrator] parsing intent from: '{args.prompt}'")
    intent = parse_intent(args.prompt, api_key)
    print(f"[orchestrator] intent: {json.dumps(intent, indent=2)}")

    # Step 2: parallel fan-out
    results = asyncio.run(run_parallel(args.frames_dir, args.frame, intent, args.out_dir, api_key))

    print(f"\n[orchestrator] DONE")
    print(f"  masks -> {results['mask_dir']}")
    print(f"  ref0  -> {results['ref0_path']}")
    print(f"  next step: roma_anchors + videopainter_generate")


if __name__ == "__main__":
    main()