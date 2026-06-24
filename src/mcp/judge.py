"""LLM judge: sample frames from output video -> Gemini -> score + justification.
Usage: python src/mcp/judge.py --video /storage/slurm/s0037/outputs/cup2_output.mp4 \
           --source cup --target "a ripe yellow banana"
"""
import os
import sys
import json
import argparse
import cv2
from PIL import Image
from google import genai
from models import JUDGE_MODEL

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

JUDGE_PROMPT = """You are a video editing quality judge. You are given {n} frames sampled from an edited video
where a '{source}' was replaced with '{target}'.
Score the edit on each of these criteria from 1-10:
- quality: realism and visual quality of the inserted object
- consistency: does the object look the same across all frames (no flickering, no dissolving)
- style_match: does the inserted object match the described target '{target}'
Return ONLY a JSON object:
{{
  "quality": <1-10>,
  "consistency": <1-10>,
  "style_match": <1-10>,
  "overall": <1-10>,
  "justification": "<2-3 sentences>"
}}"""


def sample_frames(video_path, n=5):
    """Extract n evenly-spaced frames from video as PIL Images."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"video has no readable frames (total={total}): {video_path}")
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"could not read any frames from: {video_path}")
    return frames


def judge(video_path, source, target, api_key, n_frames=5):
    from gemini_edit import load_dotenv
    load_dotenv()
    frames = sample_frames(video_path, n=n_frames)
    print(f"[judge] sampled {len(frames)} frames from {video_path}")
    client = genai.Client(api_key=api_key)
    prompt = JUDGE_PROMPT.format(n=len(frames), source=source, target=target)
    contents = [prompt] + frames
    resp = client.models.generate_content(model=JUDGE_MODEL, contents=contents)
    text = resp.text.strip().strip("```json").strip("```").strip()
    result = json.loads(text)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--source", required=True, help="e.g. 'cup'")
    ap.add_argument("--target", required=True, help="e.g. 'a ripe yellow banana'")
    ap.add_argument("--n_frames", type=int, default=5)
    args = ap.parse_args()
    from gemini_edit import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    assert api_key, "set GEMINI_API_KEY in .env"
    result = judge(args.video, args.source, args.target, api_key, n_frames=args.n_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()