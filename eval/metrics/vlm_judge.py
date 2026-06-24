"""VLM Judge (Gemini): subjective quality dims automatic metrics can't capture.

Uses the current `google-genai` SDK (the BENCHMARK.md sample uses the deprecated
`google.generativeai` + gemini-1.5-pro; model id is configurable here). Sends N
evenly-spaced keyframes + the replace prompt, asks for 0-10 scores per dimension,
and parses the returned JSON. Skips gracefully when no API key is set.
"""
import json
import os
import re

JUDGE_PROMPT = """You are evaluating a video object replacement result.
The original object was replaced with: "{replace_prompt}"

You are given {n} keyframes from the generated video.
Score each dimension from 0 to 10:

1. Style match (0-10): Does the replaced object visually match the description "{replace_prompt}"?
2. Edge blending (0-10): Are the boundaries between the object and background seamless and natural?
3. Physical consistency (0-10): Is the object's motion, lighting, and interaction with the scene physically plausible?
4. Shadow (0-10): Is the shadow of the replaced object realistic and consistent with the scene lighting?

Return ONLY a JSON object in this format:
{{
  "style_match": <score>,
  "edge_blending": <score>,
  "physical_consistency": <score>,
  "shadow": <score>,
  "overall": <average of above>,
  "comments": "<brief explanation>"
}}"""


class VlmUnavailable(Exception):
    """Raised when the VLM judge can't run (no key / SDK missing)."""


class GeminiJudge:
    def __init__(self, model="gemini-2.5-flash", api_key=None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise VlmUnavailable(
                "set GEMINI_API_KEY to enable the VLM judge (or run with --metrics "
                "without 'vlm')")
        try:
            from google import genai
        except ImportError as e:
            raise VlmUnavailable("pip install google-genai") from e
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def judge(self, frames, replace_prompt, n_keyframes=8):
        from PIL import Image

        from .frames import sample_indices

        idx = sample_indices(len(frames), n=n_keyframes)
        images = [Image.fromarray(frames[i]) for i in idx]
        prompt = JUDGE_PROMPT.format(replace_prompt=replace_prompt, n=len(images))

        resp = self.client.models.generate_content(
            model=self.model, contents=[prompt, *images])
        return _parse_json(resp.text)


def _parse_json(text):
    """Extract the JSON object from a model reply (tolerates ```json fences)."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in VLM reply: {text[:200]!r}")
    data = json.loads(m.group(0))
    # Recompute overall from the four dims to guard against model arithmetic slips.
    dims = ["style_match", "edge_blending", "physical_consistency", "shadow"]
    if all(d in data for d in dims):
        data["overall"] = round(sum(float(data[d]) for d in dims) / len(dims), 2)
    return data
