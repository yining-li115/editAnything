"""VLM Judge (Gemini): subjective quality dims automatic metrics can't capture.
Uses the current `google-genai` SDK; model id configurable. Two input modes:

  - judge(frames, prompt, n)   : sample N evenly-spaced keyframes (stills).
  - judge_video(path, prompt)  : upload the WHOLE clip via the Files API so the
                                 model sees real motion — better for the temporal
                                 dims (physical_consistency, shadow over time) that
                                 5 stills can't show. Preferred when a file path is
                                 available; the MCP judge_video tool calls this.

Skips gracefully when no API key is set.
"""
import json
import os
import re
import time

JUDGE_PROMPT = """You are evaluating a video object replacement result.
The original object was replaced with: "{replace_prompt}"
{input_desc}
Score each dimension from 0 to 10:
1. Style match (0-10): Does the replaced object visually match the description "{replace_prompt}"?
2. Edge blending (0-10): Are the boundaries between the object and background seamless and natural?
3. Physical consistency (0-10): Is the object's motion, lighting, and interaction with the scene physically plausible across the clip?
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

_KEYFRAME_DESC = "You are given {n} keyframes from the generated video."
_VIDEO_DESC = "You are given the full generated video. Judge temporal dimensions from the actual motion."


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
        """Keyframe mode: sample N stills and score them."""
        from PIL import Image
        from .frames import sample_indices
        idx = sample_indices(len(frames), n=n_keyframes)
        images = [Image.fromarray(frames[i]) for i in idx]
        prompt = JUDGE_PROMPT.format(
            replace_prompt=replace_prompt,
            input_desc=_KEYFRAME_DESC.format(n=len(images)))
        resp = self.client.models.generate_content(
            model=self.model, contents=[prompt, *images])
        return _parse_json(resp.text)

    def judge_video(self, video_path, replace_prompt, poll_s=1.0, timeout_s=300):
        """Full-video mode: upload the clip via the Files API and score the motion.

        The uploaded file is processed asynchronously by the backend; we poll until
        it leaves the PROCESSING state before sending it to the model, then delete it
        so repeated benchmark runs don't accumulate server-side files.
        """
        f = self.client.files.upload(file=video_path)
        try:
            t0 = time.time()
            while getattr(f.state, "name", str(f.state)) == "PROCESSING":
                if time.time() - t0 > timeout_s:
                    raise RuntimeError(f"file processing timed out: {video_path}")
                time.sleep(poll_s)
                f = self.client.files.get(name=f.name)
            state = getattr(f.state, "name", str(f.state))
            if state == "FAILED":
                raise RuntimeError(f"file processing failed: {video_path}")
            prompt = JUDGE_PROMPT.format(
                replace_prompt=replace_prompt, input_desc=_VIDEO_DESC)
            resp = self.client.models.generate_content(
                model=self.model, contents=[prompt, f])
            return _parse_json(resp.text)
        finally:
            try:
                self.client.files.delete(name=f.name)
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass


def _parse_json(text):
    """Extract the JSON object from a model reply (tolerates ```json fences)."""
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in VLM reply: {text[:200]!r}")
    data = json.loads(m.group(0))
    dims = ["style_match", "edge_blending", "physical_consistency", "shadow"]
    if all(d in data for d in dims):
        data["overall"] = round(sum(float(data[d]) for d in dims) / len(dims), 2)
    return data