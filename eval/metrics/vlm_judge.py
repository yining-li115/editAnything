"""Gemini VLM judge — structured, checklist-based edit evaluation (design doc §15).

Replaces FiVE-Acc's Qwen2.5-VL with our Gemini infra (per the project decision).
Given paired SOURCE and EDITED keyframes + the edit instruction / prompts, it
returns per-item 0/0.5/1 checklist scores across the four dimensions, plus the
critical-failure flags used by the scoring caps. Specific questions, never a
vague "how good is it" (design doc §15).
"""
import io
import json
import os
import re
import sys

DEFAULT_MODEL = "gemini-2.5-flash"

JUDGE_PROMPT = """You are a strict evaluator for a VIDEO OBJECT-REPLACEMENT edit.

The edit instruction was: "{instruction}"
Target description: "{target_prompt}"
Original object: "{source_object}"   New object: "{target_object}"
Internal transformation tier: {tier}   (easy=similar shape, medium=different silhouette, hard=large style/material change)

You are given N SOURCE keyframes (original) then N EDITED keyframes (result), in order.
Compare them and score EACH item as exactly 1.0 (pass), 0.5 (partial), or 0.0 (fail).

edit_success:
  target_replaced        - the original {source_object} is actually replaced.
  new_object_correct     - the new object matches "{target_object}".
  persists_all_frames    - the replacement is present in every edited frame.
  tier_transformation    - the tier's required change is clearly visible (silhouette diff for medium, style/material shift for hard; for easy just a valid replacement).
source_preservation:
  background_preserved   - background outside the object is unchanged vs source.
  non_target_preserved   - other objects/people are unchanged.
  camera_motion_preserved- camera framing/motion matches the source.
  lighting_preserved     - lighting/shadows/reflections stay plausible & consistent.
temporal_consistency:
  no_flicker             - no flicker/popping across frames.
  stable_identity        - the new object keeps a stable identity across frames.
  stable_appearance      - its material/color is stable across frames.
rendering_quality:
  no_severe_artifacts    - no severe artifacts/warping/blur.
  natural_boundary       - the edited region blends naturally into the scene.

Also set these BOOLEAN critical flags (true = the failure happened):
  edit_not_completed        - the requested edit was essentially not done.
  source_object_reappears   - the original {source_object} reappears in some frames.
  subject_identity_destroyed- the main subject/person identity is destroyed.
  background_camera_changed - background or camera motion is heavily changed.
  severe_temporal_flicker   - there is severe flicker.

Return ONLY one JSON object:
{{
 "edit_success": {{"target_replaced":_, "new_object_correct":_, "persists_all_frames":_, "tier_transformation":_}},
 "source_preservation": {{"background_preserved":_, "non_target_preserved":_, "camera_motion_preserved":_, "lighting_preserved":_}},
 "temporal_consistency": {{"no_flicker":_, "stable_identity":_, "stable_appearance":_}},
 "rendering_quality": {{"no_severe_artifacts":_, "natural_boundary":_}},
 "critical_flags": {{"edit_not_completed":false, "source_object_reappears":false, "subject_identity_destroyed":false, "background_camera_changed":false, "severe_temporal_flicker":false}},
 "notes": "<one sentence>"
}}"""


class VlmUnavailable(Exception):
    pass


def _load_env():
    """Load GEMINI_API_KEY from editAnything/.env without a hard dependency."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/
    import paths
    for env in (os.path.join(paths.EDITANYTHING, ".env"),
                os.path.join(paths.PROJECT, ".env")):
        if os.path.exists(env):
            for line in open(env):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class GeminiJudge:
    def __init__(self, model=DEFAULT_MODEL, api_key=None):
        _load_env()
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise VlmUnavailable("set GEMINI_API_KEY to enable the VLM judge")
        try:
            from google import genai
        except ImportError as e:
            raise VlmUnavailable("pip install google-genai") from e
        self.client = genai.Client(api_key=key)
        self.model = model

    def judge(self, src_frames, edit_frames, case, n_keyframes=6):
        """src_frames/edit_frames: lists of RGB uint8 HxWx3 arrays. Returns the
        parsed judge dict (raw checklist + critical_flags)."""
        from PIL import Image
        from .frames import sample_indices

        def pick(frames):
            idx = sample_indices(len(frames), stride=max(1, len(frames) // n_keyframes))[:n_keyframes]
            return [Image.fromarray(frames[i]) for i in idx]

        imgs = pick(src_frames) + pick(edit_frames)
        prompt = JUDGE_PROMPT.format(
            instruction=case.get("edit_instruction", ""),
            target_prompt=case.get("target_prompt", ""),
            source_object=case.get("source_object", ""),
            target_object=case.get("target_object", ""),
            tier=case.get("replacement_transformation_tier", "n/a"))
        resp = self.client.models.generate_content(model=self.model, contents=[prompt, *imgs])
        return _parse_json(resp.text)


def _parse_json(text):
    m = re.search(r"\{.*\}", (text or "").strip(), re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in VLM reply: {(text or '')[:200]!r}")
    return json.loads(m.group(0))


def dim_means(judge):
    """Average the 0/0.5/1 checklist items per dimension -> {dim: mean in [0,1]}."""
    out = {}
    for dim in ("edit_success", "source_preservation", "temporal_consistency", "rendering_quality"):
        vals = [float(v) for v in judge.get(dim, {}).values()]
        out[dim] = round(sum(vals) / len(vals), 4) if vals else None
    return out
