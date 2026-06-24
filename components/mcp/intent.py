"""Intent parsing: user prompt -> {source, target, style}.
Single source of truth for turning a chat request into structured fields.
Extracted from the old orchestrator.py.
"""
import json
from models import PARSE_MODEL

PARSE_PROMPT = """You are a video editing assistant. Given a user prompt describing an object replacement,
return ONLY a JSON object with these fields:
- source: the object to remove (single noun, e.g. "cup")
- target: the new object description (e.g. "a cyberpunk banana with neon lights")
- style: style keywords (e.g. "cyberpunk, neon, futuristic")
User prompt: {prompt}
Return ONLY the JSON, no markdown, no explanation."""


def parse_intent(prompt, client):
    """Parse a user prompt into {source, target, style} using Gemini.

    Args:
        prompt: the raw user request.
        client: an initialized google.genai.Client (reused, not recreated).
    Returns:
        dict with keys source, target, style.
    """
    resp = client.models.generate_content(
        model=PARSE_MODEL,
        contents=PARSE_PROMPT.format(prompt=prompt),
    )
    text = resp.text.strip().strip("```json").strip("```").strip()
    return json.loads(text)