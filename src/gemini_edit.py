"""Gemini image editing -> a clean first-frame reference (the new object in scene).

Calls the Gemini image model (nano-banana, default `gemini-2.5-flash-image`) to
edit a single frame in place, producing the reference used by the generate stage
(and, if called per segment-start frame, a per-segment anchor).

Two modes:
  A. text  — describe the new object: --target "a ripe yellow banana"
  B. ref   — give an image of the exact object you want: --ref_image banana.png
             (the frame + the reference object are both sent to Gemini)

Both keep the edit ALIGNED/in-place (same framing, only the object swapped) — the
reference must stay pixel-aligned with the video frame or CogVideoX propagation
breaks downstream.

Setup:
    pip install google-genai
    cp .env.example .env   # set GEMINI_API_KEY   (GOOGLE_API_KEY also accepted)
"""
import os
import io

DEFAULT_MODEL = "gemini-2.5-flash-image"
HERE = os.path.dirname(os.path.abspath(__file__))

# Mode A: pure text description of the replacement object.
TEXT_TEMPLATE = (
    "Replace the {source} in this image with {target}. "
    "Keep EVERYTHING else identical: same camera framing, same composition, same "
    "background, lighting, hand, table and reflections. Do not move or rescale the "
    "scene. Output the full edited image at the same aspect ratio."
)

# Mode B: use the object shown in a reference image.
REF_TEMPLATE = (
    "The first image is a video frame. The second image shows an object on a clean "
    "background. Replace the {source} in the first image with the object from the "
    "second image. Match its size, position and perspective to where the {source} was. "
    "Keep EVERYTHING else in the first image identical: same camera framing, "
    "composition, background, lighting, hand, table and reflections. Output the full "
    "edited first image at the same aspect ratio."
)


def load_dotenv(path=None):
    """Minimal .env loader (no dependency). Reads KEY=VALUE lines into os.environ
    without overriding already-set vars. Defaults to editAnything/.env."""
    path = path or os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _client(api_key=None):
    try:
        from google import genai  # noqa: F401
    except ImportError as e:
        raise ImportError("google-genai not installed. Run: pip install google-genai") from e
    from google import genai
    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) — see .env.example.")
    return genai.Client(api_key=key)


def edit_image(image_path, out_path, *, source="cup", target="a ripe yellow banana",
               ref_image=None, instruction=None, model=DEFAULT_MODEL, api_key=None):
    """Edit one frame with Gemini; save the result. Returns out_path.

    Mode B (ref_image given) sends [prompt, frame, ref]; mode A sends [prompt, frame].
    `instruction` overrides the templated prompt if provided.
    """
    from PIL import Image
    client = _client(api_key)
    frame = Image.open(image_path).convert("RGB")

    if ref_image:
        prompt = instruction or REF_TEMPLATE.format(source=source)
        contents = [prompt, frame, Image.open(ref_image).convert("RGB")]
        mode = f"ref({os.path.basename(ref_image)})"
    else:
        prompt = instruction or TEXT_TEMPLATE.format(source=source, target=target)
        contents = [prompt, frame]
        mode = "text"

    resp = client.models.generate_content(model=model, contents=contents)
    for cand in resp.candidates:
        for part in cand.content.parts:
            data = getattr(getattr(part, "inline_data", None), "data", None)
            if data:
                out_img = Image.open(io.BytesIO(data)).convert("RGB")
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                out_img.save(out_path)
                print(f"[gemini_edit] {os.path.basename(image_path)} [{mode}] -> {out_path} ({out_img.size})")
                return out_path
    raise RuntimeError("Gemini returned no image. Check model id / prompt / quota.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gemini in-place image edit (object swap)")
    ap.add_argument("--image", required=True, help="input frame to edit")
    ap.add_argument("--out", required=True, help="output edited image path")
    ap.add_argument("--source", default="cup", help="object to remove")
    ap.add_argument("--target", default="a ripe yellow banana", help="(mode A) object to insert")
    ap.add_argument("--ref_image", default=None, help="(mode B) image of the object to insert")
    ap.add_argument("--instruction", default=None, help="override the full edit instruction")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    edit_image(args.image, args.out, source=args.source, target=args.target,
               ref_image=args.ref_image, instruction=args.instruction, model=args.model)
