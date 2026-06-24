"""Data contract — run-directory layout + config-driven model registry.

The single place that knows WHERE a run's artifacts live and WHERE model weights
are. Keeping model paths here (not hardcoded in pipeline) is what lets the later
ckpt/ relocation + extra model candidates be a config change, not a code change.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (parent of contracts/)

# Model registry: checkpoint paths per candidate. Weights live in the top-level,
# gitignored ckpt/ (separate from the vendored model source). Override via config/CLI.
_CKPT = os.path.join(ROOT, "ckpt")
MODELS = {
    "videopainter": {
        "model_path": os.path.join(_CKPT, "CogVideoX-5b-I2V"),
        "branch":     os.path.join(_CKPT, "VideoPainter", "checkpoints", "branch"),
        "id_lora":    os.path.join(_CKPT, "VideoPainterID", "checkpoints"),
    },
}


class RunPaths:
    """Canonical per-run artifact locations under outputs/<name>/."""
    def __init__(self, name, out_root=None):
        self.root = os.path.join(out_root or ROOT, "outputs", name)
        self.frames_src = os.path.join(self.root, "frames_src")
        self.mask_src   = os.path.join(self.root, "mask_src")      # SAM3 source mask (assets/union)
        self.mask       = os.path.join(self.root, "mask")          # unioned edit-region (assets/union)
        self.roma       = os.path.join(self.root, "roma")          # roma/{masks,anchors}
        self.gen        = os.path.join(self.root, "gen")           # gen/frames
        self.removal    = os.path.join(self.root, "removal")       # removal/frames (ROSE clean plate)
        self.composite  = os.path.join(self.root, "composite")
        self.despike    = os.path.join(self.root, "despike_frames")
        self.final      = os.path.join(self.root, "final.mp4")

    @property
    def gen_frames(self):
        return os.path.join(self.gen, "frames")

    @property
    def clean_frames(self):
        return os.path.join(self.removal, "frames")
