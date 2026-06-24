"""CLIP-based metrics: CLIP score (text alignment) + temporal consistency.

One CLIP model is loaded and shared by both metrics. Image features are encoded
once per call and reused, so temporal consistency is nearly free after the score.
"""
import numpy as np
import torch


class ClipScorer:
    def __init__(self, model_name="ViT-B/32", device=None):
        import clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self._clip = clip

    @torch.no_grad()
    def _encode_images(self, frames):
        from PIL import Image

        batch = torch.stack([
            self.preprocess(Image.fromarray(f)) for f in frames
        ]).to(self.device)
        feats = self.model.encode_image(batch).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_text(self, prompt):
        tokens = self._clip.tokenize([prompt], truncate=True).to(self.device)
        feat = self.model.encode_text(tokens).float()
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score(self, frames, prompt):
        """Returns (clip_score, temporal_consistency) using one image encode.

        clip_score: mean cosine(frame, text), range ~0.15-0.35.
        temporal_consistency: mean cosine(frame_i, frame_i+1), range 0-1.
        """
        img = self._encode_images(frames)              # (N, D) L2-normalized
        txt = self._encode_text(prompt)                # (1, D)

        clip_sim = (img @ txt.T).squeeze(1)            # (N,)
        clip_score = float(clip_sim.mean().item())

        if len(frames) < 2:
            temporal = 1.0
        else:
            consec = (img[:-1] * img[1:]).sum(dim=-1)  # cosine, already normalized
            temporal = float(consec.mean().item())

        return clip_score, temporal
