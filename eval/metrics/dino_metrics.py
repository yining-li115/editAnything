"""DINO score: appearance consistency of the replaced object across frames.

Crops the masked object region per frame, extracts DINOv2 features, and averages
cosine similarity between consecutive frames. More sensitive to object identity /
appearance than CLIP. Falls back to the full frame when masks are unavailable.
"""
import numpy as np
import torch
from torchvision import transforms


class DinoScorer:
    def __init__(self, model_name="dinov2_vits14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def score(self, frames, masks=None):
        """frames: list of RGB uint8. masks: aligned list of {0,1} or None.

        Returns mean consecutive-frame cosine similarity over masked crops.
        """
        from PIL import Image

        from .frames import crop_to_mask

        if masks is None:
            masks = [None] * len(frames)

        tensors = []
        for frame, mask in zip(frames, masks):
            crop = crop_to_mask(frame, mask)
            tensors.append(self.transform(Image.fromarray(crop)))
        batch = torch.stack(tensors).to(self.device)

        feats = self.model(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)

        if len(feats) < 2:
            return 1.0
        consec = (feats[:-1] * feats[1:]).sum(dim=-1)
        return float(consec.mean().item())
