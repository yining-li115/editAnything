"""Adapter around FiVE-Bench's real MetricsCalculator (no reimplementation).

We reuse FiVE-Bench/evaluation/metrics_calculator.py verbatim for the automatic
metrics — outside-mask PSNR/SSIM/LPIPS/MSE, DINO structure_distance, CLIP
similarity, and CoTracker Motion-Fidelity-Score — and add pyiqa NIQE. FiVE-Acc
(Qwen2.5-VL) is stubbed out (we use the Gemini judge instead), which also avoids
its flash-attn dependency and its hard exit() on load failure.

Runs in the `five-bench` conda env (torch 2.4.1 + torchmetrics + cotracker + pyiqa).
"""
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/ (for paths.py)
import paths  # noqa: E402
sys.path.insert(0, os.path.join(paths.FIVE_BENCH, "evaluation"))  # FiVE's metrics_calculator

COTRACKER_CKPT = paths.COTRACKER_CKPT


def _mask3(mask_bool, size_wh):
    """bool HxW -> float32 HxWx3 at size_wh (W,H), for FiVE's img*mask multiply."""
    W, H = size_wh
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)
    m = (np.asarray(m) > 127).astype(np.float32)
    return np.repeat(m[:, :, None], 3, axis=2)


class FiveMetrics:
    def __init__(self, device="cuda", enable_mfs=True, enable_niqe=True):
        import metrics_calculator as mc     # FiVE's module

        # Stub FiVE-Acc (Qwen) so MetricsCalculator.__init__ doesn't load a 7B VLM
        # or exit() when flash-attn is absent; we score edit-success with Gemini.
        class _StubFiveAcc:
            def __init__(self, *a, **k):
                pass
        mc.FiVEAcc_Qwen_VL = _StubFiveAcc

        cfg = OmegaConf.create({
            "cotracker_model_path": COTRACKER_CKPT,
            "IQA_PyTorch_model_path": "",          # unused: we call pyiqa directly
            "five_acc_vlm_num_frames": 4,
            "five_acc_vlm_model_id": "stub",
        })
        if not enable_mfs:
            # also stub MFS if the checkpoint is missing, so init won't exit()
            class _StubMFS:
                def __init__(self, *a, **k):
                    pass
            mc.MotionFidelityScore = _StubMFS
        self.calc = mc.MetricsCalculator(device, config=cfg)
        self.device = device
        self.enable_mfs = enable_mfs and os.path.exists(COTRACKER_CKPT)

        self.niqe = None
        if enable_niqe:
            try:
                import pyiqa
                self.niqe = pyiqa.create_metric("niqe", device=device)
            except Exception as e:  # noqa: BLE001
                print(f"[five_adapter] NIQE disabled: {e}")

    # ---- per-frame automatic metrics, averaged over sampled frames ----
    def frame_metrics(self, src_frames, edit_frames, masks, target_prompt):
        """src_frames/edit_frames: RGB uint8 arrays (already at the SAME size, same
        sampled indices). masks: list of bool HxW (edit region). Returns averaged dict."""
        c = self.calc
        n = min(len(src_frames), len(edit_frames), len(masks))
        acc = {k: [] for k in ("psnr_unedit", "ssim_unedit", "lpips_unedit",
                               "mse_unedit", "structure_distance",
                               "clip_target", "clip_target_edit", "niqe")}
        for i in range(n):
            src = Image.fromarray(src_frames[i]); tgt = Image.fromarray(edit_frames[i])
            W, H = tgt.size
            if src.size != tgt.size:
                src = src.resize((W, H))
            m = _mask3(masks[i], (W, H))              # edit region (1=edited)
            um = 1.0 - m                              # unedited region
            acc["psnr_unedit"].append(c.calculate_psnr(src, tgt, um, um))
            acc["ssim_unedit"].append(c.calculate_ssim(src, tgt, um, um))
            acc["lpips_unedit"].append(c.calculate_lpips(src, tgt, um, um))
            acc["mse_unedit"].append(c.calculate_mse(src, tgt, um, um))
            acc["structure_distance"].append(float(c.calculate_structure_distance(src, tgt)))
            acc["clip_target"].append(c.calculate_clip_similarity(tgt, target_prompt))
            if m.sum() > 0:
                acc["clip_target_edit"].append(c.calculate_clip_similarity(tgt, target_prompt, m))
            if self.niqe is not None:
                import torch
                t = torch.from_numpy(np.asarray(tgt)).permute(2, 0, 1)[None].float() / 255.0
                acc["niqe"].append(float(self.niqe(t.to(self.device)).item()))
        return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in acc.items()}

    # ---- video-level Motion Fidelity Score (source vs edited motion) ----
    def mfs(self, source_frames_dir, edited_video_path, masks):
        if not self.enable_mfs:
            return None
        try:
            box_masks = [m.astype(np.uint8) for m in masks]   # list of HxW 0/1
            return round(float(self.calc.calculate_motion_fidelity_score(
                source_frames_dir, edited_video_path, video_masks=box_masks)), 4)
        except Exception as e:  # noqa: BLE001
            print(f"[five_adapter] MFS failed: {e}")
            return None
