"""Combine raw FiVE metrics + Gemini judge into the 4 dimensions + final score.

Design doc §6-§8: four dimensions (edit_success, source_preservation,
temporal_consistency, rendering_quality) weighted into a final score, then
critical-failure caps and tier caps applied. Weights/normalization from
configs/scoring.yaml. Missing components are renormalized over the present ones.
"""


def _norm(raw, spec):
    if raw is None:
        return None
    lo, hi = spec["lo"], spec["hi"]
    v = max(0.0, min(1.0, (raw - lo) / (hi - lo))) if hi > lo else 0.0
    return 1.0 - v if spec.get("invert") else v


def _combine(components, weights):
    """Weighted mean over present (non-None) components, renormalizing weights."""
    present = {k: v for k, v in components.items() if v is not None and weights.get(k)}
    if not present:
        return None
    wsum = sum(weights[k] for k in present)
    return sum(weights[k] * v for k, v in present.items()) / wsum


def score_case(raw, judge, cfg, tier=None):
    """raw: dict from five_adapter (psnr_unedit, ssim_unedit, ..., mfs, niqe, clip_target).
    judge: parsed Gemini judge dict (checklist + critical_flags). Returns a record
    with normalized components, 4 dimension scores, final_score, and caps applied."""
    nm = cfg["normalize"]
    from .vlm_judge import dim_means
    vlm = dim_means(judge)                       # per-dim checklist means in [0,1]
    flags = judge.get("critical_flags", {}) if judge else {}

    # ---- normalized automatic components ----
    n = {
        "clip_target": _norm(raw.get("clip_target"), nm["clip"]),
        "psnr": _norm(raw.get("psnr_unedit"), nm["psnr"]),
        "ssim": _norm(raw.get("ssim_unedit"), nm["ssim"]),
        "lpips": _norm(raw.get("lpips_unedit"), nm["lpips"]),
        "structure_distance": _norm(raw.get("structure_distance"), nm["structure_distance"]),
        "niqe": _norm(raw.get("niqe"), nm["niqe"]),
        "mfs": _norm(raw.get("mfs"), nm["mfs"]),
    }
    outside_mask_visual = _combine(
        {"psnr": n["psnr"], "ssim": n["ssim"], "lpips": n["lpips"],
         "structure_distance": n["structure_distance"]},
        cfg["outside_mask_blend"])

    sw = cfg["subweights"]
    dims = {
        "edit_success": _combine(
            {"vlm_edit_success": vlm.get("edit_success"), "clip_target": n["clip_target"]},
            sw["edit_success"]),
        "source_preservation": _combine(
            {"outside_mask_visual": outside_mask_visual, "vlm_preservation": vlm.get("source_preservation")},
            sw["source_preservation"]),
        "temporal_consistency": _combine(
            {"mfs": n["mfs"], "vlm_temporal": vlm.get("temporal_consistency")},
            sw["temporal_consistency"]),
        # rendering: NIQE if present, else fall back to the VLM quality checklist
        "rendering_quality": _combine(
            {"niqe": n["niqe"], "vlm_quality": vlm.get("rendering_quality")},
            {"niqe": sw["rendering_quality"].get("niqe", 1.0), "vlm_quality": 0.5 if n["niqe"] is None else 0.0}),
    }

    caps_applied = []
    # temporal dim cap: severe flicker (design doc: temporal_consistency <= 0.40)
    if flags.get("severe_temporal_flicker") and dims["temporal_consistency"] is not None:
        tcap = cfg["critical_failure_caps"]["severe_temporal_flicker_temporal_cap"]
        if dims["temporal_consistency"] > tcap:
            dims["temporal_consistency"] = tcap
            caps_applied.append("severe_temporal_flicker->temporal<=%.2f" % tcap)

    # tier-specific edit_success caps (before final)
    tt = (judge.get("edit_success", {}) or {}).get("tier_transformation") if judge else None
    tier_final_cap = None
    tc = cfg.get("tier_caps", {})
    if tier == "medium" and tt is not None and float(tt) < 0.5:
        rule = tc.get("medium", {}).get("silhouette_not_different", {})
        if dims["edit_success"] is not None and rule.get("edit_success_cap") is not None:
            dims["edit_success"] = min(dims["edit_success"], rule["edit_success_cap"])
        tier_final_cap = rule.get("final_cap")
        caps_applied.append("medium:silhouette_not_different")
    elif tier == "hard" and tt is not None and float(tt) < 0.5:
        rule = tc.get("hard", {}).get("material_style_not_visible", {})
        if rule.get("edit_success_zero"):
            dims["edit_success"] = 0.0
        tier_final_cap = rule.get("final_cap")
        caps_applied.append("hard:material_style_not_visible")

    # ---- weighted final over present dims ----
    final = _combine(dims, cfg["weights"])

    # ---- general critical-failure caps (after final) ----
    gc = cfg["critical_failure_caps"]
    cap_map = {
        "edit_not_completed": gc["edit_not_completed"],
        "source_object_reappears": gc["source_object_reappears"],
        "subject_identity_destroyed": gc["subject_identity_destroyed"],
        "background_camera_changed": gc["background_camera_changed"],
    }
    hard_cap = 1.0
    for flag, capval in cap_map.items():
        if flags.get(flag):
            hard_cap = min(hard_cap, capval)
            caps_applied.append(f"{flag}->final<={capval:.2f}")
    if tier_final_cap is not None:
        hard_cap = min(hard_cap, tier_final_cap)
    if final is not None:
        final = min(final, hard_cap)

    return {
        "final_score": round(final, 4) if final is not None else None,
        "dimensions": {k: (round(v, 4) if v is not None else None) for k, v in dims.items()},
        "components": {
            "outside_mask_visual": round(outside_mask_visual, 4) if outside_mask_visual is not None else None,
            **{k: (round(v, 4) if v is not None else None) for k, v in n.items()},
            "vlm": vlm,
        },
        "critical_flags": {k: bool(v) for k, v in flags.items()},
        "caps_applied": caps_applied,
    }
