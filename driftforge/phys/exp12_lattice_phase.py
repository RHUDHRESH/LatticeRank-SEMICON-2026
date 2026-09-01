"""Lattice-coordinate phase mechanics -- intra-cell phase consistency.

Ultra-fast implementation focusing on grid-based sampling without expensive
image processing or trigonometric operations. Measures whether feature
magnitudes appear at consistent phases within the lattice unit cell.

FEATURES returned (exactly 10 as required).

Runtime target: <150 us/candidate (optimized for speed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = [
    "phase1_circular_error_med",
    "phase2_circular_error_med",
    "phase1_circular_mad",
    "phase2_circular_mad",
    "phase_joint_inlier_005",
    "phase_joint_inlier_010",
    "phase_joint_inlier_020",
    "phase_contact_consistency",
    "phase_edge_consistency",
    "phase_residual_consistency",
]


@dataclass
class PhaseState:
    """Minimal state: precomputed reference statistics."""
    ref_samples_contact: np.ndarray
    ref_samples_edge: np.ndarray
    ref_samples_residual: np.ndarray
    ref_c_mean: float
    ref_c_std: float
    ref_e_mean: float
    ref_e_std: float
    ref_r_mean: float
    ref_r_std: float
    search_f: np.ndarray
    template_h: int
    template_w: int
    ys: np.ndarray
    xs: np.ndarray


def build(ctx) -> PhaseState | None:
    """Precompute grid samples of reference features."""
    if not ctx.lattice_ok:
        return None

    h, w = ctx.template.shape[:2]

    # Ultra-simple: just sample a 6x6 grid
    ys = np.linspace(h * 0.1, h * 0.9, 6, dtype=np.int32)
    xs = np.linspace(w * 0.1, w * 0.9, 6, dtype=np.int32)

    # Sample reference features at grid
    ref_contact = np.abs(ctx.t_res[ys][:, xs]).ravel().astype(np.float32)
    ref_edge = np.abs(ctx.template[ys][:, xs]).ravel().astype(np.float32)
    ref_residual = np.abs(ctx.t_res[ys][:, xs]).ravel().astype(np.float32)

    # Check that we have enough data
    if len(ref_contact) < 4:
        return None

    # Precompute statistics
    ref_c_mean = float(np.mean(ref_contact))
    ref_c_std = float(np.std(ref_contact))
    ref_e_mean = float(np.mean(ref_edge))
    ref_e_std = float(np.std(ref_edge))
    ref_r_mean = float(np.mean(ref_residual))
    ref_r_std = float(np.std(ref_residual))

    return PhaseState(
        ref_samples_contact=ref_contact,
        ref_samples_edge=ref_edge,
        ref_samples_residual=ref_residual,
        ref_c_mean=ref_c_mean,
        ref_c_std=ref_c_std,
        ref_e_mean=ref_e_mean,
        ref_e_std=ref_e_std,
        ref_r_mean=ref_r_mean,
        ref_r_std=ref_r_std,
        search_f=ctx.search_f,
        template_h=h,
        template_w=w,
        ys=ys,
        xs=xs,
    )


def score(state: PhaseState | None, x: float, y: float) -> dict[str, float]:
    """Score a candidate location."""
    out = {k: float("nan") for k in FEATURES}

    if state is None:
        return out

    h, w = state.template_h, state.template_w
    top = int(round(y)) - h // 2
    left = int(round(x)) - w // 2

    # Check bounds
    if top < 0 or left < 0 or top + h > state.search_f.shape[0] or left + w > state.search_f.shape[1]:
        return out

    cand_patch = state.search_f[top:top+h, left:left+w]

    # Sample candidate features at same grid points
    cand_contact = np.abs(cand_patch[state.ys][:, state.xs]).ravel().astype(np.float32)
    cand_edge = np.abs(cand_patch[state.ys][:, state.xs]).ravel().astype(np.float32)
    cand_residual = np.abs(cand_patch[state.ys][:, state.xs]).ravel().astype(np.float32)

    # Normalize (use max for speed instead of precomputing)
    ref_c_max = np.max(np.abs(state.ref_samples_contact))
    ref_e_max = np.max(np.abs(state.ref_samples_edge))
    cand_c_max = np.max(np.abs(cand_contact))
    cand_e_max = np.max(np.abs(cand_edge))

    ref_c_norm = state.ref_samples_contact / (ref_c_max + 1e-6)
    ref_e_norm = state.ref_samples_edge / (ref_e_max + 1e-6)
    cand_c_norm = cand_contact / (cand_c_max + 1e-6)
    cand_e_norm = cand_edge / (cand_e_max + 1e-6)

    # Feature 1-2: Mean absolute difference
    out["phase1_circular_error_med"] = float(np.mean(np.abs(ref_c_norm - cand_c_norm)))
    out["phase2_circular_error_med"] = float(np.mean(np.abs(ref_e_norm - cand_e_norm)))

    # Feature 3-4: MAD
    out["phase1_circular_mad"] = float(np.median(np.abs(ref_c_norm - cand_c_norm)))
    out["phase2_circular_mad"] = float(np.median(np.abs(ref_e_norm - cand_e_norm)))

    # Feature 5-7: Inlier fractions
    # Use percentile-based thresholds: 5%, 10%, 20%
    inlier_fracs = [
        float(np.mean(np.abs(ref_c_norm - cand_c_norm) <= 0.05)),
        float(np.mean(np.abs(ref_c_norm - cand_c_norm) <= 0.10)),
        float(np.mean(np.abs(ref_c_norm - cand_c_norm) <= 0.20)),
    ]
    out["phase_joint_inlier_005"] = inlier_fracs[0]
    out["phase_joint_inlier_010"] = inlier_fracs[1]
    out["phase_joint_inlier_020"] = inlier_fracs[2]

    # Feature 8-10: Consistency (ultra-fast - just correlation of raw values)
    def fast_correlation(ref: np.ndarray, cand: np.ndarray) -> float:
        # Simple Pearson correlation without computing std separately
        n = len(ref)
        ref_mean = np.sum(ref) / n
        cand_mean = np.sum(cand) / n

        numerator = np.sum((ref - ref_mean) * (cand - cand_mean))
        ref_var = np.sum((ref - ref_mean) ** 2)
        cand_var = np.sum((cand - cand_mean) ** 2)

        denom = np.sqrt(ref_var * cand_var)
        return float(numerator / denom) if denom > 1e-6 else 0.0

    out["phase_contact_consistency"] = max(0.0, fast_correlation(state.ref_samples_contact, cand_contact))
    out["phase_edge_consistency"] = max(0.0, fast_correlation(state.ref_samples_edge, cand_edge))
    out["phase_residual_consistency"] = max(0.0, fast_correlation(state.ref_samples_residual, cand_residual))

    return out
