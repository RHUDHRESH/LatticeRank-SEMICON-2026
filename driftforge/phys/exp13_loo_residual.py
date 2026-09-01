"""Leave-neighbour-out periodic residual stability via fast summary statistics.

The periodic-background cancellation computes P = median(N_1..N_8) where the
N_i are the 8 lattice-shifted copies of the search image. The residual is then
R = I - P. But one defective neighbouring lattice cell can corrupt what the
median treats as "periodic", making a spurious candidate look unique only
because one sibling was wrong.

Experiment #13 tests STABILITY: at each pixel in the candidate patch, compute
the template residual's variance under LOO perturbation. A true site's residual
varies little when one sibling is removed; a spurious candidate's residual
varies wildly because it depended on that one sibling's error.

METHOD
------
For each candidate, work with the shift stack patch (8, h, w). Compute robust
statistics of how the per-pixel residuals change across LOO scenarios:
- mean absolute change from baseline to LOO
- spread (std) of LOO residuals
- outlier fraction (pixels where LOO residual flips sign)

This is much cheaper than computing 8 full correlations.

FEATURES
--------
- loo_res_corr_mean: mean baseline residual magnitude (analogous to signal)
- loo_res_corr_std: std of LOO residual magnitudes (analogous to stability)
- loo_res_corr_min: min of LOO residual magnitudes
- loo_res_corr_p10: 10th percentile of LOO residual magnitudes
- loo_sign_agree_mean: mean pixel fraction with consistent sign across LOO
- loo_sign_agree_min: minimum sign agreement among LOO scenarios
- loo_worst_drop: largest change in residual when one sibling is removed
- loo_best_gain: smallest change when one sibling is removed
- loo_neighbor_dependency: which neighbor removal causes largest change
- loo_rank_stability: spread of max residuals across LOO scenarios
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..residual import lattice_shift_stack

FEATURES = [
    "loo_res_corr_mean",
    "loo_res_corr_std",
    "loo_res_corr_min",
    "loo_res_corr_p10",
    "loo_sign_agree_mean",
    "loo_sign_agree_min",
    "loo_worst_drop",
    "loo_best_gain",
    "loo_neighbor_dependency",
    "loo_rank_stability",
]


@dataclass
class LOOResidualState:
    """Precomputed shift stack and baseline periodic."""

    shift_stack: np.ndarray  # (8, H, W) float32 shifted copies
    periodic_full: np.ndarray  # (H, W) float32 baseline periodic
    search_f: np.ndarray  # (H, W) float32 robust contrast search
    template: np.ndarray  # (Th, Tw) template
    margin: int  # border padding
    half_w: float
    half_h: float


def build(ctx) -> LOOResidualState | None:
    """Precompute shift stack and baseline periodic."""
    if not ctx.lattice_ok:
        return None

    # Shift stack: 8 lattice-shifted copies of the search image
    stack = lattice_shift_stack(ctx.search_f, ctx.v1, ctx.v2, order=3)

    # Baseline: median of all 8
    periodic_full = np.median(stack, axis=0).astype(np.float32)

    return LOOResidualState(
        shift_stack=stack.astype(np.float32),
        periodic_full=periodic_full,
        search_f=ctx.search_f.astype(np.float32),
        template=ctx.template.copy().astype(np.float32),
        margin=ctx.margin,
        half_w=ctx.half_w,
        half_h=ctx.half_h,
    )


def score(state: LOOResidualState | None, x: float, y: float) -> dict[str, float]:
    """Score candidate via LOO residual stability."""
    if state is None:
        return {k: float("nan") for k in FEATURES}

    th, tw = state.template.shape
    top = int(round(y)) - th // 2
    left = int(round(x)) - tw // 2

    # Bounds check
    h_search = state.shift_stack.shape[1]
    w_search = state.shift_stack.shape[2]
    if top < 0 or left < 0 or top + th > h_search or left + tw > w_search:
        return {k: float("nan") for k in FEATURES}

    # Extract patches
    template_patch = state.template
    s_search_patch = state.search_f[top:top + th, left:left + tw]
    s_periodic_patch = state.periodic_full[top:top + th, left:left + tw]
    stack_patch = state.shift_stack[:, top:top + th, left:left + tw]  # (8, h, w)

    # Crop to margin
    m = state.margin
    if m > 0 and m < th // 2:
        template_patch = template_patch[m:th - m, m:tw - m]
        s_search_patch = s_search_patch[m:th - m, m:tw - m]
        s_periodic_patch = s_periodic_patch[m:th - m, m:tw - m]
        stack_patch = stack_patch[:, m:th - m, m:tw - m]

    if stack_patch.size < 8:
        return {k: float("nan") for k in FEATURES}

    # Baseline residual
    s_baseline_resid = (s_search_patch - s_periodic_patch).ravel().astype(np.float32)
    template_flat = template_patch.ravel().astype(np.float32)

    # Compute LOO residuals efficiently: mean(7) = (sum(8) - value[i]) / 7
    stack_sum = np.sum(stack_patch, axis=0)  # (h, w)

    loo_residuals = []  # 8 x (pixels,)
    loo_residual_mags = []  # magnitude of each LOO residual

    for i in range(8):
        # LOO periodic: mean of 7 = (sum of 8 - value at i) / 7
        periodic_loo_patch = (stack_sum - stack_patch[i]) / 7.0  # (h, w)

        # LOO residual
        s_loo_resid = (s_search_patch - periodic_loo_patch).ravel().astype(np.float32)
        loo_residuals.append(s_loo_resid)
        loo_residual_mags.append(np.abs(s_loo_resid))

    # Fast statistics over LOO scenarios (no correlations)
    loo_mags = np.stack(loo_residual_mags)  # (8, pixels)

    # Mean and std of residual magnitudes
    loo_res_corr_mean = float(np.mean(loo_mags))
    loo_res_corr_std = float(np.std(loo_mags))
    loo_res_corr_min = float(np.min(loo_mags))
    loo_res_corr_p10 = float(np.percentile(loo_mags, 10))

    # Sign consistency: for each pixel, how many LOO scenarios agree in sign?
    loo_resids = np.stack(loo_residuals)  # (8, pixels)
    loo_signs = np.sign(loo_resids)  # (8, pixels) with values in {-1, 0, 1}

    # Fraction of pixels where all 8 LOO scenarios have same sign (ignoring zero)
    sign_nonzero = np.abs(loo_signs) > 0  # (8, pixels)
    sign_same = np.all(loo_signs == loo_signs[0], axis=0)  # (pixels,)
    loo_sign_agree_mean = float(np.mean(sign_same.astype(float)))
    loo_sign_agree_min = float(np.min(sign_same.astype(float)))

    # Change from baseline
    baseline_mag = np.abs(s_baseline_resid)  # (pixels,)
    changes = np.abs(loo_mags - baseline_mag[None, :])  # (8, pixels)
    loo_worst_drop = float(np.max(changes))
    loo_best_gain = float(np.min(changes))

    # Neighbor dependency: which LOO scenario has largest max change
    max_changes_per_loo = np.max(changes, axis=1)  # (8,)
    loo_neighbor_dependency = float(max_changes_per_loo.std() / (max_changes_per_loo.mean() + 1e-9))

    # Rank stability: variance of max residuals across LOO scenarios
    max_resids_per_loo = np.max(loo_mags, axis=1)  # (8,)
    loo_rank_stability = float(max_resids_per_loo.std())

    return {
        "loo_res_corr_mean": loo_res_corr_mean,
        "loo_res_corr_std": loo_res_corr_std,
        "loo_res_corr_min": loo_res_corr_min,
        "loo_res_corr_p10": loo_res_corr_p10,
        "loo_sign_agree_mean": loo_sign_agree_mean,
        "loo_sign_agree_min": loo_sign_agree_min,
        "loo_worst_drop": loo_worst_drop,
        "loo_best_gain": loo_best_gain,
        "loo_neighbor_dependency": loo_neighbor_dependency,
        "loo_rank_stability": loo_rank_stability,
    }
