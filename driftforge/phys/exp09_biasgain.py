"""Bias-gain compensated block registration for SEM charging discrimination.

SEM charging produces spatial intensity distortion (local bright/dark change)
and can additionally deflect the beam, causing geometric error. Photometric
compensation should explain the charging while NOT explaining geometric
misregistration. So the true site may show large RAW photometric error but
small COMPENSATED error with coherent geometry, whereas a lattice sibling
stays geometrically wrong no matter how the intensities are rescaled.

Method: partition the candidate patch into blocks (use the same grid
convention as structural_descriptor._block_nccs). Per block fit the
closed-form least-squares bias/gain model S(x) = a*T(x) + b, then evaluate
the residual e(x) = S(x) - (a*T(x) + b).

The key discriminative quantity is bg_raw_minus_compensated: how much of the
mismatch photometric compensation can explain. This is more informative than
the compensated score alone because:

  - A true site with charging shows large raw photometric error but small
    compensated error (compensation explains the mismatch).
  - A false lattice sibling shows large photometric error that compensation
    *cannot* fix, because the geometry is wrong. Rescaling the intensities
    does not make misaligned structure register.

Thus the delta (raw - compensated) isolates the photometric distortion that
compensation can fix, leaving geometric misalignment as residual.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FEATURES = [
    "bg_gain_median",
    "bg_bias_median",
    "bg_gain_std",
    "bg_bias_std",
    "bg_gain_deviation",
    "bg_bias_magnitude",
    "bg_compensated_ncc",
    "bg_compensated_rmse",
    "bg_raw_minus_compensated",
    "bg_block_inlier_fraction",
]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-shape patches."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a * b).sum() / (na * nb))


@dataclass
class BiasGainState:
    """Per-scene precomputation: nothing in this experiment is scene-level.

    build(ctx) still must return a state object per the contract, but this
    experiment works entirely per-candidate from the raw search and template.
    The state is minimal and exists only to match the interface.
    """

    ctx_template: np.ndarray  # The pose-correct template from context
    ctx_search: np.ndarray    # The raw search image from context


def build(ctx) -> BiasGainState | None:
    """Precompute scene-level context.

    This experiment does not need the lattice (ctx.lattice_ok is not required),
    so it can score candidates even when lattice detection fails. Each
    candidate is scored independently by extracting its patch and performing
    block-wise bias/gain fitting against the template.
    """
    return BiasGainState(ctx_template=ctx.template.copy(), ctx_search=ctx.search.copy())


def _fit_bias_gain_block(template_block: np.ndarray, search_block: np.ndarray
                        ) -> tuple[float, float, bool]:
    """Fit S(x) = a*T(x) + b to one block via closed-form least-squares.

    Returns (gain, bias, is_valid). is_valid=False if the template block has
    near-zero variance (degenerate), in which case gain and bias are NaN and
    should be excluded from aggregates.
    """
    t = template_block.astype(np.float64).ravel()
    s = search_block.astype(np.float64).ravel()
    n = float(t.size)

    # Least-squares solution: S = a*T + b
    # minimize ||S - a*T - b||^2
    # Setting derivatives to zero: a = (n*sum(TS) - sum(T)*sum(S)) / (n*sum(T^2) - (sum(T))^2)
    #                              b = (sum(S) - a*sum(T)) / n

    sum_t = t.sum()
    sum_s = s.sum()
    sum_t2 = (t * t).sum()
    sum_ts = (t * s).sum()

    denom = n * sum_t2 - sum_t * sum_t

    # Guard against degenerate blocks (near-zero template variance)
    if abs(denom) < 1e-9:
        return float("nan"), float("nan"), False

    gain = (n * sum_ts - sum_t * sum_s) / denom
    bias = (sum_s - gain * sum_t) / n

    return float(gain), float(bias), True


def score(state: BiasGainState, x: float, y: float) -> dict[str, float]:
    """Compute bias/gain compensation features for candidate at (x, y).

    Extract a patch from the search image at the candidate location, compute
    block-wise bias/gain fits against the template, and report aggregated
    statistics. The key feature is bg_raw_minus_compensated, which measures
    how much of the raw NCC mismatch can be explained by photometric
    distortion (bias/gain only, not geometric error).
    """
    # Extract patch from search image, same size as template
    template = state.ctx_template
    search = state.ctx_search
    th, tw = template.shape
    half_h, half_w = (th - 1) / 2.0, (tw - 1) / 2.0

    # Use ctx.window-like bounds checking
    top = int(round(y)) - th // 2
    left = int(round(x)) - tw // 2
    if top < 0 or left < 0 or top + th > search.shape[0] or left + tw > search.shape[1]:
        # Patch is clipped; return NaN features
        return {k: float("nan") for k in FEATURES}

    search_patch = search[top:top + th, left:left + tw]

    # Convert to float for processing
    t_f = template.astype(np.float32)
    s_f = search_patch.astype(np.float32)

    # Block partitioning: same convention as structural_descriptor._block_nccs
    grid = 4
    ys = np.linspace(0, th, grid + 1).astype(int)
    xs = np.linspace(0, tw, grid + 1).astype(int)

    gains = []
    biases = []
    block_helps = []  # Track which blocks benefit from compensation
    valid_blocks = 0

    # Per-block least-squares bias/gain fitting
    for i in range(grid):
        for j in range(grid):
            t_block = t_f[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            s_block = s_f[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]

            gain, bias, is_valid = _fit_bias_gain_block(t_block, s_block)

            if is_valid:
                gains.append(gain)
                biases.append(bias)
                valid_blocks += 1

                # Check if compensation helps for this block
                raw_resid = t_block - s_block
                raw_mse = float(np.mean(raw_resid ** 2))
                if abs(gain) > 1e-6:
                    s_comp = (s_block - bias) / gain
                    comp_resid = t_block - s_comp
                    comp_mse = float(np.mean(comp_resid ** 2))
                    block_helps.append(comp_mse < raw_mse)
                else:
                    block_helps.append(False)

    if valid_blocks < 4:
        # Insufficient valid blocks; cannot compute reliable statistics
        return {k: float("nan") for k in FEATURES}

    gains_arr = np.array(gains, dtype=np.float64)
    biases_arr = np.array(biases, dtype=np.float64)

    # Compute aggregated statistics
    bg_gain_median = float(np.median(gains_arr))
    bg_bias_median = float(np.median(biases_arr))
    bg_gain_std = float(gains_arr.std())
    bg_bias_std = float(biases_arr.std())

    # Gain deviation: how far the average gain is from 1.0 (perfect match)
    bg_gain_deviation = abs(bg_gain_median - 1.0)

    # Bias magnitude: absolute value of median bias
    bg_bias_magnitude = abs(bg_bias_median)

    # Compute compensated image: S_comp(x) = (S(x) - bias) / gain
    # Guard against zero or near-zero gain
    if abs(bg_gain_median) < 1e-6:
        return {k: float("nan") for k in FEATURES}

    s_comp = (s_f - bg_bias_median) / bg_gain_median

    # Compensated NCC: correlation after bias/gain correction
    bg_compensated_ncc = _ncc(t_f, s_comp)

    # Compensated RMSE: residual after bias/gain compensation
    residual_comp = t_f - s_comp
    bg_compensated_rmse = float(np.sqrt(np.mean(residual_comp ** 2)))

    # Raw RMSE: residual before compensation
    residual_raw = t_f - s_f
    raw_rmse = float(np.sqrt(np.mean(residual_raw ** 2)))

    # Compute GLOBAL bias/gain (whole patch, not per-block) for comparison.
    # Per-block fitting can over-adapt; global fitting shows how much a single
    # linear correction across the entire patch can help.
    t_f_flat = t_f.astype(np.float64).ravel()
    s_f_flat = s_f.astype(np.float64).ravel()
    n = float(t_f_flat.size)
    sum_t = t_f_flat.sum()
    sum_s = s_f_flat.sum()
    sum_t2 = (t_f_flat * t_f_flat).sum()
    sum_ts = (t_f_flat * s_f_flat).sum()

    denom_global = n * sum_t2 - sum_t * sum_t
    if abs(denom_global) > 1e-9:
        a_global = (n * sum_ts - sum_t * sum_s) / denom_global
        b_global = (sum_s - a_global * sum_t) / n
        if abs(a_global) > 1e-6:
            s_comp_global = (s_f - b_global) / a_global
            residual_comp_global = t_f - s_comp_global
            comp_rmse_global = float(np.sqrt(np.mean(residual_comp_global ** 2)))
            bg_raw_minus_compensated = raw_rmse - comp_rmse_global
        else:
            bg_raw_minus_compensated = float("nan")
    else:
        bg_raw_minus_compensated = float("nan")

    # Inlier fraction: fraction of valid blocks where compensation helps
    bg_block_inlier_fraction = float(sum(block_helps)) / len(block_helps) if block_helps else 0.0

    return {
        "bg_gain_median": bg_gain_median,
        "bg_bias_median": bg_bias_median,
        "bg_gain_std": bg_gain_std,
        "bg_bias_std": bg_bias_std,
        "bg_gain_deviation": bg_gain_deviation,
        "bg_bias_magnitude": bg_bias_magnitude,
        "bg_compensated_ncc": bg_compensated_ncc,
        "bg_compensated_rmse": bg_compensated_rmse,
        "bg_raw_minus_compensated": bg_raw_minus_compensated,
        "bg_block_inlier_fraction": bg_block_inlier_fraction,
    }
