"""Scan-line displacement physics -- coherence of raster distortion as evidence.

A true SEM site experiences a smooth, spatially coherent scan distortion because
the electron beam follows a consistent raster trajectory. The distortion model
for a horizontal raster is x' = x + d_x(y), where d_x varies slowly with scan row.

An authentic site should show:
    - A smooth displacement profile d_x(y) described by low-order polynomial
    - High spatial coherence: displacement stable within a strip
    - Anisotropic distortion: horizontal registration is better than vertical
    (because the raster is precisely timed horizontally; vertical drift is
    independent mechanical jitter from the Y-scan coils).

A lattice sibling, being a different physical structure at a different location,
should show:
    - Incoherent per-strip displacements
    - No smooth trend across the patch
    - Isotropic scatter (it is a misregistration, not a raster artefact).

Method: divide the candidate patch into N horizontal strips. For each strip,
estimate horizontal displacement d_i using normalized cross-correlation (NCC)
between the corresponding template strip and search strip. Fit a quadratic
d(y) = a + b*y + c*y^2 across strips, giving features for mean, trend, and
smoothness. Repeat with vertical strips to measure anisotropy. A flat strip
(low contrast) returns NaN and is excluded from the fit; letting a random argmax
vote would corrupt the measurement.

``build`` returns a small state object holding the template patch dimensions and
precomputed template strips; it returns None if the candidate patch is clipped
(out of bounds).

This experiment does NOT require lattice_ok; it measures a physical property
independent of periodicity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["FEATURES", "build", "score"]

FEATURES = [
    "scan_dx_mean",
    "scan_dx_std",
    "scan_dx_slope",
    "scan_dx_curvature",
    "scan_dx_mad",
    "scan_inlier_fraction",
    "scan_fit_residual",
    "scan_horiz_vs_vert_ratio",
    "scan_n_strips",
]


@dataclass
class State:
    """Precomputed per-scene state for scan-line displacement.

    The state stores template and search data so that score() can extract
    candidate patches without needing to pass ctx to score(). This experiment
    does not require lattice_ok; it measures a global property independent
    of periodicity.
    """
    template: np.ndarray        # Pose-correct template
    search: np.ndarray          # Raw search image
    template_h: int
    template_w: int
    n_strips: int


def build(ctx) -> State | None:
    """Assemble state once per scene. Does not depend on lattice_ok.

    This experiment measures scan raster coherence, which is a global property
    of the image formation physics, independent of lattice structure. It returns
    a state even when ctx.lattice_ok is False.

    Args:
        ctx: SceneContext with template and search image.

    Returns:
        State object with template, search, and strip count.
    """
    h, w = ctx.template.shape[:2]
    n_strips = min(8, max(2, h // 8))
    return State(
        template=ctx.template.copy(),
        search=ctx.search.copy(),
        template_h=h,
        template_w=w,
        n_strips=n_strips,
    )


def _ncc_peak_subpixel(ncc_values: np.ndarray) -> tuple[float, float]:
    """Return subpixel offset of NCC peak via parabolic interpolation.

    Args:
        ncc_values: 1D array of NCC scores at integer offsets.

    Returns:
        (offset_px, peak_ncc): subpixel offset and peak value. If the peak is at
        an edge or NCC is flat, returns the integer offset and the peak value.
    """
    if len(ncc_values) < 3:
        idx = np.argmax(ncc_values)
        return float(idx - len(ncc_values) // 2), float(ncc_values[idx])

    idx = np.argmax(ncc_values)
    if idx == 0 or idx == len(ncc_values) - 1:
        return float(idx - len(ncc_values) // 2), float(ncc_values[idx])

    y0, y1, y2 = ncc_values[idx - 1 : idx + 2]
    if np.isnan([y0, y1, y2]).any():
        return float(idx - len(ncc_values) // 2), float(ncc_values[idx])

    denom = 2.0 * (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx - len(ncc_values) // 2), float(y1)

    offset_subpx = (y0 - y2) / denom
    peak = y1 - 0.25 * (y0 - y2) * offset_subpx
    return float(offset_subpx), float(peak)


def _estimate_strip_displacement(
    template_strip: np.ndarray,
    search_strip: np.ndarray,
    search_range: int = 3,
) -> float:
    """Estimate horizontal displacement of search_strip from template_strip.

    A strip with insufficient contrast (< 1% of the template pixel range) returns
    NaN rather than a random argmax, because such a measurement would corrupt
    the coherence estimate.

    Vectorized NCC computation across displacements.

    Args:
        template_strip: 1D or 2D template slice.
        search_strip: 1D or 2D search slice (same height as template).
        search_range: Half-width of displacement search window in pixels.

    Returns:
        Subpixel displacement in pixels, or NaN if contrast insufficient.
    """
    if template_strip.size == 0 or search_strip.size == 0:
        return float("nan")

    t = template_strip.astype(np.float32)
    s = search_strip.astype(np.float32)

    tmin, tmax = np.percentile(t, [1, 99])
    t_range = tmax - tmin
    if t_range < 1e-6:
        return float("nan")

    t_mean = np.mean(t)
    t_norm = t - t_mean
    t_energy = np.sum(t_norm * t_norm)
    if t_energy < 1e-9:
        return float("nan")

    ncc_values = []
    for delta_x in range(-search_range, search_range + 1):
        s_shifted = np.roll(s, delta_x, axis=-1)

        s_mean = np.mean(s_shifted)
        s_norm = s_shifted - s_mean
        s_energy = np.sum(s_norm * s_norm)

        if s_energy < 1e-9:
            ncc_values.append(float("nan"))
        else:
            ncc = np.sum(t_norm * s_norm) / np.sqrt(t_energy * s_energy)
            ncc_values.append(float(np.clip(ncc, -1.0, 1.0)))

    ncc_arr = np.array(ncc_values, dtype=np.float32)
    valid_mask = ~np.isnan(ncc_arr)
    if not valid_mask.any():
        return float("nan")

    offset_subpx, _ = _ncc_peak_subpixel(ncc_arr)
    return offset_subpx


def _fit_displacement_polynomial(
    displacements: np.ndarray,
    positions: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Fit d(y) = a + b*y + c*y^2 to displacement samples.

    Excludes NaN samples. If fewer than 3 valid samples, returns NaNs.

    Returns:
        (a, b, c, mad, inlier_frac, residual): coefficients and diagnostic stats.
        - a: constant term (d_x mean over valid range)
        - b: linear slope
        - c: curvature (quadratic coefficient)
        - mad: median absolute deviation of residuals
        - inlier_frac: fraction of samples within 1.5 * MAD of the fit
        - residual: RMS residual of the fit
    """
    valid = ~np.isnan(displacements)
    if valid.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    y_valid = positions[valid]
    d_valid = displacements[valid]

    y_norm = (y_valid - np.mean(y_valid)) / (np.std(y_valid) + 1e-9)
    X = np.column_stack([np.ones_like(y_norm), y_norm, y_norm ** 2])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, d_valid, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    a_norm, b, c = coeffs
    residuals = d_valid - X @ coeffs
    rms_residual = float(np.sqrt(np.mean(residuals ** 2)))

    abs_residuals = np.abs(residuals)
    mad = float(np.median(abs_residuals)) if len(abs_residuals) > 0 else float("nan")

    if not np.isfinite(mad) or mad < 1e-9:
        mad = 1.0

    inlier_frac = float(np.sum(abs_residuals <= 1.5 * mad) / len(residuals))

    a = a_norm

    return a, b, c, mad, inlier_frac, rms_residual


def _estimate_displacements_1d(
    axis: int,
    state: State,
    template: np.ndarray,
    search_patch: np.ndarray,
    search_range: int = 3,
) -> np.ndarray:
    """Estimate displacements across strips along a given axis.

    Args:
        axis: 0 for horizontal strips (measure x displacement),
              1 for vertical strips (measure y displacement).
        state: State with n_strips.
        template: Template patch.
        search_patch: Search image patch (same size as template).
        search_range: Half-width of displacement search.

    Returns:
        Array of displacements (float32), one per strip. NaN for low-contrast strips.
    """
    h, w = template.shape

    if axis == 0:
        n = state.n_strips
        strip_h = h // n
        displacements = []
        for i in range(n):
            y0 = i * strip_h
            y1 = (i + 1) * strip_h if i < n - 1 else h
            t_strip = template[y0:y1, :]
            s_strip = search_patch[y0:y1, :]
            d = _estimate_strip_displacement(t_strip, s_strip, search_range)
            displacements.append(d)
        return np.array(displacements, dtype=np.float32)
    else:
        n = state.n_strips
        strip_w = w // n
        displacements = []
        for i in range(n):
            x0 = i * strip_w
            x1 = (i + 1) * strip_w if i < n - 1 else w
            t_strip = template[:, x0:x1]
            s_strip = search_patch[:, x0:x1]
            d = _estimate_strip_displacement(t_strip, s_strip, search_range)
            displacements.append(d)
        return np.array(displacements, dtype=np.float32)


def _coherence_metric(
    displacements: np.ndarray,
    mad: float,
) -> float:
    """Coherence of displacements: inverse of scatter.

    Coherence is the fraction of displacement estimates that fall within
    1.5 * MAD of the robust median. Truly coherent raster displacements
    show ~90%+ inlier fraction; random misregistration shows much lower.

    Returns a float in [0, 1], or NaN if insufficient valid data.
    """
    valid = ~np.isnan(displacements)
    if valid.sum() < 2:
        return float("nan")

    d_valid = displacements[valid]
    median_d = np.median(d_valid)
    abs_dev = np.abs(d_valid - median_d)
    mad_est = np.median(abs_dev) if len(abs_dev) > 0 else 1.0
    if mad_est < 1e-9:
        mad_est = 1.0

    coherence = float(np.sum(abs_dev <= 1.5 * mad_est) / len(d_valid))
    return coherence


def score(state: State | None, x: float, y: float) -> dict:
    """Score scan-line displacement coherence at candidate (x, y).

    Extracts a patch from the search image centered at (x, y) and measures
    the coherence and smoothness of horizontal and vertical displacement
    profiles. The key discriminative quantities are:
      - scan_fit_residual: RMS fit residual of the polynomial to displacements
      - scan_dx_std: standard deviation of displacement estimates
      - scan_horiz_vs_vert_ratio: anisotropy of coherence (true sites should
        show higher horizontal coherence due to precise timing of the horizontal
        raster, while misregistration is isotropic).

    Args:
        state: Built state from build() (can be None).
        x, y: Candidate centre in search image pixels.

    Returns:
        Dictionary with keys exactly matching FEATURES, values float or NaN.
    """
    if state is None:
        return {k: float("nan") for k in FEATURES}

    template = state.template
    search = state.search
    h, w = template.shape

    top = int(round(y)) - h // 2
    left = int(round(x)) - w // 2
    if top < 0 or left < 0 or top + h > search.shape[0] or left + w > search.shape[1]:
        return {k: float("nan") for k in FEATURES}

    search_patch = search[top : top + h, left : left + w]

    displacements_h = _estimate_displacements_1d(0, state, template, search_patch)
    displacements_v = _estimate_displacements_1d(1, state, template, search_patch)

    positions = np.linspace(0, 1, len(displacements_h), dtype=np.float32)

    a_h, b_h, c_h, mad_h, inlier_h, resid_h = _fit_displacement_polynomial(
        displacements_h, positions
    )

    a_v, b_v, c_v, mad_v, inlier_v, resid_v = _fit_displacement_polynomial(
        displacements_v, positions
    )

    coherence_h = _coherence_metric(displacements_h, mad_h)
    coherence_v = _coherence_metric(displacements_v, mad_v)

    if np.isfinite(coherence_h) and np.isfinite(coherence_v) and coherence_v > 1e-6:
        horiz_vs_vert_ratio = coherence_h / (coherence_v + 1e-9)
    else:
        horiz_vs_vert_ratio = float("nan")

    return {
        "scan_dx_mean": a_h,
        "scan_dx_std": np.nanstd(displacements_h),
        "scan_dx_slope": b_h,
        "scan_dx_curvature": c_h,
        "scan_dx_mad": mad_h,
        "scan_inlier_fraction": inlier_h,
        "scan_fit_residual": resid_h,
        "scan_horiz_vs_vert_ratio": horiz_vs_vert_ratio,
        "scan_n_strips": float(state.n_strips),
    }
