"""Line-edge roughness fingerprint as a signature of the physical site.

Generic lattice geometry repeats perfectly in every periodic alias. What does
NOT repeat is the microscopic, subpixel-scale deviation of the edge from its
nominal straight-line path -- this is intrinsic process variation, measured as
line-edge roughness (LER) and critical dimension (CD) variability. The actual
physical site should exhibit the same edge-roughness sequence as the reference
capture, while lattice siblings should show uncorrelated roughness.

Method
------
For each candidate at (x, y), extract a local patch and measure LER by:

1. Detecting strong edges in the patch (gradient-based, both orientations).

2. For each edge, estimate subpixel edge position per scan direction using a
   gradient-weighted centroid in a narrow band perpendicular to the edge.

3. Fit and remove the nominal edge line (low-order polynomial), leaving the
   roughness residual r(y) or r(x).

4. Compare the candidate roughness against the reference roughness using:
   - Cross-correlation of normalized roughness curves
   - Spectral-band power ratios (fine LER vs. coarse CD variation)
   - RMS magnitude agreement
   - Autocorrelation structure

Failure modes
~~~~~~~~~~~~~
LER evidence fails (build() returns None) when:
- Fewer than 2 usable lines in the reference (flat regions, severe defocus).
- Defocus: edge blur >> roughness amplitude; roughness invisible in noise.
- Shot noise: stochastic grain dominates on low-signal scenes.
- Charging/scan artefacts: coherent noise mimics roughness but hits all sites.

The module returns NaN for features it cannot compute reliably.

Architecture-specific handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FinFET: Gate lines (vertical) and fin boundaries offer strong LER signatures.
DRAM: Contact or trench boundaries are used. If absent, returns None.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal
from scipy.fft import fft, fftfreq

FEATURES = [
    "ler_corr", "ler_gradient_corr", "ler_zero_crossing_overlap",
    "ler_rms_ratio", "ler_autocorr_diff", "ler_spectral_band1", "ler_spectral_band2",
    "ler_n_lines", "ler_rms_ref",
]

NAN_FEATURES = {k: float("nan") for k in FEATURES}


def _extract_local_edges(patch: np.ndarray, band_half: int = 3) -> list[np.ndarray]:
    """Extract edge position sequences from a patch using gradient centroid.

    Returns list of roughness sequences (one per detected edge).
    Scans for both vertical and horizontal edges.
    """
    h, w = patch.shape
    if h < 12 or w < 12:
        return []

    # Smooth to reduce noise.
    smooth = ndimage.gaussian_filter(patch.astype(np.float32), 0.8)
    gy = ndimage.sobel(smooth, axis=0, mode="reflect")
    gx = ndimage.sobel(smooth, axis=1, mode="reflect")

    roughnesses = []

    # Look for vertical edges (strong gy gradient).
    for x in range(4, w - 4, 3):
        col_gy = np.abs(gy[:, x])
        if col_gy.max() < 0.02:
            continue

        # Extract edge y-position along this vertical scan.
        edge_y = []
        for y in range(band_half, h - band_half, 1):
            band_gy = np.abs(gy[y - band_half:y + band_half + 1, x - band_half:x + band_half + 1])
            band_1d = band_gy.mean(axis=1)
            if band_1d.sum() > 1e-6:
                y_pos = y - band_half + np.sum(np.arange(len(band_1d)) * band_1d) / band_1d.sum()
                edge_y.append(y_pos)

        if len(edge_y) >= 8:
            # Fit and remove polynomial trend.
            edge_y_arr = np.array(edge_y, dtype=np.float32)
            try:
                coeffs = np.polyfit(np.arange(len(edge_y_arr)), edge_y_arr, 2)
                trend = np.polyval(coeffs, np.arange(len(edge_y_arr)))
                roughness = edge_y_arr - trend
                roughnesses.append(roughness)
            except (np.linalg.LinAlgError, ValueError):
                pass

    # Look for horizontal edges (strong gx gradient).
    for y in range(4, h - 4, 3):
        row_gx = np.abs(gx[y, :])
        if row_gx.max() < 0.02:
            continue

        # Extract edge x-position along this horizontal scan.
        edge_x = []
        for x in range(band_half, w - band_half, 1):
            band_gx = np.abs(gx[y - band_half:y + band_half + 1, x - band_half:x + band_half + 1])
            band_1d = band_gx.mean(axis=0)
            if band_1d.sum() > 1e-6:
                x_pos = x - band_half + np.sum(np.arange(len(band_1d)) * band_1d) / band_1d.sum()
                edge_x.append(x_pos)

        if len(edge_x) >= 8:
            # Fit and remove polynomial trend.
            edge_x_arr = np.array(edge_x, dtype=np.float32)
            try:
                coeffs = np.polyfit(np.arange(len(edge_x_arr)), edge_x_arr, 2)
                trend = np.polyval(coeffs, np.arange(len(edge_x_arr)))
                roughness = edge_x_arr - trend
                roughnesses.append(roughness)
            except (np.linalg.LinAlgError, ValueError):
                pass

    return roughnesses


def _compare_roughness_lists(ref_roughnesses: list[np.ndarray],
                              cand_roughnesses: list[np.ndarray]) -> dict:
    """Compute LER comparison features between reference and candidate roughness lists."""
    features = NAN_FEATURES.copy()

    if len(ref_roughnesses) == 0 or len(cand_roughnesses) == 0:
        return features

    features["ler_n_lines"] = float(len(ref_roughnesses))

    # Reference RMS.
    ref_rms_vals = np.array([np.sqrt(np.mean(r ** 2)) for r in ref_roughnesses])
    if len(ref_rms_vals) > 0:
        features["ler_rms_ref"] = float(np.mean(ref_rms_vals))

    # Candidate RMS ratio.
    cand_rms_vals = np.array([np.sqrt(np.mean(r ** 2)) for r in cand_roughnesses])
    if len(cand_rms_vals) > 0 and len(ref_rms_vals) > 0:
        features["ler_rms_ratio"] = float(np.mean(cand_rms_vals) / (np.mean(ref_rms_vals) + 1e-6))

    # Cross-correlations between reference and candidate roughnesses.
    correlations = []
    grad_correlations = []
    zero_cross_overlaps = []

    for cand_r in cand_roughnesses:
        if len(cand_r) < 4:
            continue

        best_corr = -2.0
        best_grad_corr = -2.0
        best_zc = 0.0

        for ref_r in ref_roughnesses:
            if len(ref_r) < 4:
                continue

            min_len = min(len(ref_r), len(cand_r))
            if min_len < 4:
                continue

            r1 = ref_r[:min_len]
            r2 = cand_r[:min_len]

            # Normalize.
            r1_norm = (r1 - np.mean(r1)) / (np.std(r1) + 1e-6)
            r2_norm = (r2 - np.mean(r2)) / (np.std(r2) + 1e-6)

            # Direct correlation.
            corr = float(np.mean(r1_norm * r2_norm))
            best_corr = max(best_corr, corr)

            # Gradient correlation (derivatives).
            g1 = np.diff(r1_norm)
            g2 = np.diff(r2_norm)
            if len(g1) > 0 and len(g2) > 0:
                grad_corr = float(np.mean(g1[:min(len(g1), len(g2))] * g2[:min(len(g1), len(g2))]))
                best_grad_corr = max(best_grad_corr, grad_corr)

            # Zero-crossing overlap (peaks/troughs).
            if len(r1) > 4 and len(r2) > 4:
                r1_zc = np.where(np.diff(np.sign(np.diff(r1_norm))))[0] + 1
                r2_zc = np.where(np.diff(np.sign(np.diff(r2_norm))))[0] + 1

                if len(r1_zc) > 0 and len(r2_zc) > 0:
                    overlap = sum(1 for czc in r2_zc if np.any(np.abs(r1_zc - czc) <= 1))
                    best_zc = max(best_zc, overlap / len(r1_zc))

        if best_corr > -1.5:
            correlations.append(best_corr)
        if best_grad_corr > -1.5:
            grad_correlations.append(best_grad_corr)
        if best_zc > 0:
            zero_cross_overlaps.append(best_zc)

    if len(correlations) > 0:
        features["ler_corr"] = float(np.mean(correlations))
    if len(grad_correlations) > 0:
        features["ler_gradient_corr"] = float(np.mean(grad_correlations))
    if len(zero_cross_overlaps) > 0:
        features["ler_zero_crossing_overlap"] = float(np.mean(zero_cross_overlaps))

    # Autocorrelation difference: compare ACF structures.
    acf_diffs = []
    for cand_r in cand_roughnesses:
        if len(cand_r) < 8:
            continue

        cand_r_norm = (cand_r - np.mean(cand_r)) / (np.std(cand_r) + 1e-6)
        cand_acf = np.correlate(cand_r_norm, cand_r_norm, mode="full")
        cand_acf = cand_acf[len(cand_acf) // 2 : len(cand_acf) // 2 + 8]
        cand_acf = cand_acf / (np.abs(cand_acf[0]) + 1e-6)

        for ref_r in ref_roughnesses:
            if len(ref_r) < 8:
                continue

            ref_r_norm = (ref_r - np.mean(ref_r)) / (np.std(ref_r) + 1e-6)
            ref_acf = np.correlate(ref_r_norm, ref_r_norm, mode="full")
            ref_acf = ref_acf[len(ref_acf) // 2 : len(ref_acf) // 2 + 8]
            ref_acf = ref_acf / (np.abs(ref_acf[0]) + 1e-6)

            min_len = min(len(cand_acf), len(ref_acf))
            if min_len > 1:
                diff = float(np.mean(np.abs(cand_acf[:min_len] - ref_acf[:min_len])))
                acf_diffs.append(diff)

    if len(acf_diffs) > 0:
        features["ler_autocorr_diff"] = float(np.mean(acf_diffs))

    # Spectral features: compare power in different frequency bands.
    spec_band1_diffs = []
    spec_band2_diffs = []

    for cand_r in cand_roughnesses:
        if len(cand_r) < 8:
            continue

        cand_r_norm = (cand_r - np.mean(cand_r)) / (np.std(cand_r) + 1e-6)
        cand_fft = np.abs(fft(cand_r_norm)) ** 2

        for ref_r in ref_roughnesses:
            if len(ref_r) < 8:
                continue

            min_len = min(len(cand_r), len(ref_r))
            if min_len < 8:
                continue

            ref_r_norm = (ref_r[:min_len] - np.mean(ref_r[:min_len])) / (np.std(ref_r[:min_len]) + 1e-6)
            cand_r_resamp = cand_r_norm[:min_len]
            cand_fft = np.abs(fft(cand_r_resamp)) ** 2
            ref_fft = np.abs(fft(ref_r_norm)) ** 2

            freqs = fftfreq(min_len)
            abs_freqs = np.abs(freqs)

            # Band 1: 1-3 px period => normalized freq ~0.33-1.0.
            band1 = (abs_freqs >= 0.2) & (abs_freqs <= 0.5)
            band1_c = cand_fft[band1].sum() if band1.any() else 1e-6
            band1_r = ref_fft[band1].sum() if band1.any() else 1e-6
            if band1_r > 1e-6:
                spec_band1_diffs.append(np.log(band1_c / band1_r + 1e-6))

            # Band 2: 3-10 px period => normalized freq ~0.1-0.33.
            band2 = (abs_freqs >= 0.05) & (abs_freqs <= 0.25)
            band2_c = cand_fft[band2].sum() if band2.any() else 1e-6
            band2_r = ref_fft[band2].sum() if band2.any() else 1e-6
            if band2_r > 1e-6:
                spec_band2_diffs.append(np.log(band2_c / band2_r + 1e-6))

    if len(spec_band1_diffs) > 0:
        features["ler_spectral_band1"] = float(np.mean(spec_band1_diffs))
    if len(spec_band2_diffs) > 0:
        features["ler_spectral_band2"] = float(np.mean(spec_band2_diffs))

    return features


def build(ctx) -> dict | None:
    """Extract reference LER signatures from the template.

    Detects edges and measures roughness in the reference template.
    Caches the search image for later candidate scoring.
    Returns None if fewer than 2 usable lines are found.
    """
    ref_template = ctx.template.astype(np.float32)

    # Extract roughness from reference template.
    ref_roughnesses = _extract_local_edges(ref_template)

    if len(ref_roughnesses) < 2:
        return None  # Abstain: not enough lines.

    # Cache search image (normalized) for score() to extract candidate patches.
    return {
        "ref_roughnesses": ref_roughnesses,
        "search_img": ctx.search_f,  # Already normalized to [0, 1]
        "half_w": ctx.half_w,
        "half_h": ctx.half_h,
        "template_h": ctx.template.shape[0],
        "template_w": ctx.template.shape[1],
    }


def score(state: dict, x: float, y: float) -> dict:
    """Score a candidate by extracting and comparing LER at (x, y).

    Extracts a patch centered at (x, y) from the cached search image,
    measures LER roughness, and compares to reference.
    """
    if state is None:
        return NAN_FEATURES.copy()

    ref_roughnesses = state.get("ref_roughnesses", [])
    if len(ref_roughnesses) < 2:
        return NAN_FEATURES.copy()

    search_img = state.get("search_img")
    if search_img is None:
        return NAN_FEATURES.copy()

    # Extract a patch around the candidate.
    half_w = state.get("half_w", 42.0)
    half_h = state.get("half_h", 42.0)
    patch_h = int(state.get("template_h", 84))
    patch_w = int(state.get("template_w", 84))

    top = int(round(y)) - patch_h // 2
    left = int(round(x)) - patch_w // 2

    # Bounds check.
    if top < 0 or left < 0 or top + patch_h > search_img.shape[0] or left + patch_w > search_img.shape[1]:
        return NAN_FEATURES.copy()

    patch = search_img[top:top + patch_h, left:left + patch_w]

    # Extract roughness from candidate patch.
    cand_roughnesses = _extract_local_edges(patch)

    if len(cand_roughnesses) == 0:
        return NAN_FEATURES.copy()

    # Compare reference and candidate roughness.
    features = _compare_roughness_lists(ref_roughnesses, cand_roughnesses)

    return features
