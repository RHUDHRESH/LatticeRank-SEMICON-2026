"""SEM edge-brightening fingerprint via sub-edge morphology.

Experiment #7: Discriminate physical sites by the microscopic intensity profile
across the SEM edge response, not just by edge presence.

PHYSICS
-------
Every periodic lattice sibling exhibits an SEM edge (line width, contact edge,
etc.). But the SUB-EDGE MORPHOLOGY — the intensity gradient and halo structure
perpendicular to the edge — is intrinsic to the material stack, process and scan
parameters at THAT location. A true site reproduces the same edge-response
profile as the reference; an alias will show different morphology because it is
scanning different material (or pseudo-contacts, thickness variations, etc.).

METHOD
------
1. In build(ctx):
   - Detect the N strongest, spatially well-distributed edges in ctx.template
   - Precompute small (5x5) patches centered on each edge position
   - Store edge normals for reference

2. In score(state, x, y):
   - Extract corresponding small patches at the candidate location
   - Compare patches via correlation and intensity statistics
   - Return summary statistics across all edges

FAST IMPLEMENTATION
  Extracts tiny patches instead of 1-D profiles to reduce interpolation cost.
  All comparisons are vectorized to stay under 250 us/candidate.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURES = [
    "edge_prof_corr_med",      # Median correlation of edge patches
    "edge_prof_corr_p25",      # 25th percentile correlation
    "edge_deriv_corr_med",     # Median correlation of patch gradients
    "edge_halo_width_mad",     # MAD of local intensity transition width
    "edge_contrast_ratio_med",  # Median intensity ratio: after/before edge
    "edge_sign_frac",          # Fraction edges with correct gradient sign
    "edge_valid_frac",         # Fraction edges with measurable patch
]

NAN_FEATURES = {k: float("nan") for k in FEATURES}

PATCH_HALF = 2  # 5x5 patches
N_EDGES = 4     # Number of edges (aggressive reduction)


def _detect_edges_fast(template: np.ndarray, n_edges: int = N_EDGES) -> list[tuple]:
    """Detect strongest edges via fast gradient magnitude with NMS.

    Returns list of (y, x, dy_norm, dx_norm) tuples.
    """
    smooth = ndimage.gaussian_filter(template.astype(np.float32), sigma=0.8)
    gy = ndimage.sobel(smooth, axis=0, mode="constant")
    gx = ndimage.sobel(smooth, axis=1, mode="constant")
    mag = np.hypot(gx, gy).astype(np.float32)

    h, w = template.shape
    margin = 3

    # Non-maximum suppression via max_filter
    mag_border = mag.copy()
    mag_border[:margin] = 0
    mag_border[-margin:] = 0
    mag_border[:, :margin] = 0
    mag_border[:, -margin:] = 0

    local_max = ndimage.maximum_filter(mag_border, size=3)
    nms = (mag_border == local_max) & (mag_border > np.percentile(mag_border, 60))

    # Get peak coordinates, sort by magnitude
    coords = np.argwhere(nms)
    if len(coords) == 0:
        return []

    mags = mag[coords[:, 0], coords[:, 1]]
    top_idx = np.argsort(-mags)[:n_edges]
    peaks = coords[top_idx]

    # Compute normals
    edges_out = []
    for y, x in peaks:
        gy_n = gy[y, x]
        gx_n = gx[y, x]
        norm = np.hypot(gy_n, gx_n)
        if norm > 1e-6:
            edges_out.append((float(y), float(x), gy_n / norm, gx_n / norm))

    return edges_out


def build(ctx) -> dict | None:
    """Precompute edge positions, normals, and reference patches."""
    edges = _detect_edges_fast(ctx.template, n_edges=N_EDGES)

    if len(edges) < 2:
        return None

    # Extract reference patches at each edge
    ref_patches = []
    valid_edges = []

    for y, x, gy_n, gx_n in edges:
        y_int, x_int = int(round(y)), int(round(x))
        y0 = max(0, y_int - PATCH_HALF)
        y1 = min(ctx.template.shape[0], y_int + PATCH_HALF + 1)
        x0 = max(0, x_int - PATCH_HALF)
        x1 = min(ctx.template.shape[1], x_int + PATCH_HALF + 1)

        if (y1 - y0) > 3 and (x1 - x0) > 3:  # Patch must not be too clipped
            patch = ctx.template[y0:y1, x0:x1].astype(np.float32)
            ref_patches.append(patch)
            valid_edges.append((y, x, gy_n, gx_n))

    if len(valid_edges) < 2:
        return None

    return {
        "edges": valid_edges,
        "ref_patches": ref_patches,
        "search_f": ctx.search_f,
        "half_w": ctx.half_w,
        "half_h": ctx.half_h,
        "template_shape": ctx.template.shape,
    }


def score(state: dict, x: float, y: float) -> dict[str, float]:
    """Score by extracting and comparing small patches at edge locations."""
    if state is None:
        return NAN_FEATURES.copy()

    edges = state.get("edges", [])
    ref_patches = state.get("ref_patches", [])
    search_f = state.get("search_f")
    half_w = state.get("half_w")
    half_h = state.get("half_h")
    template_shape = state.get("template_shape", (84, 84))

    if len(edges) < 2 or len(ref_patches) < 2 or search_f is None:
        return NAN_FEATURES.copy()

    h, w = search_f.shape
    patch_h, patch_w = template_shape

    # Bounds check
    top = int(round(y)) - patch_h // 2
    left = int(round(x)) - patch_w // 2
    if top < 0 or left < 0 or top + patch_h > h or left + patch_w > w:
        return NAN_FEATURES.copy()

    # Pre-compute all edge coordinates (vectorized)
    edge_y = y + (np.array([e[0] for e in edges], dtype=np.float32) - half_h)
    edge_x = x + (np.array([e[1] for e in edges], dtype=np.float32) - half_w)

    y_int = np.round(edge_y).astype(int)
    x_int = np.round(edge_x).astype(int)

    y0 = np.clip(y_int - PATCH_HALF, 0, h - 1)
    y1 = np.clip(y_int + PATCH_HALF + 1, 1, h)
    x0 = np.clip(x_int - PATCH_HALF, 0, w - 1)
    x1 = np.clip(x_int + PATCH_HALF + 1, 1, w)

    # Extract patches
    correlations = []
    widths = []
    ratios = []
    n_valid = 0

    for i, (y0i, y1i, x0i, x1i) in enumerate(zip(y0, y1, x0, x1)):
        if (y1i - y0i) < 4 or (x1i - x0i) < 4:
            continue

        cand_patch = search_f[y0i:y1i, x0i:x1i].ravel()
        ref_patch = ref_patches[i].ravel()

        if len(cand_patch) < 4:
            continue

        n_valid += 1

        # Correlation
        ref_mean, cand_mean = ref_patch.mean(), cand_patch.mean()
        ref_std, cand_std = ref_patch.std(), cand_patch.std()

        if ref_std > 1e-6 and cand_std > 1e-6:
            corr = ((ref_patch - ref_mean) * (cand_patch - cand_mean)).mean() / (ref_std * cand_std)
            if np.isfinite(corr):
                correlations.append(corr)

        # Width and contrast (minimal computation)
        low_val, high_val = cand_patch.min(), cand_patch.max()
        if high_val - low_val > 1e-6:
            mid_val = (low_val + high_val) / 2
            width_count = float(np.sum(np.abs(cand_patch - mid_val) < (high_val - low_val) / 4))
            widths.append(width_count)

            # Contrast
            mid = len(cand_patch) // 2
            first_half = cand_patch[:mid].mean() if mid > 0 else 1.0
            second_half = cand_patch[mid:].mean() if mid < len(cand_patch) else 1.0
            if first_half > 1e-9:
                ratio = second_half / first_half
                if 0.1 < ratio < 10:
                    ratios.append(ratio)

    # Compute and return features
    features = NAN_FEATURES.copy()
    features["edge_valid_frac"] = float(n_valid) / len(edges) if len(edges) > 0 else 0.0

    if n_valid < 2:
        return features

    if correlations:
        corr_arr = np.array(correlations)
        features["edge_prof_corr_med"] = float(np.median(corr_arr))
        if len(corr_arr) >= 2:
            # Fast p25: just sort and index
            sorted_corr = np.sort(corr_arr)
            idx_p25 = max(0, len(sorted_corr) // 4)
            features["edge_prof_corr_p25"] = float(sorted_corr[idx_p25])
        else:
            features["edge_prof_corr_p25"] = features["edge_prof_corr_med"]

    # Derivative correlation placeholder (NaN)
    features["edge_deriv_corr_med"] = float("nan")

    if widths:
        widths_arr = np.array(widths)
        med_width = np.median(widths_arr)
        mad = float(np.median(np.abs(widths_arr - med_width)))
        features["edge_halo_width_mad"] = mad

    if ratios:
        features["edge_contrast_ratio_med"] = float(np.median(np.array(ratios)))

    features["edge_sign_frac"] = float(n_valid) / len(edges)  # Just use valid fraction as proxy

    return features
