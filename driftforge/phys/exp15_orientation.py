"""Gradient-orientation transport: circular-statistics correspondence test.

Experiment #15: Validate a candidate by testing whether its local orientation
field is a coherent transported version of the reference's orientation field.

PHYSICS
-------
The reference and search images capture the same physical features under a known
transformation (scale, rotation). The gradient orientations should match after
compensating for that known rotation. True sites show tight orientation agreement
across multiple pixels; aliases will scatter or reverse because they represent
different material.

The orientation field is AXIAL (mod pi, not mod 2*pi) because SEM edges are
invariant under 180-degree rotation: a bright edge looks the same as a dark
edge in orientation space.

METHOD
------
1. In build(ctx):
   - Compute reference gradient orientations: theta_r = atan2(gy, gx)
   - Compute reference gradient magnitudes: m_r = sqrt(gx^2 + gy^2)
   - Precompute on heavily subsampled grid (every 4th pixel) for speed
   - Precompute the known pose rotation from ctx.rotation (degrees -> radians)

2. In score(state, x, y):
   - Extract the candidate patch at (x, y)
   - Compute candidate gradient orientations theta_s and magnitudes m_s
     (also on subsampled grid to match reference sampling)
   - For each pixel pair (reference and candidate):
     * Compute orientation error: e_j = wrap_pi(theta_s_j - theta_r_j - theta_pose)
     * Weight by: w_j = min(|m_r_j|, |m_s_j|)
   - Compute circular statistics:
     * Circular mean (resultant) and its magnitude
     * Circular median and MAD
     * Fraction of inliers at various thresholds
     * Entropy and spatial structure metrics

RUNTIME STRATEGY
  Heavy subsampling (stride=4) reduces gradient computation by 16x, keeping
  enough spatial coverage (~20 pixels per patch) to measure orientation fields.
  Precomputation in build() eliminates redundant reference-side work.

DOCSTRING CONVENTIONS
  Angles are in radians unless explicitly noted. Wrapped to [-pi, pi).
  Orientation is AXIAL, so wrap mod pi: wrap_pi(x) = atan2(sin(2*x), cos(2*x)) / 2
  Magnitudes are uint8->float when reading raw, float32 when using robust contrast.
  Coordinates are always search-image pixels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURES = [
    "orientation_error_med",      # Circular median of orientation errors
    "orientation_error_mad",      # Circular MAD of orientation errors
    "orientation_inlier_05deg",   # Fraction of errors <= 5 degrees
    "orientation_inlier_10deg",   # Fraction of errors <= 10 degrees
    "orientation_inlier_20deg",   # Fraction of errors <= 20 degrees
    "orientation_resultant",      # Magnitude of circular resultant
    "orientation_entropy",        # Shannon entropy of orientation distribution
    "orientation_x_structure",    # Anisotropy in x vs y gradient agreement
    "orientation_y_structure",    # Anisotropy in y vs x gradient agreement
]

NAN_FEATURES = {k: float("nan") for k in FEATURES}

# Runtime target: <200 us/candidate
# Heavy subsampling to stay fast - stride=8 gives ~64 samples per patch
GRAD_STRIDE = 8  # Sample every 8th pixel in the patch


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi) range. x can be any value."""
    # atan2(sin(x), cos(x)) always returns [-pi, pi)
    return np.arctan2(np.sin(x), np.cos(x))


def _axial_wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap AXIAL angle to [-pi/2, pi/2) range using doubled-angle trick.

    Orientation is modulo pi, not 2*pi. We compute in doubled-angle space:
    wrap_pi(2*x), then divide by 2. This ensures:
      - atan2(sin(2*x), cos(2*x)) / 2 wraps to [-pi/2, pi/2)
      - Which is equivalent to mod pi with proper branch cuts

    Why this matters: getting axial wrap wrong (using 2*pi instead of pi) silently
    halves the signal. An orientation mismatch of 45 degrees appears to be a
    mismatch of 90 degrees, which biases inlier counts and entropy metrics.
    """
    # Double the angle, wrap to [-pi, pi), then halve
    doubled = 2.0 * x
    wrapped = np.arctan2(np.sin(doubled), np.cos(doubled))
    return wrapped / 2.0


def _circular_mean_and_mag(angles: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Compute circular mean and magnitude of resultant.

    For axial data (angles mod pi), we work in doubled-angle space to preserve
    the modulo-pi property.

    Returns (mean_angle, resultant_magnitude).
    """
    if len(angles) == 0:
        return float("nan"), float("nan")

    if weights is None:
        weights = np.ones_like(angles)

    # Mask out invalid angles
    valid = np.isfinite(angles) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan"), float("nan")

    angles = angles[valid]
    weights = weights[valid]

    if len(angles) == 0:
        return float("nan"), float("nan")

    # Work in doubled-angle space to preserve axial property
    doubled = 2.0 * angles

    # Circular sum: R = sum(w_j * exp(i * 2 * theta_j))
    real = np.sum(weights * np.cos(doubled))
    imag = np.sum(weights * np.sin(doubled))

    # Magnitude of resultant
    r_mag = np.sqrt(real**2 + imag**2) / np.sum(weights)

    # Mean angle (halved back from doubled space)
    mean_angle = np.arctan2(imag, real) / 2.0

    return float(mean_angle), float(r_mag)


def _circular_median(angles: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Compute circular median and MAD using axial statistics.

    Circular median is the angle that minimizes the sum of circular distances.
    MAD is the median absolute circular deviation.

    Returns (median_angle, MAD).
    """
    if len(angles) == 0:
        return float("nan"), float("nan")

    if weights is None:
        weights = np.ones_like(angles)

    # Mask out invalid values
    valid = np.isfinite(angles) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan"), float("nan")

    angles = angles[valid]
    weights = weights[valid]

    if len(angles) == 0:
        return float("nan"), float("nan")

    # For axial data, we use doubled-angle space
    doubled = 2.0 * angles

    # Sort angles
    idx = np.argsort(doubled)
    doubled_sorted = doubled[idx]
    weights_sorted = weights[idx]

    # Find median: cumulative sum
    cumsum = np.cumsum(weights_sorted)
    total_weight = cumsum[-1]
    median_idx = np.searchsorted(cumsum, total_weight / 2.0)
    median_idx = min(median_idx, len(doubled_sorted) - 1)
    median_angle = doubled_sorted[median_idx] / 2.0

    # MAD: median absolute deviation from the median
    # Compute circular distances from median
    diffs = _axial_wrap_pi(angles - median_angle)
    abs_diffs = np.abs(diffs)

    # Find median of absolute deviations
    idx_abs = np.argsort(abs_diffs)
    abs_diffs_sorted = abs_diffs[idx_abs]
    weights_sorted_abs = weights[idx_abs]
    cumsum_abs = np.cumsum(weights_sorted_abs)
    mad_idx = np.searchsorted(cumsum_abs, cumsum_abs[-1] / 2.0)
    mad_idx = min(mad_idx, len(abs_diffs_sorted) - 1)
    mad = float(abs_diffs_sorted[mad_idx])

    return float(median_angle), mad


def build(ctx) -> dict | None:
    """Precompute reference orientations and gradient magnitudes.

    Returns state dict with reference gradients, or None if patch is too small
    or gradients are too weak to measure.

    Precomputation is done on ctx.template (the pose-correct reference), which
    is expensive, but only happens once per pair. score() only samples the
    search-side candidate.
    """
    # Compute reference gradients on the template using simple finite differences
    template_f = ctx.template.astype(np.float32)

    # Use centered finite differences (faster than Sobel)
    ref_gy = (template_f[2:, 1:-1] - template_f[:-2, 1:-1]) / 2.0
    ref_gx = (template_f[1:-1, 2:] - template_f[1:-1, :-2]) / 2.0

    # Trim to match dimensions
    ref_gy = ref_gy[1:-1, :]
    ref_gx = ref_gx[:, 1:-1]

    min_h, min_w = min(ref_gy.shape[0], ref_gx.shape[0]), min(ref_gy.shape[1], ref_gx.shape[1])
    ref_gy = ref_gy[:min_h, :min_w]
    ref_gx = ref_gx[:min_h, :min_w]

    ref_mag = np.hypot(ref_gx, ref_gy).astype(np.float32)

    # Compute reference orientations (atan2 gives [-pi, pi))
    ref_theta = np.arctan2(ref_gy, ref_gx).astype(np.float32)

    # Subsample heavily to stay fast (only use every 8th pixel for VERY fast sampling)
    ref_theta_sub = ref_theta[::GRAD_STRIDE, ::GRAD_STRIDE].astype(np.float32)
    ref_mag_sub = ref_mag[::GRAD_STRIDE, ::GRAD_STRIDE].astype(np.float32)

    # Check if we have enough strong gradients
    if ref_mag_sub.size < 4:
        return None  # Patch too small

    strong_enough = ref_mag_sub > 0.05 * ref_mag_sub.max()
    if np.sum(strong_enough) < 4:
        return None  # Abstain: too few strong gradients

    # Convert pose rotation from degrees to radians
    pose_rotation = np.deg2rad(ctx.rotation)

    return {
        "ref_theta_sub": ref_theta_sub,  # Subsampled reference orientations
        "ref_mag_sub": ref_mag_sub,      # Subsampled reference magnitudes
        "pose_rotation": pose_rotation,  # Known rotation to compensate for
        "search_f": ctx.search_f,
        "template_shape": ctx.template.shape,
        "half_w": ctx.half_w,
        "half_h": ctx.half_h,
    }


def score(state: dict, x: float, y: float) -> dict[str, float]:
    """Score a candidate by orientation field correspondence.

    Measures how well the candidate's orientation field matches the reference's
    after compensating for known pose rotation. Uses circular statistics.

    This function does NOT recompute reference gradients; it uses the precomputed
    subsampled grids from build(). This keeps per-candidate cost low.
    """
    if state is None:
        return NAN_FEATURES.copy()

    ref_theta_sub = state.get("ref_theta_sub")
    ref_mag_sub = state.get("ref_mag_sub")
    pose_rotation = state.get("pose_rotation")
    search_f = state.get("search_f")
    template_shape = state.get("template_shape")
    half_w = state.get("half_w")
    half_h = state.get("half_h")

    if any(v is None for v in [ref_theta_sub, ref_mag_sub, pose_rotation, search_f]):
        return NAN_FEATURES.copy()

    # Extract candidate patch
    patch_h, patch_w = template_shape
    top = int(round(y)) - patch_h // 2
    left = int(round(x)) - patch_w // 2

    # Bounds check
    if top < 0 or left < 0 or top + patch_h > search_f.shape[0] or left + patch_w > search_f.shape[1]:
        return NAN_FEATURES.copy()

    candidate = search_f[top:top + patch_h, left:left + patch_w].astype(np.float32)

    # Compute candidate gradients on a coarser pre-sampled grid (every 4 pixels)
    # to reduce gradient computation cost
    cand_sub = candidate[::4, ::4]
    if cand_sub.shape[0] < 3 or cand_sub.shape[1] < 3:
        return NAN_FEATURES.copy()  # Patch too small

    # Use simple centered finite differences (faster than Sobel)
    cand_gy = (cand_sub[2:, 1:-1] - cand_sub[:-2, 1:-1]) / 2.0
    cand_gx = (cand_sub[1:-1, 2:] - cand_sub[1:-1, :-2]) / 2.0

    # Trim to match dimensions after finite differences
    cand_gy = cand_gy[1:-1, :]  # Now (h-4, w-2)
    cand_gx = cand_gx[:, 1:-1]  # Now (h-2, w-4)

    # Ensure same shape for magnitude computation
    min_h, min_w = min(cand_gy.shape[0], cand_gx.shape[0]), min(cand_gy.shape[1], cand_gx.shape[1])
    if min_h < 1 or min_w < 1:
        return NAN_FEATURES.copy()

    cand_gy = cand_gy[:min_h, :min_w]
    cand_gx = cand_gx[:min_h, :min_w]

    cand_mag = np.hypot(cand_gx, cand_gy).astype(np.float32)
    cand_theta = np.arctan2(cand_gy, cand_gx).astype(np.float32)

    # Subsample candidate to match reference sampling (every other pixel of the finite-diff result)
    cand_theta_sub = cand_theta[::2, ::2].astype(np.float32)
    cand_mag_sub = cand_mag[::2, ::2].astype(np.float32)

    # Ensure shapes match (handle edge cases where subsampling gives different sizes)
    min_h = min(ref_theta_sub.shape[0], cand_theta_sub.shape[0])
    min_w = min(ref_theta_sub.shape[1], cand_theta_sub.shape[1])

    if min_h < 2 or min_w < 2:
        # Patch too small after subsampling
        return NAN_FEATURES.copy()

    ref_theta_flat = ref_theta_sub[:min_h, :min_w].flatten()
    ref_mag_flat = ref_mag_sub[:min_h, :min_w].flatten()
    cand_theta_flat = cand_theta_sub[:min_h, :min_w].flatten()
    cand_mag_flat = cand_mag_sub[:min_h, :min_w].flatten()

    # Compute orientation errors: e_j = wrap_pi(theta_s_j - theta_r_j - theta_pose)
    # Using axial wrap (mod pi) because orientation is symmetric
    orientation_error = cand_theta_flat - ref_theta_flat - pose_rotation
    orientation_error = _axial_wrap_pi(orientation_error)

    # Weight by minimum magnitude (both must have gradient to vote)
    weights = np.minimum(np.abs(ref_mag_flat), np.abs(cand_mag_flat))

    # Mask out low-magnitude pixels
    min_mag = 0.05 * max(np.max(np.abs(ref_mag_flat)), np.max(np.abs(cand_mag_flat)))
    valid = weights > min_mag

    if np.sum(valid) < 4:
        # Not enough valid observations
        return NAN_FEATURES.copy()

    errors = orientation_error[valid]
    weights_valid = weights[valid]

    # Normalize weights
    weights_valid = weights_valid / np.sum(weights_valid)

    # Feature 1: Circular median of orientation errors
    med_error, mad_error = _circular_median(errors, weights_valid)

    # Feature 2: Circular MAD (already computed above)
    # mad_error computed above

    # Feature 3-5: Inlier counts at various thresholds
    deg_05 = np.deg2rad(5.0)
    deg_10 = np.deg2rad(10.0)
    deg_20 = np.deg2rad(20.0)

    inlier_05 = float(np.sum(weights_valid[np.abs(errors) <= deg_05]))
    inlier_10 = float(np.sum(weights_valid[np.abs(errors) <= deg_10]))
    inlier_20 = float(np.sum(weights_valid[np.abs(errors) <= deg_20]))

    # Feature 6: Magnitude of circular resultant
    _, resultant = _circular_mean_and_mag(errors, weights_valid)

    # Feature 7: Entropy of orientation distribution
    # Simple entropy: variance in the weighted angles
    # This is much faster than histogram-based entropy
    angle_var = np.sum(weights_valid * (errors - med_error) ** 2)
    entropy = float(angle_var) if np.isfinite(angle_var) else float("nan")

    # Feature 8-9: Spatial structure metrics
    # Simplified: compute based on the agreement pattern without separate circular stats
    # Measure anisotropy by looking at the spread along x vs y directions
    ref_theta_valid = ref_theta_flat[valid]

    # Separate errors by whether they come from strong x vs y components
    # using the doubled angle representation for consistency with axial property
    doubled_theta = 2.0 * ref_theta_valid
    cos_theta = np.cos(doubled_theta)
    sin_theta = np.sin(doubled_theta)

    gx_strong = np.abs(cos_theta) > 0.3
    gy_strong = np.abs(sin_theta) > 0.3

    # Simplified structure metrics: just use the resultant for the subset
    if np.sum(gx_strong) >= 2:
        x_errors = errors[gx_strong]
        x_weights = weights_valid[gx_strong]
        _, x_structure = _circular_mean_and_mag(x_errors, x_weights)
        x_structure = float(x_structure) if np.isfinite(x_structure) else 0.0
    else:
        x_structure = float("nan")

    if np.sum(gy_strong) >= 2:
        y_errors = errors[gy_strong]
        y_weights = weights_valid[gy_strong]
        _, y_structure = _circular_mean_and_mag(y_errors, y_weights)
        y_structure = float(y_structure) if np.isfinite(y_structure) else 0.0
    else:
        y_structure = float("nan")

    features = {
        "orientation_error_med": float(med_error) if np.isfinite(med_error) else float("nan"),
        "orientation_error_mad": float(mad_error) if np.isfinite(mad_error) else float("nan"),
        "orientation_inlier_05deg": inlier_05,
        "orientation_inlier_10deg": inlier_10,
        "orientation_inlier_20deg": inlier_20,
        "orientation_resultant": float(resultant) if np.isfinite(resultant) else float("nan"),
        "orientation_entropy": float(entropy) if np.isfinite(entropy) else float("nan"),
        "orientation_x_structure": x_structure,
        "orientation_y_structure": y_structure,
    }

    return features
