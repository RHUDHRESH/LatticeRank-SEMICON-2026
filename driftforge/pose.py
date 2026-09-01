"""Pose conventions and brute-force verification oracles for Phase 2.

**Scale convention (§2.4).** Everywhere in this module, ``scale`` is the
*down-scaling factor* ``s`` between the pair: the reference covers
``WORLD_FOV_NM / s`` nanometres at the same pixel count as the search, so the
template is produced with ``zoom = 1/s``. This is deliberately NOT the
convention of ``baseline._template_from_reference``, where ``scale`` is a
multiplier on a hard-coded 0.1 zoom (there, ``zoom = 0.1 * scale``; feed it
``scale = 10 / s`` to get the same template size). Any new code that builds
templates must state its convention in its docstring and be covered by a
finiteness assertion.

The oracles below exist because the rotation label depends on the sign
conventions of ``sem._affine_warp``, ``ndimage.rotate`` and the template
builder *simultaneously*; the label is whatever these brute-force searches
recover, and ``scripts/verify_conventions.py`` asserts the generator's label
matches.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def build_template(
    reference: np.ndarray,
    scale: float,
    rotation_deg: float,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Down-scale a reference to search scale and rotate it.

    ``scale`` is the down-scaling factor ``s`` (reference FOV = world FOV / s
    at equal pixel counts), so the template side is about ``1000 / s`` px and
    the internal zoom is ``1 / s``. ``rotation_deg`` rotates the template with
    ``ndimage.rotate`` (CCW in the array plane, reshape=False).

    ``output_shape`` pins the template size across candidates. Without it the
    size snaps to integers and the ZNCC-vs-scale curve becomes a sawtooth
    (each discrete shape is a slightly different correlation problem), which
    makes brute-force scale readouts unstable; the oracles always pin it.

    Returns a float32 template in [0, 1]. Callers must check finiteness: a
    degenerate reference yields an all-constant template, which makes any
    normalized correlation NaN - the silent-failure mode called out in §2.4.
    """
    if scale <= 1.0:
        raise ValueError("scale is a down-scaling factor and must exceed 1.0")
    image = reference.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    image /= 255.0 if image.max() > 1.5 else 1.0
    # Anti-alias for decimation by `scale`: Gaussian sigma ~ 0.35 * s
    # approximates the box filter of width s (Phase 1 used 0.4 * 10).
    sigma = float(np.clip(0.35 * scale, 0.8, 5.0))
    antialiased = ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")
    if output_shape is None:
        base = ndimage.zoom(antialiased, zoom=1.0 / scale, order=1, prefilter=False)
    else:
        oh, ow = output_shape
        center_in = (np.array(antialiased.shape, dtype=np.float64) - 1.0) / 2.0
        center_out = (np.array(output_shape, dtype=np.float64) - 1.0) / 2.0
        matrix = np.array([scale, scale], dtype=np.float64)
        offset = center_in - matrix * center_out
        base = ndimage.affine_transform(
            antialiased, matrix, offset=offset, output_shape=(oh, ow),
            order=1, mode="reflect", prefilter=False,
        )
    if abs(rotation_deg) > 1e-8:
        base = ndimage.rotate(
            base, rotation_deg, reshape=False, order=1, mode="reflect", prefilter=False
        )
    return base.astype(np.float32)


def _window(image: np.ndarray, cx: float, cy: float, height: int, width: int) -> np.ndarray:
    """Extract the window centred on (cx, cy), reflect-padded at borders."""
    half_h, half_w = height // 2, width // 2
    top = int(round(cy)) - half_h
    left = int(round(cx)) - half_w
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, top + height - image.shape[0])
    pad_right = max(0, left + width - image.shape[1])
    if pad_top or pad_left or pad_bottom or pad_right:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    return image[top + pad_top : top + pad_top + height, left + pad_left : left + pad_left + width]


def zncc_at(search: np.ndarray, template: np.ndarray, cx: float, cy: float) -> float:
    """Zero-mean normalized cross-correlation of the template at one site."""
    template = template.astype(np.float64)
    template -= template.mean()
    energy = float(np.sum(template * template))
    if energy < 1e-9:
        return float("nan")
    window = _window(search, cx, cy, template.shape[0], template.shape[1]).astype(np.float64)
    if window.shape != template.shape:
        return float("nan")
    window -= window.mean()
    denom = float(np.sqrt(np.sum(window * window) * energy))
    if denom < 1e-9:
        return float("nan")
    return float(np.sum(template * window) / denom)


def band_pass(image: np.ndarray, sigma_lo: float = 2.0, sigma_hi: float = 8.0) -> np.ndarray:
    """Difference-of-Gaussians band pass in search-pixel units.

    The oracles read pose from this structure response rather than raw
    intensities: on severely degraded pairs the raw ZNCC rotation curve can
    be multi-modal (flat double peaks half a degree apart), while the
    band-passed response is unimodal and sharper. Both the generator's label
    measurement and the verification oracles use it, so they read the same
    convention.
    """
    data = image.astype(np.float32)
    if data.ndim == 3:
        data = data.mean(axis=-1)
    return (
        ndimage.gaussian_filter(data, sigma_lo, mode="reflect")
        - ndimage.gaussian_filter(data, sigma_hi, mode="reflect")
    ).astype(np.float32)


def _refine_parabolic(p0: tuple, p1: tuple, p2: tuple) -> float:
    """Vertex of a parabola through three (x, y) points, clipped to the span."""
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(x1)
    shift = 0.5 * (y0 - y2) / denom * (x2 - x0)
    return float(np.clip(x1 + shift, min(x0, x2), max(x0, x2)))


def rotation_oracle(
    reference: np.ndarray,
    search: np.ndarray,
    x: float,
    y: float,
    scale: float,
    lo: float = -8.5,
    hi: float = 8.5,
    step: float = 0.1,
) -> tuple[float, float]:
    """Brute-force the template rotation that maximizes ZNCC at (x, y).

    The search range must cover the full net-rotation span: stage theta
    (+/-5 deg) plus the reference acquisition jitter (+/-2.2 deg) plus the
    search jitter (+/-0.35 deg) can reach +/-7.55 deg. The response is the
    band-passed structure ZNCC (see :func:`band_pass`).
    """
    search_f = band_pass(search)
    shape = tuple(int(round(dim / scale)) for dim in search.shape[:2])
    base = band_pass(build_template(reference, scale, 0.0, output_shape=shape))
    grid = np.arange(lo, hi + step / 2.0, step)
    scores = np.array(
        [
            zncc_at(
                search_f,
                ndimage.rotate(base, float(t), reshape=False, order=1, mode="reflect", prefilter=False),
                x,
                y,
            )
            for t in grid
        ]
    )
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan"), float("nan")
    best = int(np.nanargmax(scores))
    lo_i, hi_i = max(best - 1, 0), min(best + 1, len(grid) - 1)
    if hi_i - lo_i == 2:
        theta = _refine_parabolic(
            (grid[lo_i], scores[lo_i]), (grid[best], scores[best]), (grid[hi_i], scores[hi_i])
        )
    else:
        theta = float(grid[best])
    return float(theta), float(scores[best])


def scale_oracle(
    reference: np.ndarray,
    search: np.ndarray,
    x: float,
    y: float,
    rotation_deg: float,
    lo: float = 7.5,
    hi: float = 12.5,
    coarse_step: float = 0.1,
    fine_step: float = 0.01,
    shape_scale: float | None = None,
) -> tuple[float, float]:
    """Brute-force the down-scaling factor that maximizes ZNCC at (x, y).

    Two-stage: a coarse pass over [lo, hi] locates the peak, a fine pass
    refines it to ``fine_step``. G1's tolerance is 0.5% of s. The response is
    the band-passed structure ZNCC (see :func:`band_pass`). The template
    output shape is pinned (at ``shape_scale``, default the window centre) so
    the curve is smooth in s instead of sawtoothed by integer size snaps;
    pinning only fixes the correlation window size, the scale search itself
    stays independent.

    Throughput: the reference anti-alias pass runs once per oracle call at
    the pinned centre scale instead of once per candidate. Within a pass the
    candidate sigmas span only +/- 0.35*coarse_step around that centre - a
    sub-percent blur difference that shifts correlation values but not the
    peak location the oracle reports.
    """
    search_f = band_pass(search)
    pinned = float(shape_scale) if shape_scale is not None else 0.5 * (lo + hi)
    shape = tuple(int(round(dim / pinned)) for dim in search.shape[:2])
    image = reference.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    image /= 255.0 if image.max() > 1.5 else 1.0
    prepared = ndimage.gaussian_filter(
        image, sigma=float(np.clip(0.35 * pinned, 0.8, 5.0)), mode="reflect"
    )

    def template_at(scale: float) -> np.ndarray:
        center_in = (np.array(prepared.shape, dtype=np.float64) - 1.0) / 2.0
        center_out = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
        matrix = np.array([scale, scale], dtype=np.float64)
        offset = center_in - matrix * center_out
        base = ndimage.affine_transform(
            prepared, matrix, offset=offset, output_shape=shape,
            order=1, mode="reflect", prefilter=False,
        )
        if abs(rotation_deg) > 1e-8:
            base = ndimage.rotate(
                base, rotation_deg, reshape=False, order=1, mode="reflect", prefilter=False
            )
        return base.astype(np.float32)

    coarse = np.arange(lo, hi + coarse_step / 2.0, coarse_step)
    scores = np.array(
        [zncc_at(search_f, band_pass(template_at(float(s))), x, y) for s in coarse]
    )
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan"), float("nan")
    best = int(np.nanargmax(scores))
    fine = np.arange(coarse[best] - coarse_step, coarse[best] + coarse_step + fine_step / 2.0, fine_step)
    fine_scores = np.array(
        [
            zncc_at(search_f, band_pass(template_at(float(s))), x, y)
            for s in fine
        ]
    )
    best_f = int(np.nanargmax(fine_scores))
    lo_i, hi_i = max(best_f - 1, 0), min(best_f + 1, len(fine) - 1)
    if hi_i - lo_i == 2:
        s = _refine_parabolic(
            (fine[lo_i], fine_scores[lo_i]), (fine[best_f], fine_scores[best_f]), (fine[hi_i], fine_scores[hi_i])
        )
    else:
        s = float(fine[best_f])
    return float(s), float(fine_scores[best_f])
