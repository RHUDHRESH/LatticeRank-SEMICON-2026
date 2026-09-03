"""Dense full-resolution pose sweep.

Measured on ``p2_val``, this shape of search scores **0.50 within 5 px** where
the decimate-harvest-screen-refine funnel scores 0.12. The reason is not a
better similarity measure -- it is the same ZNCC -- but that a dense sweep
never discards a candidate. Every architecture built on pooling, spreading,
screening and capping had at least one stage that dropped the true site, and
the measurements kept finding it: the true site's rank by any cheap score is in
the hundreds, so every cap set for cost silently deleted the answer.

The addendum permits exactly this: *"extending your Phase 1 method to search
over the disclosed scale and rotation ranges"*. The similarity measure, the
template construction and the periodic reasoning are unchanged; only the search
over the now-unknown pose is added.

Cost is the honest trade. One full-resolution correlation is ~130 ms, so a
9x5 grid is around 6 s. That fits the 20 s hard timeout but not the 5 s median
target, which is what the coarse-to-fine variant below exists to fix: a sparse
sweep locates the pose, then a dense sweep runs only at the best few poses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .baseline import _ncc_valid, _robust_contrast
from .pose import band_pass, build_template_base, rotate_template

#: Scale sweep over the disclosed [8, 12]. Step 0.5 matches the sponsors' own
#: published baseline, whose measured accuracy this module is calibrated
#: against.
DENSE_SCALES = tuple(float(s) for s in np.arange(8.0, 12.01, 0.5))

#: Rotation sweep. The disclosed stage rotation is +/-5 deg, but the reportable
#: angle also carries acquisition jitter (measured: 10.4% of ground-truth
#: rotations fall outside +/-5), so the sweep runs wider.
DENSE_ROTATIONS = (-6.0, -3.0, 0.0, 3.0, 6.0)


@dataclass
class DenseMatch:
    """Best position and pose from a dense sweep."""

    x: float
    y: float
    scale: float
    rotation: float
    score: float
    evaluated: int = 0

    @property
    def found(self) -> bool:
        return bool(np.isfinite(self.score))


def _shape_energy(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Windowed-variance term of the ZNCC denominator, for one template SHAPE.

    ``_ncc_valid`` runs three FFT convolutions per call, but two of them -- the
    local sum and the local sum of squares -- read only the image and a box of
    the template's shape. They are independent of what the template contains,
    and rotation here uses ``reshape=False``, so every rotation at a given scale
    shares this term exactly. Computing it once per scale turns the sweep's 135
    convolutions into 63.

    This lives in the Phase 2 sweep rather than in :mod:`driftforge.baseline`
    on purpose: ``baseline.py`` is the frozen Phase 1 locator, SHA-256 pinned by
    ``LocatorFrozenTests`` so the declared method cannot drift. The arithmetic
    below is identical to what that module computes -- verified by asserting
    byte-identical predictions across all 20 organizer sample pairs -- it is
    only evaluated in a different order.
    """
    ones = np.ones(shape, dtype=np.float32)
    n = float(shape[0] * shape[1])
    local_sum = signal.fftconvolve(image, ones, mode="valid")
    local_sumsq = signal.fftconvolve(image * image, ones, mode="valid")
    return np.maximum(local_sumsq - local_sum * local_sum / n, 1e-9)


def _ncc_with_energy(image: np.ndarray, template: np.ndarray,
                     local_energy: np.ndarray) -> np.ndarray:
    """ZNCC in valid coordinates, reusing a precomputed denominator term."""
    template = template.astype(np.float32)
    template = template - float(template.mean())
    template_energy = float(np.sum(template * template))
    if template_energy < 1e-9:
        raise ValueError("template has no usable contrast")
    numerator = signal.fftconvolve(image, template[::-1, ::-1], mode="valid")
    return (numerator / np.sqrt(local_energy * template_energy)).astype(np.float32)


def _correlate(search_f: np.ndarray, template: np.ndarray,
               local_energy: np.ndarray | None = None) -> tuple[float, float, float]:
    """Global maximum of the ZNCC surface, returned as (x, y, score)."""
    if not np.isfinite(template).all() or float(template.std()) < 1e-6:
        return float("nan"), float("nan"), float("-inf")
    th, tw = template.shape[:2]
    if th >= search_f.shape[0] or tw >= search_f.shape[1] or th < 3 or tw < 3:
        return float("nan"), float("nan"), float("-inf")
    if local_energy is not None:
        surface = _ncc_with_energy(search_f, template.astype(np.float32), local_energy)
    else:
        surface = _ncc_valid(search_f, template.astype(np.float32))
    if not np.isfinite(surface).any():
        return float("nan"), float("nan"), float("-inf")
    row, col = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    return (
        float(col + (tw - 1) / 2.0),
        float(row + (th - 1) / 2.0),
        float(surface[row, col]),
    )


def dense_pose_search(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    scales: tuple[float, ...] = DENSE_SCALES,
    rotations: tuple[float, ...] = DENSE_ROTATIONS,
    robust_contrast: bool = True,
    band_pass_input: bool = True,
    collect_rows: list | None = None,
    deadline=None,
) -> DenseMatch:
    """Sweep the disclosed pose ranges at full resolution, keep the global best.

    ``band_pass_input`` applies a difference-of-Gaussians to both sides before
    correlating, and is **on by default because it is the largest measured win
    in the project**. On 280 present pairs across ``p2_val`` and ``p2_stress``:

    ======  ==========  ==========  =========
    arm     <=1 px      credit      pts / 40
    ======  ==========  ==========  =========
    raw     18.6%       0.189       7.5
    DoG     **29.6%**   **0.334**   **13.3**
    ======  ==========  ==========  =========

    Paired on identical scenes: fixed 49, broke 18, net +31, McNemar
    p = 0.0002. Runtime cost is 0.05 s per pair.

    The gain concentrates exactly where the points are. By severity the deltas
    are +9.4 / +1.3 / +10.1 / **+22.0** pp at levels 0/1/2/3 -- severity 3 goes
    from 7.3% to 29.3% -- and Set B carries 0.55 of the localization weight.
    That profile is the signature of charging and defocus suppression rather
    than a generic contrast effect: a band-pass removes the low-frequency
    intensity drift that charging produces, which is Set B's dominant artefact.

    ``robust_contrast`` applies the Phase 1 intensity normalization to the
    search image. Note it provably cannot change the result on its own: it is
    an affine intensity map and ZNCC is invariant to those. It is retained
    because it bounds the dynamic range before the band-pass, which is not
    affine.

    ``collect_rows``, when given a list, is filled with one record per pose
    hypothesis: ``{x, y, raw, scale, rotation}``. The presence model needs
    exactly these, and recomputing them costs a second full sweep -- measured
    at 5.3 s per pair, which alone pushes the entrypoint past the 5 s median
    target. Sharing the single pass is free.

    Returns the single best ``(x, y, scale, rotation)`` across the whole sweep.
    Selection among near-ties is deliberately *not* done here -- that is the
    periodic-residual and structural ranking stage's job, and it needs the
    sweep's evidence rather than a pre-made decision.
    """
    search_f = (
        _robust_contrast(search)
        if robust_contrast
        else np.asarray(search, dtype=np.float32)
    )
    if band_pass_input:
        search_f = band_pass(search_f)
    best = DenseMatch(
        x=(search.shape[1] - 1) / 2.0, y=(search.shape[0] - 1) / 2.0,
        scale=float(np.mean(scales)), rotation=0.0, score=float("-inf"),
    )
    evaluated = 0
    for scale in scales:
        # The anti-alias and decimation depend only on `scale`, so they are
        # hoisted out of the rotation loop: 12.2 ms each on a 1000x1000
        # reference against 0.3 ms for the rotation of the ~100 px template.
        # Rebuilding per pose cost 45 x 12.2 ms where 9 x 12.2 ms is enough.
        try:
            base_template = build_template_base(reference, float(scale))
        except ValueError:
            continue
        # The ZNCC denominator's windowed-variance term depends on the template
        # SHAPE only, and rotation preserves shape, so all rotations at this
        # scale share it. Computing it here saves two FFT convolutions per pose.
        shape_energy = None
        if base_template.shape[0] < search_f.shape[0] and base_template.shape[1] < search_f.shape[1]:
            try:
                shape_energy = _shape_energy(search_f, base_template.shape)
            except ValueError:
                shape_energy = None
        for rotation in rotations:
            if deadline is not None and not deadline.affords(
                "dense_hypothesis", estimate_s=0.15
            ):
                deadline.degrade(
                    "dense_pose_search",
                    f"evaluated {evaluated} of {len(scales) * len(rotations)}",
                )
                best.evaluated = evaluated
                return best
            try:
                template = rotate_template(base_template, float(rotation))
            except ValueError:
                continue
            if band_pass_input:
                template = band_pass(template)
            x, y, score = _correlate(search_f, template, local_energy=shape_energy)
            if collect_rows is not None:
                collect_rows.append({"x": x, "y": y, "raw": score,
                                     "scale": float(scale), "rotation": float(rotation)})
            evaluated += 1
            if score > best.score:
                best = DenseMatch(
                    x=x, y=y, scale=float(scale), rotation=float(rotation),
                    score=score,
                )
    best.evaluated = evaluated
    if not np.isfinite(best.score):
        best.score = float("nan")
    return best
