"""Full-resolution refinement of a coarse pose candidate.

The coarse stage (:mod:`driftforge.pose_search`) answers *where, roughly*: on
``p2_val`` it puts the true site within 30 px of some candidate on 100% of
present pairs, but within 5 px on only 70%, and its scale label is within 5% of
truth on only about a third. Both limits are structural, not bugs -- a search
decimated by ``d`` cannot localize better than ``d/2`` px, and a scale grid
stepping by 1.0 in ``s`` cannot do better than +/-5%.

Refinement is what converts that neighbourhood into credit. The scoring is
unforgiving about the difference: localization pays 1.00 at <=1 px and 0.40 at
<=5 px, and pose pays nothing at all beyond 5% scale or 1 degree rotation, so a
candidate that is "nearly right" everywhere scores a fraction of one that is
refined.

The loop alternates the three parameter groups rather than solving them
jointly, because they are separable to first order and each has a proven
single-site estimator already in :mod:`driftforge.pose`:

1. **translation** at the current pose, by local correlation with a subpixel
   peak fit -- this module;
2. **scale** at the refined position, via :func:`driftforge.pose.scale_oracle`;
3. **rotation** at the refined position, via
   :func:`driftforge.pose.rotation_oracle`.

Order matters. Scale error masquerades as translation error -- a 1% scale
error displaces template features by roughly ``0.005 * side`` px at the edges,
which is about 0.5 px on a 100 px template -- so translation is re-estimated
after the pose updates rather than before them only.

The oracles are the same code the generator uses to *measure* ground truth, so
this stage reads pose in exactly the convention the labels were written in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baseline import _ncc_valid
from .pose import _refine_parabolic, band_pass, build_template, rotation_oracle, scale_oracle

#: Half-width of the local translation search, in full-resolution pixels. The
#: coarse stage's worst observed miss is 30 px, so a smaller window would
#: silently fail to reach the true site on the hardest pairs.
TRANSLATION_RADIUS_PX = 32

#: Scale half-window for the oracle, around the coarse estimate. The coarse
#: grid steps by 1.0, so +/-1.0 covers the worst grid-snap error with margin.
SCALE_HALF_WINDOW = 1.0

#: Rotation half-window, in degrees, around the coarse estimate. The coarse
#: grid steps by 2.5 deg.
ROTATION_HALF_WINDOW_DEG = 3.0

#: Alternating passes. Two was chosen when the pose sweep and the evaluation
#: data shared a grid, where the coarse hit already sits close to the optimum.
#: Off-grid poses start further away: the disclosed zoom is continuous but our
#: sweep steps by 0.5, so a true z of 9.37 is up to 0.25 from any pose we
#: evaluate and the whole residual falls to refinement. Measured end-to-end on
#: two independent sets -- 70 present pairs of our own p2_val and 40 pairs drawn
#: off-grid from the organizers' published generator -- a third pass fixes 8
#: pairs and breaks 1 (exact McNemar p = 0.039 pooled), worth about +0.7 points
#: on our corpus and +2.1 on the off-grid one, for +0.2 s per pair.
#:
#: Held to three rather than more on purpose: four measured *worse* than three
#: on both sets and five landed between them, so the profile is not monotone and
#: the apparent gain past three is noise. Refinement runs at the site already
#: chosen by the sweep, so this can only move sub-pixel pose -- it cannot change
#: which site is reported.
DEFAULT_PASSES = 3

#: Reportable pose bounds, which are deliberately **wider than the disclosed
#: ranges**. The addendum discloses a stage rotation of +/-5 deg and a zoom in
#: [8, 12], but it defines ``theta`` as the rotation of the reference *as it
#: appears in the search image* -- which is the stage rotation plus the
#: acquisition jitter of both captures. Measured on ``p2_val``, 10.4% of
#: ground-truth rotations fall outside +/-5 deg (range -7.05 to 7.38) and some
#: scales fall outside [8, 12] (7.80 to 12.14).
#:
#: Clamping to the disclosed range is therefore a trap: it is nearly free when
#: the truth is inside the band and catastrophic when it is not. Reporting 5.2
#: against a truth of 5.0 loses one credit tier; clamping 6.7 to 5.0 loses the
#: pose entirely. The search range comes from
#: :func:`driftforge.pose.rotation_oracle`'s own default, justified there as
#: stage +/-5 plus reference jitter +/-2.2 plus margin.
ROTATION_LIMIT_DEG = 8.5
SCALE_LIMITS = (7.5, 12.5)

#: Correlation margin for the evidence-equivalent tie rule during translation
#: refinement. Peaks within this of the maximum are treated as indistinguishable
#: evidence, and the one nearest the incoming estimate wins.
TRANSLATION_EQUIVALENCE_MARGIN = 0.02


@dataclass
class RefinedPose:
    """A candidate after full-resolution refinement."""

    x: float
    y: float
    scale: float
    rotation: float
    score: float
    converged: bool = False
    passes: int = 0
    history: list[tuple[float, float, float, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "scale": float(self.scale),
            "rotation": float(self.rotation),
            "score": float(self.score),
            "converged": bool(self.converged),
            "passes": int(self.passes),
        }


def _subpixel_peak(
    surface: np.ndarray,
    prefer: tuple[float, float] | None = None,
    margin: float = TRANSLATION_EQUIVALENCE_MARGIN,
) -> tuple[float, float, float]:
    """Locate the peak of a correlation surface to subpixel accuracy.

    Returns ``(row, col, value)`` in surface coordinates. The parabolic fit is
    applied per axis around the integer maximum; a peak on the border cannot be
    fitted and is returned at its integer position, which is also the signal
    that the search window was too small.

    ``prefer`` applies the evidence-equivalent tie rule: on a periodic field
    the window contains many lattice copies whose correlations differ by less
    than the noise, and taking the global argmax lets refinement slide a whole
    period to a neighbouring copy. Observed on ``p2_val-000002``, that turned a
    4.2 px starting error into 40.4 px -- a correct candidate refined into a
    wrong one. Among peaks within ``margin`` of the maximum, the one nearest
    ``prefer`` wins, which is the same rule
    :func:`driftforge.pipeline.select_equivalent_candidate` applies at the
    selection stage.
    """
    if surface.size == 0 or not np.isfinite(surface).any():
        return 0.0, 0.0, float("nan")
    flat = int(np.nanargmax(surface))
    row, col = np.unravel_index(flat, surface.shape)
    peak = float(surface[row, col])

    if prefer is not None:
        near = np.argwhere(np.nan_to_num(surface, nan=-np.inf) >= peak - margin)
        if len(near) > 1:
            distances = np.hypot(near[:, 0] - prefer[0], near[:, 1] - prefer[1])
            row, col = (int(v) for v in near[int(np.argmin(distances))])
            peak = float(surface[row, col])

    y = float(row)
    if 0 < row < surface.shape[0] - 1:
        y = _refine_parabolic(
            (row - 1.0, float(surface[row - 1, col])),
            (float(row), peak),
            (row + 1.0, float(surface[row + 1, col])),
        )
    x = float(col)
    if 0 < col < surface.shape[1] - 1:
        x = _refine_parabolic(
            (col - 1.0, float(surface[row, col - 1])),
            (float(col), peak),
            (col + 1.0, float(surface[row, col + 1])),
        )
    return y, x, peak


def refine_translation(
    search_bp: np.ndarray,
    template_bp: np.ndarray,
    x: float,
    y: float,
    radius: int = TRANSLATION_RADIUS_PX,
) -> tuple[float, float, float]:
    """Re-centre ``(x, y)`` by local correlation at full resolution.

    Correlating a window rather than the whole search image is what makes this
    affordable: cost scales with the window, not the 1000x1000 field. The
    window is sized to hold the template plus ``radius`` of slack on every
    side, so the reachable displacement is exactly ``radius``.
    """
    th, tw = template_bp.shape[:2]
    sh, sw = search_bp.shape[:2]
    if th < 3 or tw < 3 or th + 2 * radius > sh or tw + 2 * radius > sw:
        return float(x), float(y), float("nan")

    half_h, half_w = (th - 1) / 2.0, (tw - 1) / 2.0
    top = int(round(y - half_h)) - radius
    left = int(round(x - half_w)) - radius
    win_h, win_w = th + 2 * radius, tw + 2 * radius
    top = int(np.clip(top, 0, sh - win_h))
    left = int(np.clip(left, 0, sw - win_w))

    window = search_bp[top:top + win_h, left:left + win_w]
    if window.shape != (win_h, win_w):
        return float(x), float(y), float("nan")
    if float(np.std(template_bp)) < 1e-6 or float(np.std(window)) < 1e-6:
        return float(x), float(y), float("nan")

    # The incoming estimate sits at the window centre by construction, which
    # is where the equivalence tie rule should pull toward.
    prefer = ((surface_h := win_h - th) / 2.0, (surface_w := win_w - tw) / 2.0)
    surface = _ncc_valid(window.astype(np.float32), template_bp.astype(np.float32))
    row, col, peak = _subpixel_peak(surface, prefer=prefer)
    if not np.isfinite(peak):
        return float(x), float(y), float("nan")

    # Surface index (row, col) places the template's top-left there, so its
    # centre lands half a template further in.
    return float(left + col + half_w), float(top + row + half_h), float(peak)


def refine_candidate(
    reference: np.ndarray,
    search: np.ndarray,
    x: float,
    y: float,
    scale: float,
    rotation: float,
    *,
    passes: int = DEFAULT_PASSES,
    translation_radius: int = TRANSLATION_RADIUS_PX,
    scale_half_window: float = SCALE_HALF_WINDOW,
    rotation_half_window: float = ROTATION_HALF_WINDOW_DEG,
    position_tol_px: float = 0.05,
) -> RefinedPose:
    """Alternate translation, scale and rotation refinement at one site.

    ``scale`` is the **down-scaling factor** ``s`` throughout, matching
    :func:`driftforge.pose.build_template`. Search windows are clamped to the
    disclosed ranges -- ``s in [8, 12]``, ``theta in [-5, 5]`` -- because the
    addendum states those bounds are exact and hard-coding them is intended.

    A pose that lands hard against a clamp is reported as-is rather than being
    silently pulled inside; that a candidate needed an out-of-range pose is
    evidence about the candidate, and belongs to the rejection stage.
    """
    search_bp = band_pass(search)
    current_x, current_y = float(x), float(y)
    current_s = float(np.clip(scale, *SCALE_LIMITS))
    current_r = float(np.clip(rotation, -ROTATION_LIMIT_DEG, ROTATION_LIMIT_DEG))
    result = RefinedPose(current_x, current_y, current_s, current_r, float("nan"))

    for index in range(max(1, passes)):
        template = build_template(reference, current_s, current_r)
        if not np.isfinite(template).all() or float(template.std()) < 1e-6:
            break
        moved_x, moved_y, peak = refine_translation(
            search_bp, band_pass(template), current_x, current_y, translation_radius
        )
        if np.isfinite(peak):
            shift = float(np.hypot(moved_x - current_x, moved_y - current_y))
            current_x, current_y = moved_x, moved_y
            result.score = peak
        else:
            shift = 0.0

        s_lo = max(SCALE_LIMITS[0], current_s - scale_half_window)
        s_hi = min(SCALE_LIMITS[1], current_s + scale_half_window)
        if s_hi > s_lo:
            estimate, _ = scale_oracle(
                reference, search, current_x, current_y, current_r,
                lo=s_lo, hi=s_hi, shape_scale=current_s,
            )
            if np.isfinite(estimate):
                current_s = float(np.clip(estimate, *SCALE_LIMITS))

        r_lo = max(-ROTATION_LIMIT_DEG, current_r - rotation_half_window)
        r_hi = min(ROTATION_LIMIT_DEG, current_r + rotation_half_window)
        if r_hi > r_lo:
            estimate, _ = rotation_oracle(
                reference, search, current_x, current_y, current_s,
                lo=r_lo, hi=r_hi,
            )
            if np.isfinite(estimate):
                current_r = float(
                    np.clip(estimate, -ROTATION_LIMIT_DEG, ROTATION_LIMIT_DEG)
                )

        result.history.append((current_x, current_y, current_s, current_r))
        result.passes = index + 1
        if shift <= position_tol_px and index > 0:
            result.converged = True
            break

    result.x, result.y = current_x, current_y
    result.scale, result.rotation = current_s, current_r
    return result
