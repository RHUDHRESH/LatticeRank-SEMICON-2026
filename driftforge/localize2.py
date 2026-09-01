"""End-to-end Phase 2 localization: coarse pose search, screen, refine, select.

The Phase 1 pipeline detects well and selects badly -- the correct site is a raw
local maximum in 100% of fixed scenes and enters the candidate pool in 90%, but
final selection reaches 48.75%. Phase 2 makes that worse by hiding the zoom and
rotation, and adds a class of pairs where the correct answer is to select
nothing at all.

The idea here is that **refinement is itself the selector**. A candidate at the
true site pulls into a sharp, high correlation when its pose is refined at full
resolution; an alias does not, because no pose makes a wrong site fit. Coarse
scores from a decimated image cannot separate those two cases -- they are
computed on exactly the blurred, aliased evidence that makes lattice copies look
identical -- but a refined full-resolution correlation can.

That gives a three-tier funnel, each tier an order of magnitude more expensive
and an order of magnitude more selective:

===========  ===========================  ==============  ==============
tier         work                         candidates      cost each
===========  ===========================  ==============  ==============
coarse       decimated grid, 25 poses     ~4000 out       ~1.2 s total
screen       one full-res ZNCC per site   ~32 in          ~1-3 ms
refine       alternating pose refinement  ~4 in           ~400 ms
===========  ===========================  ==============  ==============

Templates are cached by ``(scale, rotation)`` across the screen, because the
coarse grid has at most ``len(scales) * len(rotations)`` distinct poses however
many candidates carry them -- building one per candidate would dominate the
screen and buy nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pipeline import lattice_compatibility_diagnostic, locate_v2
from .pose import band_pass, build_template, zncc_at
from .pose_search import COARSE_ROTATIONS, COARSE_SCALES, DECIMATION, coarse_pose_candidates
from .refine import RefinedPose, refine_candidate

#: Candidates carried from the coarse pool into the full-resolution screen.
#:
#: **This must not be a small number, and the reason is measured.** The true
#: site's rank by *coarse* score is terrible -- after spatial spreading its
#: median rank is 204 and its worst observed rank is 443. Screening only the
#: coarse top 32 therefore hands the screen the right answer on just 2 pairs in
#: 10, and no amount of downstream cleverness recovers a candidate that was
#: never passed on. End to end that scored 2.9 of 40 points.
#:
#: ====  ==========================
#: k     true site reaches screen
#: ====  ==========================
#: 32          2/10
#: 100         4/10
#: 400         9/10
#: 800        10/10
#: ====  ==========================
#:
#: Spatial spreading already collapses ~4000 raw candidates to ~400-700
#: distinct sites, and the screen is one single-site ZNCC against a cached
#: template, so screening all of them is affordable. The coarse score's job is
#: recall; ranking is the screen's job, and it must be given the chance to do
#: it.
SCREEN_K = 1200

#: Screened candidates that receive full pose refinement.
REFINE_K = 4

#: Minimum separation between screened candidates, in full-resolution pixels.
#: Without it the screen fills with neighbours of one peak and the funnel
#: narrows to a single site before the expensive tier ever sees an alternative.
SCREEN_MIN_SEPARATION_PX = 24.0


@dataclass
class Phase2Result:
    """One localized pair, in the Phase 2 output contract's terms."""

    x: float
    y: float
    scale: float
    rotation: float
    score: float
    n_coarse: int = 0
    n_screened: int = 0
    n_refined: int = 0
    runner_up_score: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Refined score minus the best distinct alternative.

        This is the single most informative presence feature the pipeline
        produces: on a present pair one site wins outright, and on an absent
        pair the field stays flat because no site is the real one.
        """
        if not np.isfinite(self.runner_up_score):
            return float("nan")
        return float(self.score - self.runner_up_score)


def _template_cache():
    cache: dict[tuple[float, float], np.ndarray | None] = {}

    def get(reference: np.ndarray, scale: float, rotation: float) -> np.ndarray | None:
        key = (round(float(scale), 4), round(float(rotation), 4))
        if key not in cache:
            try:
                template = build_template(reference, float(scale), float(rotation))
            except ValueError:
                cache[key] = None
                return None
            if not np.isfinite(template).all() or float(template.std()) < 1e-6:
                cache[key] = None
            else:
                cache[key] = band_pass(template)
        return cache[key]

    return get


def _spread(records: list[dict], keep: int, min_separation: float) -> list[dict]:
    """Take the best ``keep`` records that are mutually well separated."""
    chosen: list[dict] = []
    for record in records:
        if all(
            np.hypot(record["x"] - other["x"], record["y"] - other["y"])
            >= min_separation
            for other in chosen
        ):
            chosen.append(record)
            if len(chosen) >= keep:
                break
    return chosen


def localize_phase2_p1(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    model_bundle=None,
    deadline=None,
    refine_pose: bool = True,
) -> Phase2Result:
    """Estimate the zoom, then run the **unchanged Phase 1 selection** at it.

    This is the architecture the measurements argue for, and the one the rules
    argue for, and they turn out to be the same architecture.

    Raw correlation cannot pick the true site on a periodic field. Measured with
    *ground-truth* pose on ``p2_val``, the true site is the global correlation
    maximum on only 4 of 8 pairs -- yet its rank is between 0 and 695 out of
    roughly 800,000 positions, i.e. always inside the top 0.09%. Detection is
    not the problem; ``argmax`` is. Phase 1 already solved that with periodic
    cancellation, the structural descriptor and the learned ranker, and it is
    the declared method the submission rules require Phase 2 to extend rather
    than replace.

    So Phase 2 adds exactly one thing in front: the zoom is no longer given, so
    estimate it. :func:`lattice_compatibility_diagnostic` does that from lattice
    geometry alone, needing no candidate position -- measured 9/12 within 5% of
    truth, median 1.6% error, in 0.16 s.

    Rotation is deliberately *not* taken from the lattice. A lattice is
    four-fold symmetric, so its orientation is only defined modulo 90 degrees;
    the suggested rotation was measured at 42-47 degrees of error, having picked
    the wrong branch. Over a band as narrow as the disclosed +/-5 degrees the
    multi-channel response tolerates the residual rotation, and the exact angle
    is recovered by refinement at the selected site.
    """
    height, width = search.shape[:2]
    diagnostic = lattice_compatibility_diagnostic(reference, search)
    suggested = float(diagnostic.get("suggested_scale", 1.0))
    down_scaling = float(np.clip(10.0 / suggested, 7.5, 12.5)) if suggested > 1e-6 else 10.0
    # _template_from_reference's convention: internal zoom is 0.1 * template_scale,
    # so a down-scaling factor s needs template_scale = 10 / s.
    template_scale = 10.0 / down_scaling

    located = locate_v2(
        reference, search,
        model_bundle=model_bundle,
        deadline=deadline,
        template_scale=template_scale,
        template_rotation=0.0,
    )

    x, y = float(located.x), float(located.y)
    scale, rotation, score = down_scaling, 0.0, float(located.probability)
    if refine_pose:
        pose = refine_candidate(reference, search, x, y, down_scaling, 0.0)
        if np.isfinite(pose.score):
            x, y, scale, rotation, score = (
                pose.x, pose.y, pose.scale, pose.rotation, pose.score
            )

    return Phase2Result(
        x=x, y=y, scale=scale, rotation=rotation, score=score,
        n_coarse=located.n_candidates,
        diagnostics={
            "selection_mode": located.diagnostics.get("selection_mode"),
            "lattice_suggested_down_scaling": down_scaling,
            "eq_set_size": located.eq_set_size,
        },
    )


def localize_phase2(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    decimation: float = DECIMATION,
    scales: tuple[float, ...] = COARSE_SCALES,
    rotations: tuple[float, ...] = COARSE_ROTATIONS,
    screen_k: int = SCREEN_K,
    refine_k: int = REFINE_K,
    min_separation: float = SCREEN_MIN_SEPARATION_PX,
    deadline=None,
) -> Phase2Result:
    """Locate the reference in the search image and recover its pose.

    Always returns a result. A pair with no usable candidate returns the search
    centre with a non-finite score, which the caller must treat as "no evidence"
    rather than as a confident answer -- the distinction the rejection stage
    depends on.
    """
    height, width = search.shape[:2]
    centre = Phase2Result(
        x=(width - 1) / 2.0, y=(height - 1) / 2.0,
        scale=float(np.mean(scales)), rotation=0.0, score=float("nan"),
        diagnostics={"selection_mode": "centre_fallback"},
    )

    coarse = coarse_pose_candidates(
        reference, search, decimation=decimation, scales=scales,
        rotations=rotations, deadline=deadline,
    )
    if not coarse:
        return centre
    centre.n_coarse = len(coarse)

    # -- screen: one full-resolution single-site ZNCC per candidate ----------
    search_bp = band_pass(search)
    get_template = _template_cache()
    screened: list[dict] = []
    for record in _spread(coarse, screen_k, min_separation):
        template = get_template(reference, record["scale"], record["rotation"])
        if template is None:
            continue
        value = zncc_at(search_bp, template, record["x"], record["y"])
        if np.isfinite(value):
            screened.append({**record, "screen_score": float(value)})
    if not screened:
        return centre
    screened.sort(key=lambda r: -r["screen_score"])
    centre.n_screened = len(screened)

    if deadline is not None and not deadline.affords("refine", estimate_s=0.5):
        # Out of budget: the screen's own winner is the honest answer, and the
        # degradation is recorded rather than presented as a refined result.
        deadline.degrade("refine", "skipped: insufficient budget, screen winner used")
        best = screened[0]
        return Phase2Result(
            x=best["x"], y=best["y"], scale=best["scale"], rotation=best["rotation"],
            score=best["screen_score"], n_coarse=len(coarse), n_screened=len(screened),
            runner_up_score=(
                screened[1]["screen_score"] if len(screened) > 1 else float("nan")
            ),
            diagnostics={"selection_mode": "screen_only"},
        )

    # -- refine: the expensive tier, and the actual selector -----------------
    refined: list[tuple[RefinedPose, dict]] = []
    for record in screened[:refine_k]:
        if deadline is not None and not deadline.affords("refine_one", estimate_s=0.45):
            deadline.degrade(
                "refine", f"refined {len(refined)} of {min(refine_k, len(screened))}"
            )
            break
        pose = refine_candidate(
            reference, search, record["x"], record["y"],
            record["scale"], record["rotation"],
        )
        if np.isfinite(pose.score):
            refined.append((pose, record))

    if not refined:
        best = screened[0]
        return Phase2Result(
            x=best["x"], y=best["y"], scale=best["scale"], rotation=best["rotation"],
            score=best["screen_score"], n_coarse=len(coarse), n_screened=len(screened),
            diagnostics={"selection_mode": "screen_only_refine_failed"},
        )

    refined.sort(key=lambda pair: -pair[0].score)
    winner = refined[0][0]
    runner_up = float("nan")
    for pose, _ in refined[1:]:
        if np.hypot(pose.x - winner.x, pose.y - winner.y) >= min_separation:
            runner_up = pose.score
            break

    return Phase2Result(
        x=winner.x, y=winner.y, scale=winner.scale, rotation=winner.rotation,
        score=winner.score, n_coarse=len(coarse), n_screened=len(screened),
        n_refined=len(refined), runner_up_score=runner_up,
        diagnostics={
            "selection_mode": "refined_evidence",
            "converged": winner.converged,
            "passes": winner.passes,
        },
    )
