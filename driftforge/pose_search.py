"""Coarse pose search over the disclosed Phase 2 ranges.

Phase 1 knew the zoom exactly (10x) and treated rotation as noise. Phase 2
discloses ``s in [8, 12]`` and ``theta in [-5, +5]`` but not their values, so
the localizer has to search them. This module adds that search **in front of**
the existing pipeline: it produces the candidate pool that
:func:`driftforge.pipeline.compute_candidate_rows` would otherwise produce at a
single assumed pose, and everything downstream -- periodic cancellation,
structural ranking, the equivalence tie-break -- is unchanged.

Why a decimated pyramid, measured on this repository:

===========================  ==================
one full-resolution hypothesis   ~260-360 ms
one hypothesis at 5x decimation  ~16 ms
one single-site oracle step      ~0.6-2.4 ms
===========================  ==================

A full-image response map answers *where*; a single-site evaluation answers
*what pose*, once the location is roughly known. Conflating the two is what
makes brute force infeasible -- even a modest 5x5 full-resolution grid costs
around 14 s against a 20 s hard timeout, before the structural descriptor and
residual stages run at all. So the coarse stage decimates, and the refinement
stage evaluates single sites at full resolution.

Two conventions are load-bearing here and are stated wherever they are used:

* ``scale`` is always the **down-scaling factor** ``s``: the reference covers
  ``1/s`` of the search's field of view, so the template side is about
  ``1000 / s`` px. This is :func:`driftforge.pose.build_template`'s convention,
  **not** ``baseline._template_from_reference``'s (there ``scale`` multiplies a
  hard-coded 0.1 zoom).
* When the search is decimated by ``d``, a template matching it must be built
  at the **effective** down-scaling ``s * d``, because the search pixels are
  ``d`` times coarser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .channels import CHANNELS, ChannelMaps, harvest, prepare_search, response_maps
from .pose import build_template

#: Decimation for the coarse stage, chosen by measured recall on
#: ``data/phase2/p2_val`` present pairs, not by cost alone. A first screen at
#: n=8 put d=3 at 8/8 within 5 px; **that was small-sample optimism**. On n=20
#: the honest figures for d=3 are:
#:
#: ==========  =========  ==========
#: within      recall     cumulative
#: ==========  =========  ==========
#: 3 px          10/20      50%
#: 5 px          14/20      70%
#: 10 px         16/20      80%
#: 15 px         18/20      90%
#: 30 px         20/20     100%
#: ==========  =========  ==========
#:
#: Median closest-candidate error 3.1 px; runtime median 1.24 s, max 2.18 s.
#: The n=8 screen ranked the decimation options consistently (d=5 worst, d=3
#: best) even though its absolute rates were too optimistic, which is what a
#: screening design is for -- but no absolute claim should be made from it.
#:
#: Decimation, not scale-grid density, is the binding parameter: doubling the
#: scale grid to nine samples at d=4 *lowered* recall while costing 80% more
#: time. The reason is template size -- at d=5 an s=12 template is only
#: ``1000 / (12 * 5) = 16.7`` px across, which is too little structure to
#: discriminate on a periodic field; at d=3 it is 27.8 px.
DECIMATION = 3

#: Coarse scale grid over the disclosed [8, 12]. Recall only -- the refinement
#: stage supplies the precision needed for the 1% pose credit tier.
COARSE_SCALES = (8.0, 9.0, 10.0, 11.0, 12.0)

#: Coarse rotation grid over the disclosed +/-5 deg.
COARSE_ROTATIONS = (-5.0, -2.5, 0.0, 2.5, 5.0)

#: Candidates closer than this (in full-resolution search pixels) are treated
#: as the same site when pooling across pose hypotheses.
MERGE_RADIUS_PX = 6.0

#: Smallest template side worth correlating. Below this the ZNCC is dominated
#: by noise and the map is not informative.
MIN_TEMPLATE_PX = 8


@dataclass(frozen=True)
class Pyramid:
    """A decimated search image plus the mapping back to full resolution."""

    image: np.ndarray
    factor: float
    full_shape: tuple[int, int]

    def to_full(self, x: float, y: float) -> tuple[float, float]:
        """Map decimated coordinates back to full-resolution search pixels.

        ``ndimage.zoom`` maps output index ``j`` to input index
        ``j * (n_in - 1) / (n_out - 1)``, so the inverse uses the same ratio.
        Using the nominal factor instead would drift by up to half the
        decimation across the image -- several pixels at 5x, which is the
        whole localization credit range.
        """
        fh, fw = self.full_shape
        dh, dw = self.image.shape[:2]
        sx = (fw - 1) / (dw - 1) if dw > 1 else 1.0
        sy = (fh - 1) / (dh - 1) if dh > 1 else 1.0
        return float(x * sx), float(y * sy)


def decimate(search: np.ndarray, factor: float = DECIMATION) -> Pyramid:
    """Anti-aliased decimation of the search image.

    The Gaussian must precede the resample: decimating by ``d`` without first
    removing content above the new Nyquist limit aliases fine periodic
    structure into false low-frequency patterns -- and on DRAM and FinFET
    fields that structure *is* the signal, so the aliases land exactly where a
    matcher looks. ``sigma = 0.35 * d`` approximates a box filter of width
    ``d``, the same rule :func:`driftforge.pose.build_template` uses.
    """
    image = np.asarray(search, dtype=np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    sigma = float(np.clip(0.35 * factor, 0.8, 5.0))
    blurred = ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")
    small = ndimage.zoom(blurred, zoom=1.0 / factor, order=1, prefilter=False)
    return Pyramid(
        image=small.astype(np.float32),
        factor=float(factor),
        full_shape=(int(image.shape[0]), int(image.shape[1])),
    )


class _SpatialPool:
    """Merge candidates by proximity in constant time per insert.

    A linear scan per insert is O(n^2) over the whole pool, and the pool runs
    to thousands of candidates across a 25-hypothesis grid: measured, that
    alone pushed the worst pair to 51 s against a 20 s hard timeout, while the
    median stayed near 2 s. Bucketing by ``radius`` bounds each insert to the
    nine neighbouring cells regardless of pool size.
    """

    __slots__ = ("radius", "_cells")

    def __init__(self, radius: float) -> None:
        self.radius = float(radius)
        self._cells: dict[tuple[int, int], list[dict]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.radius), int(y // self.radius))

    def add(self, candidate: dict) -> None:
        cx, cy = candidate["x"], candidate["y"]
        kx, ky = self._key(cx, cy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for existing in self._cells.get((kx + dx, ky + dy), ()):
                    if (
                        abs(existing["x"] - cx) <= self.radius
                        and abs(existing["y"] - cy) <= self.radius
                    ):
                        if candidate["pose_score"] > existing["pose_score"]:
                            existing.update(candidate)
                        return
        self._cells.setdefault((kx, ky), []).append(candidate)

    def records(self) -> list[dict]:
        return [record for cell in self._cells.values() for record in cell]

    def __len__(self) -> int:
        return sum(len(cell) for cell in self._cells.values())


def _best_channel(record: dict) -> float:
    values = [
        record[c]
        for c in CHANNELS
        if c in record and np.isfinite(record[c])
    ]
    return float(max(values)) if values else float("-inf")


def coarse_pose_candidates(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    decimation: float = DECIMATION,
    scales: tuple[float, ...] = COARSE_SCALES,
    rotations: tuple[float, ...] = COARSE_ROTATIONS,
    delta: float = 0.10,
    max_candidates: int = 4000,
    deadline=None,
) -> list[dict]:
    """Harvest candidate sites across the disclosed pose grid.

    Returns records carrying full-resolution ``x``/``y``, the ``scale`` and
    ``rotation`` of the hypothesis that found them, the three channel scores,
    and ``pose_score`` (the best channel response). Sites found under several
    hypotheses are merged, keeping the strongest.

    This stage is for **recall**, not precision. Phase 1 measured that the
    correct site is a raw local maximum in 100% of fixed scenes and enters the
    pool in 90%; the job here is to preserve that under an unknown pose, and
    leave selection to the existing ranking stages.

    Two measured cautions. The true site's rank by ``pose_score`` is poor --
    median several hundred, worst case beyond 2600 -- so ``max_candidates``
    must stay generous or the cap silently discards the answer; capping at 400
    dropped recall from 4/6 to 2/6 on the first sample measured. And the
    reported position carries the decimation's quantization floor of about
    ``decimation / 2`` px, which is already outside the 1 px credit tier, so
    the refinement stage is not optional.
    """
    pyramid = decimate(search, decimation)
    prepared = prepare_search(pyramid.image)
    pool = _SpatialPool(MERGE_RADIUS_PX)
    evaluated = 0
    skipped_small = 0

    for scale in scales:
        effective = float(scale) * float(decimation)
        side = 1000.0 / effective
        if side < MIN_TEMPLATE_PX:
            skipped_small += 1
            continue
        for rotation in rotations:
            if deadline is not None and not deadline.affords(
                "coarse_pose_hypothesis", estimate_s=0.05
            ):
                if deadline is not None:
                    deadline.degrade(
                        "coarse_pose_grid",
                        f"evaluated {evaluated} of {len(scales) * len(rotations)} hypotheses",
                    )
                return _finalize(pool, max_candidates)

            template = build_template(reference, effective, float(rotation))
            if (
                template.shape[0] >= pyramid.image.shape[0]
                or template.shape[1] >= pyramid.image.shape[1]
                or not np.isfinite(template).all()
                or float(template.std()) < 1e-6
            ):
                # A constant template makes every normalized correlation NaN;
                # an oversized one has no valid correlation region at all.
                continue

            maps: ChannelMaps = response_maps(
                reference,
                pyramid.image,
                scale=float(scale),
                rotation=float(rotation),
                prepared=prepared,
                template=template,
            )
            evaluated += 1
            for record in harvest(maps, delta=delta):
                fx, fy = pyramid.to_full(record["x"], record["y"])
                record["x"], record["y"] = fx, fy
                record["scale"] = float(scale)
                record["rotation"] = float(rotation)
                record["pose_score"] = _best_channel(record)
                pool.add(record)

    return _finalize(pool, max_candidates)


def _finalize(pool: "_SpatialPool", max_candidates: int) -> list[dict]:
    records = pool.records()
    records.sort(key=lambda r: -r.get("pose_score", float("-inf")))
    return records[:max_candidates]
