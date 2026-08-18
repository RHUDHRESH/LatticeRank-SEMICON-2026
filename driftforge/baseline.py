from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    score: float
    rotation_deg: float
    scale: float


@dataclass
class LocateResult:
    x: float
    y: float
    score: float
    peak_gap: float
    psr: float
    rotation_deg: float
    scale: float
    accepted: bool
    ambiguity_detected: bool
    candidates: list[Candidate]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["candidates"] = [asdict(item) for item in self.candidates]
        return data


def _as_float(image: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        out /= float(np.iinfo(image.dtype).max)
    return out


def _robust_contrast(image: np.ndarray) -> np.ndarray:
    image = _as_float(image)
    lo, hi = np.percentile(image, (0.5, 99.5))
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def _template_from_reference(reference: np.ndarray, scale: float, rotation_deg: float) -> np.ndarray:
    # The challenge has a fixed 10:1 physical pixel-size ratio. Resize the
    # entire high-magnification reference; never crop or paste it into Search.
    # Anti-alias before the 10x decimation. Direct interpolation preserves
    # high-magnification grid aliases that do not exist in the Search capture.
    antialiased = ndimage.gaussian_filter(_as_float(reference), sigma=4.0, mode="reflect")
    base = ndimage.zoom(antialiased, zoom=0.1 * scale, order=1, prefilter=False)
    if abs(rotation_deg) > 1e-8:
        base = ndimage.rotate(base, rotation_deg, reshape=False, order=1, mode="reflect", prefilter=False)
    return _robust_contrast(base)


def _ncc_valid(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalized cross-correlation in valid coordinates."""
    template = template.astype(np.float32)
    template = template - float(template.mean())
    template_energy = float(np.sum(template * template))
    if template_energy < 1e-9:
        raise ValueError("template has no usable contrast")

    ones = np.ones(template.shape, dtype=np.float32)
    n = float(template.size)
    local_sum = signal.fftconvolve(image, ones, mode="valid")
    local_sumsq = signal.fftconvolve(image * image, ones, mode="valid")
    local_energy = np.maximum(local_sumsq - local_sum * local_sum / n, 1e-9)
    numerator = signal.fftconvolve(image, template[::-1, ::-1], mode="valid")
    return (numerator / np.sqrt(local_energy * template_energy)).astype(np.float32)


def _peaks(score: np.ndarray, template_shape: tuple[int, int], limit: int) -> list[tuple[float, int, int]]:
    # Large NMS footprint prevents many adjacent pixels of one broad peak from
    # masquerading as separate hypotheses. Period-equivalent peaks remain.
    footprint = max(5, int(round(min(template_shape) * 0.30)))
    maxima = score == ndimage.maximum_filter(score, size=footprint, mode="nearest")
    ys, xs = np.nonzero(maxima)
    if len(xs) == 0:
        return []
    values = score[ys, xs]
    take = min(limit, len(values))
    idx = np.argpartition(values, -take)[-take:]
    idx = idx[np.argsort(values[idx])[::-1]]
    return [(float(values[i]), int(xs[i]), int(ys[i])) for i in idx]


def locate(
    reference: np.ndarray,
    search: np.ndarray,
    rotations: tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0),
    scales: tuple[float, ...] = (0.94, 0.97, 1.0, 1.03, 1.06),
    score_threshold: float = 0.20,
    psr_threshold: float = 3.0,
    tie_delta: float = 0.015,
    candidates_per_variant: int = 12,
) -> LocateResult:
    """Conservative classical baseline for auditing generated pairs.

    `accepted` is an abstention signal, not a claim of hidden-test certainty.
    The returned coordinate always follows the challenge convention x=column,
    y=row. When statistically tied peaks occur, the one nearest Search centre
    wins, as specified by the challenge.
    """
    # NCC itself removes local offset/gain. Retaining low-frequency structural
    # signatures is essential; aggressive high-pass filtering makes a dense
    # periodic lattice overwhelm the very defect/periphery cues that identify
    # the correct field of view.
    search_feature = _robust_contrast(search)
    search_structure = (
        ndimage.gaussian_filter(search_feature, sigma=3.0, mode="reflect")
        - ndimage.gaussian_filter(search_feature, sigma=15.0, mode="reflect")
    ).astype(np.float32)
    all_candidates: list[Candidate] = []
    all_scores: list[np.ndarray] = []

    for scale in scales:
        for rotation in rotations:
            template = _template_from_reference(reference, scale, rotation)
            if template.shape[0] >= search.shape[0] or template.shape[1] >= search.shape[1]:
                continue
            template_structure = (
                ndimage.gaussian_filter(template, sigma=3.0, mode="reflect")
                - ndimage.gaussian_filter(template, sigma=15.0, mode="reflect")
            ).astype(np.float32)
            # Dense lattice pixels are numerous but weakly identifying. Give
            # most weight to the mid-scale band that carries array boundaries,
            # missing contacts, residue and other shared structural context.
            response = (
                0.25 * _ncc_valid(search_feature, template)
                + 0.75 * _ncc_valid(search_structure, template_structure)
            ).astype(np.float32)
            all_scores.append(response)
            half_h = (template.shape[0] - 1.0) / 2.0
            half_w = (template.shape[1] - 1.0) / 2.0
            for value, left, top in _peaks(response, template.shape, candidates_per_variant):
                all_candidates.append(Candidate(left + half_w, top + half_h, value, rotation, scale))

    if not all_candidates:
        raise RuntimeError("no localization candidates were produced")
    all_candidates.sort(key=lambda item: item.score, reverse=True)
    best = all_candidates[0]
    best_score = best.score
    centre_x = (search.shape[1] - 1.0) / 2.0
    centre_y = (search.shape[0] - 1.0) / 2.0
    # Gap excludes nearly coincident transform variants around the top peak.
    separation = 0.35 * 100.0
    rivals = [
        item for item in all_candidates
        if (item.x - best.x) ** 2 + (item.y - best.y) ** 2 >= separation ** 2
    ]
    rival_score = rivals[0].score if rivals else -1.0
    peak_gap = float(best.score - rival_score)
    ambiguity_detected = peak_gap < 0.010
    # Apply the official centre rule only inside a narrow statistical tie.
    # Do not blindly return the image centre: a distinctive target can still
    # have a close periodic rival, and the best peak remains valuable when the
    # caller is forced to output a coordinate despite the confidence rejection.
    tied = [item for item in all_candidates if item.score >= best_score - tie_delta]
    chosen = min(tied, key=lambda item: (item.x - centre_x) ** 2 + (item.y - centre_y) ** 2)

    # Approximate peak-to-sidelobe ratio across all response maps. This remains
    # useful for ranking/rejection even though variants have different shapes.
    pooled = np.concatenate([m.ravel()[:: max(1, m.size // 60_000)] for m in all_scores])
    median = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - median))) * 1.4826 + 1e-6
    psr = float((chosen.score - median) / mad)
    # The gap requirement is deliberately conservative: an apparently strong
    # correlation with an equally strong remote rival is not safe to auto-accept.
    accepted = bool(
        chosen.score >= score_threshold
        and psr >= psr_threshold
        and peak_gap >= 0.020
    )
    return LocateResult(
        x=chosen.x,
        y=chosen.y,
        score=chosen.score,
        peak_gap=peak_gap,
        psr=psr,
        rotation_deg=chosen.rotation_deg,
        scale=chosen.scale,
        accepted=accepted,
        ambiguity_detected=ambiguity_detected,
        candidates=all_candidates[:25],
    )
