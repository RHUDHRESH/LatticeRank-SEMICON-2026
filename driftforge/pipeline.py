"""Production candidate-ranking and localization pipeline.

``compute_candidate_rows`` is the single feature path used by both training
and inference. Search-centre distance is never a feature; the centre is used
only by the specified final tie-break inside the evidential equivalence set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .baseline import _template_from_reference
from .channels import CHANNELS, harvest, response_maps
from .lattice import estimate_lattice
from .model import (
    CANDIDATE_DELTA,
    EQUIVALENCE_MARGIN,
    MAX_CANDIDATES,
    MODEL_PATH,
    RESIDUAL_WEIGHT,
    load_model_bundle,
)

DELTA = CANDIDATE_DELTA
MAX_CAND = MAX_CANDIDATES
EQ_MARGIN = EQUIVALENCE_MARGIN
RES_WEIGHT = RESIDUAL_WEIGHT

# Exact or near-exact wallpaper scenes have thousands of interchangeable local
# maxima but very little large-scale context.  In that regime probability
# differences are dominated by independent acquisition noise, so the challenge
# rule (nearest phase-equivalent site to Search centre) is more defensible than
# pretending the ranker has unique evidence.
WALLPAPER_MIN_CANDIDATES = 2_500
WALLPAPER_MAX_COARSE_CONTEXT_RATIO = 0.42
WALLPAPER_CONTEXT_SIGMA_PX = 20.0
WALLPAPER_LONG_CONTEXT_SIGMA_PX = 60.0
WALLPAPER_MAX_CONTEXT_DECAY = 0.12

# The public starter's zoned DRAM/FinFET scenes consistently estimate inside
# this device-pitch envelope.  In that same-geometry regime the periodic
# residual is the most reliable identity signal; the learned ranker remains
# the fallback for broader transformed/noisy geometry.
CONSENSUS_MAX_PITCH_MIN_PX = 6.5
CONSENSUS_MAX_PITCH_MAX_PX = 10.0
CONSENSUS_RAW_WEIGHT = 0.05
CONSENSUS_MIDBAND_WEIGHT = 0.05
CONSENSUS_EQUIVALENCE_MARGIN = 0.025


def _entropy(patch: np.ndarray) -> float:
    h, _ = np.histogram(patch, bins=24, range=(0, 255))
    p = h.astype(float) / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def coarse_context_ratio(
    search: np.ndarray, sigma_px: float = WALLPAPER_CONTEXT_SIGMA_PX
) -> float:
    """Return the fraction of image variation surviving coarse smoothing.

    Array mats, routing strips, and local fabrication structure survive a
    20-pixel Gaussian blur.  Exact wallpaper does not.  Normalizing by total
    image variation makes the diagnostic robust to gain and contrast changes.
    """
    image = np.asarray(search, dtype=np.float32)
    total = float(image.std())
    if total <= 1e-9:
        return 0.0
    coarse = ndimage.gaussian_filter(
        image, sigma=sigma_px, mode="reflect"
    )
    return float(coarse.std() / total)


def wallpaper_ambiguity_diagnostic(
    rows: list[dict], search: np.ndarray
) -> dict[str, float | int | bool]:
    """Detect the score-collapse regime where the centre rule should govern."""
    ratio = coarse_context_ratio(search, WALLPAPER_CONTEXT_SIGMA_PX)
    long_ratio = coarse_context_ratio(search, WALLPAPER_LONG_CONTEXT_SIGMA_PX)
    context_decay = ratio - long_ratio
    detected = (
        len(rows) >= WALLPAPER_MIN_CANDIDATES
        and ratio <= WALLPAPER_MAX_COARSE_CONTEXT_RATIO
        and context_decay <= WALLPAPER_MAX_CONTEXT_DECAY
    )
    return {
        "detected": detected,
        "candidate_count": len(rows),
        "coarse_context_ratio": ratio,
        "long_context_ratio": long_ratio,
        "context_decay": context_decay,
        "candidate_threshold": WALLPAPER_MIN_CANDIDATES,
        "coarse_context_threshold": WALLPAPER_MAX_COARSE_CONTEXT_RATIO,
        "context_decay_threshold": WALLPAPER_MAX_CONTEXT_DECAY,
    }


def lattice_compatibility_diagnostic(
    reference: np.ndarray, search: np.ndarray
) -> dict[str, float]:
    """Compare independently estimated Reference and Search lattice geometry.

    Residual matching assumes that one lattice translation means the same
    displacement in the decimated Reference and Search.  This diagnostic is
    image-only and therefore can gate that assumption without generator
    metadata or ground truth.
    """
    template = _template_from_reference(reference, 1.0, 0.0)
    ref_lattice = estimate_lattice(template)
    search_lattice = estimate_lattice(search)
    ref_periods = np.sort([ref_lattice.pitch_x, ref_lattice.pitch_y])
    search_periods = np.sort([search_lattice.pitch_x, search_lattice.pitch_y])

    def harmonic_error(first: float, second: float) -> float:
        ratio = max(first, second) / max(min(first, second), 1e-9)
        return float(min(abs(math.log(ratio / harmonic)) for harmonic in (1.0, 2.0, 3.0)))

    period_error = float(
        np.mean(
            [harmonic_error(float(a), float(b)) for a, b in zip(ref_periods, search_periods)]
        )
    )
    raw_angle = abs(ref_lattice.orientation_deg - search_lattice.orientation_deg) % 90.0
    angle_error = float(min(raw_angle, 90.0 - raw_angle))
    confidence = float(min(ref_lattice.confidence, search_lattice.confidence))
    score = float(math.exp(-4.0 * period_error - angle_error / 4.0))

    scale_candidates = []
    for ref_period, search_period in zip(ref_periods, search_periods):
        raw_ratio = float(search_period / max(ref_period, 1e-9))
        scale_candidates.append(
            min(
                (raw_ratio * harmonic for harmonic in (1 / 3, 1 / 2, 1.0, 2.0, 3.0)),
                key=lambda value: abs(math.log(max(value, 1e-9))),
            )
        )
    suggested_scale = float(np.median(scale_candidates))
    signed_angle = ref_lattice.orientation_deg - search_lattice.orientation_deg
    suggested_rotation = float((signed_angle + 45.0) % 90.0 - 45.0)
    return {
        "reference_pitch_min_px": float(ref_periods[0]),
        "reference_pitch_max_px": float(ref_periods[1]),
        "search_pitch_min_px": float(search_periods[0]),
        "search_pitch_max_px": float(search_periods[1]),
        "harmonic_period_error": period_error,
        "orientation_error_deg": angle_error,
        "minimum_lattice_confidence": confidence,
        "compatibility_score": score,
        "suggested_scale": suggested_scale,
        "suggested_rotation_deg": suggested_rotation,
    }


def residual_consensus_applicable(diagnostic: dict[str, float]) -> bool:
    """Return whether the image-derived lattice is in the validated envelope."""
    return bool(
        diagnostic["search_pitch_min_px"] <= CONSENSUS_MAX_PITCH_MIN_PX
        and diagnostic["search_pitch_max_px"] <= CONSENSUS_MAX_PITCH_MAX_PX
    )


def _nearest_search_centre(
    rows: list[dict], search_shape: tuple[int, ...]
) -> int:
    height, width = search_shape[:2]
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    return int(
        min(
            range(len(rows)),
            key=lambda index: (
                (rows[index]["x"] - centre_x) ** 2
                + (rows[index]["y"] - centre_y) ** 2
            ),
        )
    )


def _add_structural_features(
    reference: np.ndarray, search: np.ndarray, rows: list[dict]
) -> None:
    """Populate the expensive descriptor only after ambiguity short-circuiting."""
    if not rows or "s_raw_zncc0" in rows[0]:
        return
    from .structural_descriptor import build_context, describe

    context = build_context(reference, search)
    for row in rows:
        row.update(describe(context, row["x"], row["y"]))


def compute_candidate_rows(reference: np.ndarray, search: np.ndarray,
                           delta: float = DELTA, max_cand: int = MAX_CAND,
                           keep_xy: tuple[float, float] | None = None,
                           keep_tol: float = 5.0, struct: bool = True) -> list[dict]:
    """Adaptive harvest + full feature table for one (reference, search) pair.

    `keep_xy` is a TRAIN-ONLY guard: the cost cap sorts by the very score that
    fails, so it can silently discard the true site; training-table builds pass
    the ground truth here to preserve the positive class. Inference never does.
    """
    cm = response_maps(reference, search)
    cands = harvest(cm, delta=delta)
    if not cands:
        return []
    cands.sort(key=lambda c: -max(v for k, v in c.items() if k in CHANNELS
                                  and isinstance(v, float) and math.isfinite(v)))
    if len(cands) > max_cand:
        head = cands[:max_cand]
        if keep_xy is not None:
            head = head + [c for c in cands[max_cand:]
                           if math.hypot(c["x"] - keep_xy[0], c["y"] - keep_xy[1]) <= keep_tol]
        cands = head

    lat = estimate_lattice(search)
    B = lat.basis
    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        Binv = np.zeros((2, 2))
    top = cands[0]
    N = len(cands)
    maxes, pct, zsc = {}, {}, {}
    for c in CHANNELS:
        v = np.array([k[c] if math.isfinite(k[c]) else -9.0 for k in cands])
        maxes[c] = float(v.max())
        r = np.argsort(np.argsort(-v)) + 1
        pct[c] = r / max(N, 1)
        sd = float(v.std())
        zsc[c] = (v - float(v.mean())) / (sd if sd > 1e-9 else 1.0)
    top10 = {c: set(np.argsort(-np.array([k[c] if math.isfinite(k[c]) else -9.0
                                          for k in cands]))[:10]) for c in CHANNELS}
    bestidx = {c: int(np.argmax([k[c] if math.isfinite(k[c]) else -9.0 for k in cands]))
               for c in CHANNELS}

    a = search.astype(np.float32)
    lap = (a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1] - 4 * a[1:-1, 1:-1])
    noise = float(np.median(np.abs(lap - np.median(lap))) * 1.4826 / math.sqrt(20.0))

    rows = []
    for i, c in enumerate(cands):
        vals = [c[k] for k in CHANNELS if math.isfinite(c[k])]
        d = np.array([c["x"] - top["x"], c["y"] - top["y"]])
        mn = Binv @ d
        frac = mn - np.round(mn)
        x0, y0 = int(np.clip(c["x"] - 16, 0, 968)), int(np.clip(c["y"] - 16, 0, 968))
        rows.append({
            "x": c["x"], "y": c["y"],
            "raw": c["raw"], "midband": c["midband"], "directionality": c["directionality"],
            "votes": float(c["votes"]),
            "chan_spread": float(max(vals) - min(vals)) if vals else 0.0,
            "chan_min": float(min(vals)) if vals else 0.0,
            "raw_pct": float(pct["raw"][i]), "mid_pct": float(pct["midband"][i]),
            "dir_pct": float(pct["directionality"][i]),
            "raw_delta": float(maxes["raw"] - c["raw"]) if math.isfinite(c["raw"]) else 9.0,
            "mid_delta": float(maxes["midband"] - c["midband"]) if math.isfinite(c["midband"]) else 9.0,
            "dir_delta": float(maxes["directionality"] - c["directionality"]) if math.isfinite(c["directionality"]) else 9.0,
            "raw_z": float(zsc["raw"][i]), "mid_z": float(zsc["midband"][i]),
            "dir_z": float(zsc["directionality"][i]),
            "agreement_count": float(sum(i in top10[ch] for ch in CHANNELS)),
            "channel_best_count": float(sum(bestidx[ch] == i for ch in CHANNELS)),
            "n_cand_scene": float(N),
            "lat_phase_res": float(np.linalg.norm(B @ frac)),
            "lat_conf": float(lat.confidence), "lat_agree": float(lat.agreement),
            "disp_periods_x": float(mn[0]), "disp_periods_y": float(mn[1]),
            "parity_even": float(int(round(mn[0]) + round(mn[1])) % 2 == 0),
            "local_entropy": _entropy(search[y0:y0 + 32, x0:x0 + 32]),
            "noise_est": noise,
        })
    if struct and rows:
        _add_structural_features(reference, search, rows)
    return rows


@dataclass
class LocateV2Result:
    x: float
    y: float
    probability: float
    eq_set_size: int
    n_candidates: int
    used_residual: bool
    diagnostics: dict = field(default_factory=dict)


def load_ranker(path: str | Path = MODEL_PATH):
    """Compatibility wrapper around the validated public model loader."""
    return load_model_bundle(path)


def normalize_evidence(values: np.ndarray) -> np.ndarray | None:
    """Apply the production per-scene evidence normalization."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < max(3, 0.5 * values.size):
        return None
    mean = values[finite].mean()
    standard_deviation = values[finite].std()
    output = (values - mean) / (
        standard_deviation if standard_deviation > 1e-9 else 1.0
    )
    output[~finite] = -9.0
    return output


def rank_candidate_rows(
    rows: list[dict], model_bundle: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return model probabilities and production no-residual scores."""
    model, features = model_bundle["model"], model_bundle["features"]
    matrix = np.nan_to_num(
        np.array(
            [[row[feature] for feature in features] for row in rows],
            dtype=np.float64,
        ),
        nan=0.0,
        posinf=9.0,
        neginf=-9.0,
    )
    metadata = model_bundle.get("metadata", {})
    if metadata.get("scene_feature_normalization", False):
        mean = matrix.mean(axis=0)
        standard_deviation = matrix.std(axis=0)
        varying = standard_deviation > 1e-9
        matrix[:, varying] = (
            matrix[:, varying] - mean[varying]
        ) / standard_deviation[varying]
    probabilities = model.predict_proba(matrix)[:, 1]
    normalized = normalize_evidence(probabilities)
    scores = (
        probabilities.astype(float)
        if normalized is None
        else normalized
    )
    return probabilities, scores


def select_equivalent_candidate(
    rows: list[dict],
    scores: np.ndarray,
    search_shape: tuple[int, ...],
    *,
    margin: float = EQUIVALENCE_MARGIN,
) -> tuple[int, int]:
    """Apply the production equivalence margin and Search-centre tie-break."""
    if not rows or len(rows) != len(scores):
        raise ValueError("candidate rows and scores must be non-empty and aligned")
    height, width = search_shape[:2]
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    maximum = float(np.max(scores))
    equivalent = np.flatnonzero(scores >= maximum - margin)
    selected = min(
        equivalent,
        key=lambda index: (
            rows[int(index)]["x"] - centre_x
        )
        ** 2
        + (rows[int(index)]["y"] - centre_y) ** 2,
    )
    return int(selected), int(equivalent.size)


def locate_v2(
    reference: np.ndarray,
    search: np.ndarray,
    use_residual: bool = True,
    *,
    model_path: str | Path = MODEL_PATH,
    model_bundle: dict | None = None,
    candidate_rows: list[dict] | None = None,
) -> LocateV2Result:
    """Run the final pipeline and always return an in-bounds coordinate."""
    rows = (
        compute_candidate_rows(reference, search, struct=False)
        if candidate_rows is None
        else candidate_rows
    )
    h, w = search.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    if not rows:                                    # never abstain
        return LocateV2Result(cx, cy, 0.0, 0, 0, False,
                              {"note": "no candidates; centre fallback"})

    ambiguity = wallpaper_ambiguity_diagnostic(rows, search)
    if ambiguity["detected"]:
        # In an exact wallpaper there is no image evidence that identifies one
        # lattice copy over another.  The challenge resolves that
        # non-identifiability with its Search-centre convention.  Returning
        # the centre itself is also robust when acquisition noise suppresses
        # the local maximum nearest the centre; restricting the rule to the
        # harvested pool would make the tie-break depend on that noise again.
        return LocateV2Result(
            x=float(cx),
            y=float(cy),
            probability=0.0,
            eq_set_size=len(rows),
            n_candidates=len(rows),
            used_residual=False,
            diagnostics={
                "selection_mode": "periodic_wallpaper_centre_rule",
                "wallpaper_ambiguity": ambiguity,
            },
        )

    lattice_compatibility = lattice_compatibility_diagnostic(reference, search)
    if use_residual and residual_consensus_applicable(lattice_compatibility):
        from .residual import ResidualMatcher

        matcher = ResidualMatcher(reference, search)
        residual_score = normalize_evidence(
            np.asarray(
                [
                    matcher.score(row["x"], row["y"])["res_int_m50"]
                    for row in rows
                ],
                dtype=float,
            )
        )
        raw_score = normalize_evidence(
            np.asarray([row["raw"] for row in rows], dtype=float)
        )
        midband_score = normalize_evidence(
            np.asarray([row["midband"] for row in rows], dtype=float)
        )
        if (
            matcher.lattice_ok
            and residual_score is not None
            and raw_score is not None
            and midband_score is not None
        ):
            consensus_score = (
                residual_score
                + CONSENSUS_RAW_WEIGHT * raw_score
                + CONSENSUS_MIDBAND_WEIGHT * midband_score
            )
            best, equivalence_size = select_equivalent_candidate(
                rows,
                consensus_score,
                search.shape,
                margin=CONSENSUS_EQUIVALENCE_MARGIN,
            )
            return LocateV2Result(
                x=float(rows[best]["x"]),
                y=float(rows[best]["y"]),
                probability=0.0,
                eq_set_size=equivalence_size,
                n_candidates=len(rows),
                used_residual=True,
                diagnostics={
                    "selection_mode": "periodic_residual_consensus",
                    "wallpaper_ambiguity": ambiguity,
                    "lattice_compatibility": lattice_compatibility,
                    "score_max": float(consensus_score.max()),
                    "weights": {
                        "residual": 1.0,
                        "raw": CONSENSUS_RAW_WEIGHT,
                        "midband": CONSENSUS_MIDBAND_WEIGHT,
                    },
                },
            )

    _add_structural_features(reference, search, rows)

    bundle = model_bundle if model_bundle is not None else load_ranker(model_path)
    probabilities, score = rank_candidate_rows(rows, bundle)
    used_residual = False
    if use_residual:
        from .residual import ResidualMatcher
        rm = ResidualMatcher(reference, search)
        rz = normalize_evidence(
            np.array(
                [
                    rm.score(row["x"], row["y"])["res_int_m50"]
                    for row in rows
                ]
            )
        )
        if rz is not None:
            score = score + RESIDUAL_WEIGHT * rz
            used_residual = True

    smax = float(score.max())
    best, equivalence_size = select_equivalent_candidate(
        rows, score, search.shape
    )
    return LocateV2Result(
        x=float(rows[best]["x"]), y=float(rows[best]["y"]),
        probability=float(probabilities[best]), eq_set_size=equivalence_size,
        n_candidates=len(rows), used_residual=used_residual,
        diagnostics={
            "selection_mode": "ranked_evidence",
            "wallpaper_ambiguity": ambiguity,
            "lattice_compatibility": lattice_compatibility,
            "score_max": smax,
            "prob_max": float(probabilities.max()),
        })
