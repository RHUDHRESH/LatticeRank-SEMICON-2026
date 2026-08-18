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


def _entropy(patch: np.ndarray) -> float:
    h, _ = np.histogram(patch, bins=24, range=(0, 255))
    p = h.astype(float) / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


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
        from .structural_descriptor import build_context, describe
        ctx = build_context(reference, search)
        for r in rows:
            r.update(describe(ctx, r["x"], r["y"]))
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
) -> tuple[int, int]:
    """Apply the production equivalence margin and Search-centre tie-break."""
    if not rows or len(rows) != len(scores):
        raise ValueError("candidate rows and scores must be non-empty and aligned")
    height, width = search_shape[:2]
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    maximum = float(np.max(scores))
    equivalent = np.flatnonzero(scores >= maximum - EQUIVALENCE_MARGIN)
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
        compute_candidate_rows(reference, search)
        if candidate_rows is None
        else candidate_rows
    )
    h, w = search.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    if not rows:                                    # never abstain
        return LocateV2Result(cx, cy, 0.0, 0, 0, False,
                              {"note": "no candidates; centre fallback"})

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
            "score_max": smax,
            "prob_max": float(probabilities.max()),
        })
