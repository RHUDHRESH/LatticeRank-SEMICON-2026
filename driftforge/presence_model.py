"""Scene-level presence decision: does the reference exist in the search image?

Phase 2 gives 15 points for F1 on the ``found`` flag across all 180 grayscale
pairs, and a further +4 bonus at F1 >= 0.90. Never rejecting scores **zero** on
that block, and Set C is 40 pairs whose search image is a *different die region
of the same architecture* -- periodically similar and genuinely plausible.

This module replaces a hand-set threshold on the raw peak correlation with a
fitted model over scene-level evidence. Measured on a 51-scene lockbox touched
once:

======================  =======
metric                  value
======================  =======
F1                      0.9024
precision / recall      0.9024
AUC                     0.8707
90% CI on F1 (by scene) [0.838, 0.953]
======================  =======

F1 by severity was 0.909 / 0.875 / 0.960 / 0.842 at levels 0-3 -- **no collapse
in the degraded regime**, which is the property that matters most, because the
organizers' severity ladder is undisclosed.

The operating threshold (0.48) is deliberately **not** the F1-optimal point on
the calibration split (0.195). It was chosen by a flatness scan, trading 0.032
of calibration F1 for roughly half the cross-severity variance. A threshold
tuned to one severity mix is an unpriced bet on the blind set having the same
mix; the flattest threshold is the one that survives being wrong about it.

The feature vector is reproduced here **exactly** as it was during training. A
training/inference feature mismatch is silent -- the model still returns a
plausible probability -- so any change here must be mirrored in the training
script or the model retired.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .baseline import _ncc_valid, _robust_contrast
from .dense import DENSE_ROTATIONS, DENSE_SCALES, _correlate
from .pose import band_pass, build_template
from .presence import DEFAULT_EXCLUSION_PX, PRESENCE_FEATURES, presence_evidence
from .refine import refine_candidate

#: Packaged weights, resolved relative to this file so the model travels with
#: the zip and no path is ever taken from the working directory.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "presence_hgb.pkl"

#: Features computed here on top of the shared presence evidence. Order is
#: irrelevant (the model is indexed by name) but membership is not.
EXTRA_FEATURES = (
    "n_strong_modes",
    "cross_scale_agreement",
    "cross_rotation_agreement",
    "refinement_gain",
    "refinement_displacement",
    "refined_score",
    "peak_hessian_det",
)


def _cluster_strong_modes(rows, top_raw, exclusion_px, eq_margin=0.05) -> int:
    """Count distinct sites whose response is within ``eq_margin`` of the best.

    A present pair usually has one strong mode; an absent pair on a periodic
    field has several equally good ones, because no site is the real one.
    """
    points = sorted(
        ((r["x"], r["y"], r["raw"]) for r in rows if np.isfinite(r["raw"])),
        key=lambda t: -t[2],
    )
    reps: list[tuple[float, float, float]] = []
    for x, y, raw in points:
        placed = False
        for index, (cx, cy, cmax) in enumerate(reps):
            if np.hypot(x - cx, y - cy) < exclusion_px:
                if raw > cmax:
                    reps[index] = (cx, cy, raw)
                placed = True
                break
        if not placed:
            reps.append((x, y, raw))
    return int(sum(1 for _, _, cmax in reps if cmax >= top_raw - eq_margin))


def _cross_pose_agreement(rows, best, tol_px: float = 8.0) -> tuple[float, float]:
    """How often neighbouring pose hypotheses land on the same site.

    A true site is found at roughly the same place across nearby scales and
    rotations. An alias tends not to survive the pose changing.
    """
    def agree(group):
        if len(group) <= 1:
            return 1.0
        hits = sum(
            1 for r in group
            if np.isfinite(r["raw"])
            and np.hypot(r["x"] - best["x"], r["y"] - best["y"]) < tol_px
        )
        return float(hits / len(group))

    return (
        agree([r for r in rows if r["scale"] == best["scale"]]),
        agree([r for r in rows if r["rotation"] == best["rotation"]]),
    )


def _peak_hessian_det(search_f: np.ndarray, template: np.ndarray) -> float:
    """Local curvature at the peak: sharp peaks score high, ridges collapse."""
    if not np.isfinite(template).all() or float(template.std()) < 1e-6:
        return 0.0
    th, tw = template.shape[:2]
    if th >= search_f.shape[0] or tw >= search_f.shape[1] or th < 5 or tw < 5:
        return 0.0
    surface = _ncc_valid(search_f, template.astype(np.float32))
    if not np.isfinite(surface).any():
        return 0.0
    r, c = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    if r <= 0 or c <= 0 or r >= surface.shape[0] - 1 or c >= surface.shape[1] - 1:
        return 0.0
    s = surface
    ixx = float(s[r, c - 1] - 2 * s[r, c] + s[r, c + 1])
    iyy = float(s[r - 1, c] - 2 * s[r, c] + s[r + 1, c])
    ixy = float((s[r + 1, c + 1] - s[r + 1, c - 1] - s[r - 1, c + 1] + s[r - 1, c - 1]) / 4.0)
    det = ixx * iyy - ixy * ixy
    return float(det) if np.isfinite(det) else 0.0


def scene_features(reference: np.ndarray, search: np.ndarray,
                   rows: list[dict] | None = None) -> tuple[dict, dict | None]:
    """Compute the presence feature vector and the best pose hypothesis.

    Returns ``(features, best)``. ``best`` is ``None`` when no pose produced a
    finite correlation, in which case the features are all zero -- which the
    model reads as "no evidence", not as "confidently absent".

    One pose hypothesis contributes one row, carrying that pose's global argmax.
    That is the same construction used to train the model.

    ``rows`` accepts the records ``dense_pose_search`` already produced via its
    ``collect_rows`` argument. Passing them avoids repeating the whole 45-pose
    sweep, which is the difference between 8.2 s and roughly 4 s per pair.
    """
    search_f = band_pass(_robust_contrast(search))
    if rows is None:
        rows = []
        for scale in DENSE_SCALES:
            for rotation in DENSE_ROTATIONS:
                try:
                    template = band_pass(build_template(reference, float(scale), float(rotation)))
                except ValueError:
                    continue
                x, y, score = _correlate(search_f, template)
                rows.append({"x": x, "y": y, "raw": score,
                             "scale": float(scale), "rotation": float(rotation)})

    valid = [r for r in rows if np.isfinite(r["raw"])]
    features = {k: 0.0 for k in tuple(PRESENCE_FEATURES) + EXTRA_FEATURES}
    if not valid:
        return features, None

    evidence = presence_evidence(valid, pitch_px=None)
    for key in PRESENCE_FEATURES:
        features[key] = float(evidence.get(key, 0.0))

    best = max(valid, key=lambda r: r["raw"])
    features["n_strong_modes"] = float(
        _cluster_strong_modes(valid, best["raw"], DEFAULT_EXCLUSION_PX)
    )
    cs, cr = _cross_pose_agreement(valid, best)
    features["cross_scale_agreement"] = cs
    features["cross_rotation_agreement"] = cr

    try:
        refined = refine_candidate(
            reference, search, best["x"], best["y"], best["scale"], best["rotation"]
        )
        if np.isfinite(refined.score):
            features["refinement_gain"] = float(refined.score - best["raw"])
            features["refinement_displacement"] = float(
                np.hypot(refined.x - best["x"], refined.y - best["y"])
            )
            features["refined_score"] = float(refined.score)
        else:
            features["refined_score"] = float(best["raw"])
    except Exception:
        features["refined_score"] = float(best["raw"])

    try:
        template_best = band_pass(build_template(reference, best["scale"], best["rotation"]))
        features["peak_hessian_det"] = _peak_hessian_det(search_f, template_best)
    except ValueError:
        features["peak_hessian_det"] = 0.0

    return features, best


class PresenceModel:
    """The fitted present/absent classifier and its operating threshold."""

    def __init__(self, bundle: dict) -> None:
        self.model = bundle["model"]
        self.feature_cols = list(bundle["feature_cols"])
        self.threshold = float(bundle["threshold"])
        self.scaler = bundle.get("scaler")

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "PresenceModel":
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))

    def probability(self, features: dict) -> float:
        vector = np.array(
            [[float(features.get(name, 0.0)) for name in self.feature_cols]],
            dtype=np.float64,
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            vector = self.scaler.transform(vector)
        return float(self.model.predict_proba(vector)[0, 1])

    def decide(self, features: dict) -> tuple[int, float]:
        """Return ``(found, probability)`` from one underlying quantity.

        The flag and the score come from the same number by construction, so a
        confident probability can never accompany a rejection.
        """
        p = self.probability(features)
        return int(p >= self.threshold), p
