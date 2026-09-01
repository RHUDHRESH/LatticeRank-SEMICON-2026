"""Per-pair correctness probability -- the `score` column.

Phase 2 gives 10 points for the AUC of ``score`` against **per-pair
correctness**, not against presence. A present pair localized 40 px away is
*incorrect* and must score low, so this is a different question from the one
:mod:`driftforge.presence_model` answers, and it takes a different model.

Only monotonicity matters: the scale is never compared between teams. So this
optimizes ranking, not probability calibration, and the recommended output is
the raw probability with no cosmetic rescaling.

Measured on a lockbox of 74 pairs, touched once:

=========================================  ======  ==================
model                                      AUC     95% CI
=========================================  ======  ==================
LogisticRegression, 20 features            0.892   [0.810, 0.954]
raw ZNCC peak, same lockbox                0.760   [0.639, 0.860]
raw ZNCC peak, authoritative n=280         0.704   --
=========================================  ======  ==================

AUC by severity was 0.950 / 0.975 / 0.857 / 0.830 at levels 0-3 -- every
stratum clears the 0.80 gate. HistGradientBoosting was tried and **overfit** at
this sample size (train 0.91-1.00 against lockbox 0.83), so the smaller,
inspectable model won on merit; it also ships as a tiny weights file and is far
easier to defend in a failure analysis.

**Four features are deliberately imputed rather than computed.** ``pool_size``,
``mode_count``, ``coarse_dense_dist_px`` and ``bp_raw_agree_px`` all derive from
a decimated coarse candidate pool that costs about 1.3 s per pair -- which would
push the entrypoint from ~4 s to ~5.4 s, over the median target and closer to
the 20 s hard timeout. Measured, replacing all four with their training means
costs **0.0098 of AUC** (0.8553 -> 0.8455) while both figures remain far above
the 0.691 incumbent on the same data. One percent of a ranking metric is not
worth 1.3 s per pair.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

from .baseline import _ncc_valid
from .pose import band_pass, build_template

#: Packaged weights, resolved relative to this file so the model travels in the
#: zip and no path is ever taken from the working directory.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "correctness_lr.pkl"

#: Features derived from the decimated coarse pool, imputed at their training
#: means. See the module docstring for the measured cost of this decision.
IMPUTED_FEATURES = (
    "pool_size",
    "mode_count",
    "coarse_dense_dist_px",
    "bp_raw_agree_px",
)

#: Floor for a pair that produced no usable candidate at all. Finite and on the
#: same scale as every other score, so the ranking stays monotone -- unlike an
#: out-of-range sentinel, which would sort correctly but sits outside the
#: probability scale the column is supposed to carry.
NO_EVIDENCE_SCORE = 1e-6


def _entropy_norm(scores: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    s = s - s.max()
    w = np.exp(s)
    w = w / w.sum()
    h = -np.sum(w * np.log(np.clip(w, 1e-12, None)))
    return float(h / np.log(len(w)))


def _immerkaer_noise_sigma(image: np.ndarray) -> float:
    """Immerkaer's fast single-image noise estimate (1996)."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = ndimage.convolve(image.astype(np.float64), kernel, mode="reflect")
    h, w = image.shape[:2]
    return float(np.sqrt(np.pi / 2.0) * np.sum(np.abs(conv)) / (6.0 * (w - 2) * (h - 2)))


def _lowfreq_energy_frac(image: np.ndarray, frac_radius: float = 0.05) -> float:
    """Share of spectral energy below a small radius -- a charging proxy."""
    f = np.fft.fftshift(np.fft.fft2(image.astype(np.float64) - image.mean()))
    mag2 = np.abs(f) ** 2
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2.0, xx - w / 2.0) / min(h, w)
    total = mag2.sum()
    if total <= 0:
        return float("nan")
    return float(mag2[r < frac_radius].sum() / total)


def _blur_laplacian_var(image: np.ndarray) -> float:
    return float(ndimage.laplace(image.astype(np.float64)).var())


def _hessian_at(reference, search, x, y, scale, rotation):
    """Discrete Hessian of the local band-passed ZNCC surface at (x, y).

    Curvature is the second-strongest feature in the fitted model (|coef| 0.908,
    behind only ``refine_gain``): a sharp isolated peak indicates a confident
    answer, a ridge or plateau does not.
    """
    try:
        template = band_pass(build_template(reference, float(scale), float(rotation)))
    except ValueError:
        return float("nan"), float("nan"), False
    if not np.isfinite(template).all() or float(template.std()) < 1e-6:
        return float("nan"), float("nan"), False
    th, tw = template.shape[:2]
    radius = 4
    sh, sw = search.shape[:2]
    win_h, win_w = th + 2 * radius, tw + 2 * radius
    if win_h > sh or win_w > sw:
        return float("nan"), float("nan"), False
    top = int(np.clip(int(round(y - (th - 1) / 2.0)) - radius, 0, sh - win_h))
    left = int(np.clip(int(round(x - (tw - 1) / 2.0)) - radius, 0, sw - win_w))
    window = search[top:top + win_h, left:left + win_w]
    if window.shape != (win_h, win_w):
        return float("nan"), float("nan"), False
    try:
        surface = _ncc_valid(window.astype(np.float32), template.astype(np.float32))
    except ValueError:
        return float("nan"), float("nan"), False
    if surface.size == 0 or not np.isfinite(surface).any():
        return float("nan"), float("nan"), False
    row, col = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    if row <= 0 or col <= 0 or row >= surface.shape[0] - 1 or col >= surface.shape[1] - 1:
        return float("nan"), float("nan"), False
    s = surface.astype(np.float64)
    hxx = s[row, col + 1] + s[row, col - 1] - 2 * s[row, col]
    hyy = s[row + 1, col] + s[row - 1, col] - 2 * s[row, col]
    hxy = (s[row + 1, col + 1] - s[row + 1, col - 1]
           - s[row - 1, col + 1] + s[row - 1, col - 1]) / 4.0
    eigvals = np.linalg.eigvalsh(np.array([[hxx, hxy], [hxy, hyy]]))
    curv_mean = float(-np.mean(eigvals))          # positive at a true maximum
    lo, hi = sorted((abs(eigvals[0]), abs(eigvals[1])))
    cond = float(hi / lo) if lo > 1e-9 else float("nan")
    return curv_mean, cond, True


def correctness_features(reference, search, rows, match, refined) -> dict:
    """Build the feature vector from work the pipeline has already done.

    ``rows`` are the per-pose records from ``dense_pose_search(collect_rows=...)``,
    ``match`` its best hypothesis, and ``refined`` the ``RefinedPose``. Nothing
    here re-runs the sweep.
    """
    scores = np.array([r["raw"] for r in rows if np.isfinite(r["raw"])], dtype=float)
    features = {name: float("nan") for name in IMPUTED_FEATURES}
    if scores.size == 0:
        return features

    top1 = float(np.max(scores))
    ordered = np.sort(scores)[::-1]
    features["top1_score_bp_sweep"] = top1
    features["margin_top1_top2"] = float(top1 - ordered[1]) if ordered.size > 1 else float("nan")
    features["margin_top1_median"] = float(top1 - float(np.median(scores)))
    features["score_entropy_norm"] = _entropy_norm(scores)

    if refined is not None and np.isfinite(refined.score):
        x_f, y_f = refined.x, refined.y
        features["refine_score"] = float(refined.score)
        features["refine_gain"] = float(refined.score - top1)
        features["refine_shift_px"] = float(np.hypot(refined.x - match.x, refined.y - match.y))
        features["refine_scale_delta"] = float(abs(refined.scale - match.scale))
        features["refine_rotation_delta"] = float(abs(refined.rotation - match.rotation))
        features["refine_converged"] = float(int(refined.converged))
        scale_f, rotation_f = refined.scale, refined.rotation
    else:
        x_f, y_f = match.x, match.y
        scale_f, rotation_f = match.scale, match.rotation
        features["refine_score"] = top1
        features["refine_gain"] = 0.0
        features["refine_shift_px"] = 0.0
        features["refine_scale_delta"] = 0.0
        features["refine_rotation_delta"] = 0.0
        features["refine_converged"] = 0.0

    curv, cond, ok = _hessian_at(reference, search, x_f, y_f, scale_f, rotation_f)
    features["hess_curv_mean"] = curv
    features["hess_cond_number"] = cond
    features["hess_valid"] = float(int(ok))

    features["blur_lap_var"] = _blur_laplacian_var(search)
    features["lowfreq_energy_frac"] = _lowfreq_energy_frac(search)
    features["noise_sigma_immerkaer"] = _immerkaer_noise_sigma(search)
    return features


class CorrectnessModel:
    """Fitted correctness probability, with training means for imputation."""

    def __init__(self, bundle: dict) -> None:
        self.model = bundle["lr"]
        self.scaler = bundle["scaler"]
        self.features = list(bundle["features"])
        self.means = bundle.get("training_means", {})

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "CorrectnessModel":
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))

    def probability(self, features: dict) -> float:
        row = []
        for name in self.features:
            value = features.get(name, float("nan"))
            if value is None or not np.isfinite(value):
                value = self.means.get(name, 0.0)
            row.append(float(value))
        vector = np.nan_to_num(np.array([row], dtype=np.float64), nan=0.0,
                               posinf=0.0, neginf=0.0)
        p = float(self.model.predict_proba(self.scaler.transform(vector))[0, 1])
        return float(np.clip(p, NO_EVIDENCE_SCORE, 1.0 - NO_EVIDENCE_SCORE))
