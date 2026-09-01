"""Robust local translation field by iteratively reweighted least squares.

Experiment #16: Upgrade RCC's per-anchor displacement aggregation from
component-wise median/MAD to a proper robust estimator.

RATIONALE
---------
Repeated patterns (the semiconductor lattice) generate correspondence errors:
every periodic alias shares the same lattice shifts and can produce similar
block NCCs. The distinguishing signal lies in whether a set of correspondences
supports ONE transformation consistently. A few anchors destroyed by charging
or a local defect must not destroy the estimate—that is the nature of robust
fitting. RCC already identifies which anchors can testify (ConstellationScorer
samples each anchor's NCC-vs-displacement); this module upgrades the consensus
step by fitting the true underlying signal rather than taking medians.

ALGORITHM
---------
For each candidate (x, y), collect per-anchor displacements d_j = (dx_j, dy_j).
Fit the 2D rigid translation t = (tx, ty) such that d_j ≈ t + eps_j using
Iteratively Reweighted Least Squares (IRLS) with Huber loss:

    rho(r) = 0.5*r^2                    for |r| <= delta
    rho(r) = delta*(|r| - 0.5*delta)    otherwise

Weight at iteration k:
    w_j^k = rho'(r_j^k) / r_j^k        (Huber's derivative / magnitude)

The scale parameter delta is set adaptively based on the MAD (Median Absolute
Deviation) of the residuals, which is robust to outliers. This avoids
hard-coding a pixel threshold and ensures the fit adapts to the displacement
scale of the scene.

FEATURES (all prefixed irls_)
-----------------------------
- irls_tx, irls_ty: fitted 2D translation
- irls_scale: robust scale estimate (MAD-based)
- irls_inlier_fraction: fraction of anchors with |residual| <= 2*scale
- irls_residual_median, irls_residual_p90: percentiles of per-anchor residuals
- irls_max_residual: maximum residual magnitude
- irls_iterations: number of IRLS iterations until convergence
- irls_n_anchors: number of usable anchors (before outlier filtering)
"""
from __future__ import annotations

import numpy as np

FEATURES = [
    "irls_tx",
    "irls_ty",
    "irls_scale",
    "irls_inlier_fraction",
    "irls_residual_median",
    "irls_residual_p90",
    "irls_max_residual",
    "irls_iterations",
    "irls_n_anchors",
]

NAN_FEATURES = {k: float("nan") for k in FEATURES}


def _mad_scale(residuals: np.ndarray) -> float:
    """Robust scale estimate via Median Absolute Deviation.

    MAD is defined as the median of |x_i - median(x_i)|, and is robust to
    outliers. Multiplied by 1.4826 to match the standard deviation under
    Gaussian noise (consistency factor).
    """
    if residuals.size == 0:
        return 1.0
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    if mad < 1e-6:
        # All residuals clustered; use the spread instead
        return max(np.ptp(residuals) / 4.0, 1e-6)
    return 1.4826 * mad


def _huber_fit(
    displacements: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-4,
    c: float = 1.345,
) -> tuple[np.ndarray, np.ndarray, int]:
    """IRLS with Huber loss for 2D translation fitting.

    Args:
        displacements: (n_anchors, 2) array of per-anchor (dx, dy)
        max_iter: maximum iterations
        tol: convergence tolerance for change in t
        c: Huber tuning constant (default 1.345 for ~95% efficiency vs normal)

    Returns:
        (t, residuals, n_iter) where t is (2,) fitted translation,
        residuals is (n_anchors,) per-anchor residual magnitudes, and
        n_iter is the number of iterations performed.
    """
    n = displacements.shape[0]
    if n < 2:
        return np.array([0.0, 0.0]), np.zeros(n), 0

    # Initialize with componentwise median
    t = np.median(displacements, axis=0).astype(np.float64)

    for iteration in range(max_iter):
        t_old = t.copy()

        # Compute residuals
        eps = displacements - t[None, :]  # (n, 2)
        resid_mag = np.linalg.norm(eps, axis=1)  # (n,)

        # Estimate scale robustly
        scale = _mad_scale(resid_mag)
        delta = c * scale

        # Huber weighting: w_i = rho'(r_i) / r_i
        # rho'(r) = r for |r| <= delta, delta * sign(r) for |r| > delta
        abs_resid = np.abs(resid_mag)
        weights = np.where(
            abs_resid <= delta,
            1.0,
            delta / np.maximum(abs_resid, 1e-9)
        )

        # Weighted least squares: solve argmin_t sum(w_i * ||d_i - t||_2^2)
        # For constant fit, the solution is the weighted mean:
        # (sum w_i) * t = sum(w_i * d_i)
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            break
        t = (weights[:, None] * displacements).sum(axis=0) / w_sum

        # Check convergence
        if np.linalg.norm(t - t_old) < tol:
            break

    # Final residual computation
    eps_final = displacements - t[None, :]
    resid_final = np.linalg.norm(eps_final, axis=1)

    return t, resid_final, iteration + 1


def build(ctx) -> dict | None:
    """Build per-scene state: gather anchors and configure for scoring.

    Reuses exactly the anchor selection from RCC, ensuring spatial coherence
    and proper witness diversity. Returns None if fewer than 3 usable anchors
    are available (too few for meaningful robust estimation).
    """
    from ..rcc import select_anchors, ConstellationScorer

    if not ctx.lattice_ok:
        return None

    # Reuse RCC's anchor selection (guaranteed spatially coherent subgrid)
    anchors = select_anchors(
        ctx.t_res,
        ctx.t_unique,
        border=ctx.margin,
    )
    if anchors is None or anchors.n < 3:
        return None

    # Build the scorer to read per-anchor displacements
    scorer = ConstellationScorer(ctx.s_res, anchors)

    return {
        "scorer": scorer,
        "anchors": anchors,
    }


def score(state: dict, x: float, y: float) -> dict[str, float]:
    """Score one candidate with robust IRLS translation fitting.

    Collects per-anchor displacements, fits a robust 2D translation,
    and reports the fit parameters and residual statistics.
    """
    if state is None:
        return NAN_FEATURES.copy()

    scorer = state["scorer"]
    anchors = state["anchors"]

    # Get per-anchor displacements (using ConstellationScorer's machinery)
    a = anchors
    h, w = scorer.res.shape
    xi, yi = int(round(x)), int(round(y))

    # (n_anchors, n_disp, n_pixels) sample coordinates
    rows = a.dr[:, None, :] + scorer.off_dy[None, :, None] + yi
    cols = a.dc[:, None, :] + scorer.off_dx[None, :, None] + xi
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    np.clip(rows, 0, h - 1, out=rows)
    np.clip(cols, 0, w - 1, out=cols)

    samples = scorer.res[rows, cols]
    samples = np.where(inside, samples, np.float32(0.0))
    usable = inside.all(axis=(1, 2))

    samples = samples - samples.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(samples, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = np.einsum("adp,ap->ad", samples, a.values) / norms
    ncc = np.where(np.isfinite(ncc), ncc, -1.0)

    best = np.argmax(ncc, axis=1)
    rows_i = np.arange(a.n)
    best_ncc = ncc[rows_i, best]
    dy = scorer.off_dy[best].astype(np.float64)
    dx = scorer.off_dx[best].astype(np.float64)

    keep = usable & np.isfinite(best_ncc) & (best_ncc > -1.0)
    n_valid = int(keep.sum())

    out = NAN_FEATURES.copy()
    out["irls_n_anchors"] = float(a.n)

    if n_valid < 3:
        return out

    # Collect usable displacements
    dxk, dyk = dx[keep], dy[keep]
    displacements = np.column_stack([dxk, dyk])

    # Fit robust translation
    t, residuals, n_iter = _huber_fit(displacements)

    # Compute scale and features
    scale = _mad_scale(residuals)
    inlier_frac = float(np.mean(residuals <= 2.0 * scale))

    out.update({
        "irls_tx": float(t[0]),
        "irls_ty": float(t[1]),
        "irls_scale": float(scale),
        "irls_inlier_fraction": inlier_frac,
        "irls_residual_median": float(np.median(residuals)),
        "irls_residual_p90": float(np.percentile(residuals, 90)),
        "irls_max_residual": float(np.max(residuals)),
        "irls_iterations": float(n_iter),
    })

    return out
