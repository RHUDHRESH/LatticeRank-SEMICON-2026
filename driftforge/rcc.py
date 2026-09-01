"""Sparse Residual Constellation Consensus -- geometric agreement as evidence.

The structural descriptor already partitions a candidate patch into blocks and
scores each one, but every block is read at **one** globally-chosen tiny shift
(``structural_descriptor`` picks ``dx0, dy0`` once and applies it everywhere).
So the descriptor measures whether the pieces *look* alike; it cannot measure
whether they *agree on where they are*. That distinction is the whole content
of the periodic-alias failure: a wrong lattice sibling can score respectable
block NCCs while each block's true displacement points somewhere different,
because nothing in the patch actually registers.

RCC restores that missing axis. A handful of non-periodic witnesses are chosen
from the reference, each is allowed to find its **own** displacement, and the
evidence reported is the *agreement* of those displacements rather than their
photometric quality:

    d_j        best local displacement of anchor j
    d*         robust (component-wise median) displacement
    residual_j ||d_j - d*||
    consensus  fraction(residual_j <= tau)

A true site produces a tight cluster of displacements; an alias produces a
scatter. Both can produce similar mean NCC.

Cost is the reason this is sparse rather than dense. The existing residual
stage builds six full-image weighted-ZNCC maps (three FFT correlations each);
RCC reads ``n_anchors * n_displacements * n_pixels`` samples per candidate --
about 4,800 for the default 12 anchors, 5x5 window and 16 pixels. Evidence is
concentrated where it discriminates instead of being averaged over a field that
is, by construction, identical everywhere.

Anchor quality deliberately multiplies two independent requirements::

    quality = uniqueness * gradient_energy * border_validity

``uniqueness`` (the std of the lattice-shifted stack, from
:func:`driftforge.residual.periodic_residual`) says this location differs from
its periodic copies. Gradient energy says it carries enough measurable
structure to localize at all -- a witness with no gradient cannot testify about
displacement however unique it is. Spatial NMS then spreads the anchors across
the template, because twelve witnesses to the same defect are one witness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: Anchors kept per template. Twelve is the v0 setting: enough that a robust
#: median survives several destroyed witnesses, small enough to stay cheap.
N_ANCHORS = 12

#: Half-width of the square patch each anchor draws its pixels from.
ANCHOR_HALF = 7

#: Sampling stride inside the anchor patch. Pixels are taken on a regular
#: subgrid, **not** as the top-N by quality. Quality decides *where an anchor
#: sits*; it must not decide which pixels inside it are read. A scattered
#: top-N set has almost no spatial specificity -- shifting it by one pixel
#: resamples unrelated residual noise -- so its NCC-vs-displacement surface is
#: flat and the argmax is random. Measured directly: with 16 quality-picked
#: pixels the displacement p90 pinned to sqrt(5), the corner of the +/-2
#: window, on true and false sites alike. A coherent subgrid keeps the patch
#: geometrically identifiable while still reading a fraction of the pixels.
ANCHOR_STRIDE = 2

#: Per-anchor displacement search half-width, in search pixels. Must leave the
#: peak room to sit strictly inside the window; a peak on the boundary is
#: indistinguishable from an unconverged one.
DISP_RADIUS = 3

#: Minimum separation between anchor centres, in template pixels.
MIN_SEPARATION = 12

#: Consensus radii (px) reported as separate features.
CONSENSUS_TAUS = (0.75, 1.5, 2.5)

FEATURES = [
    "rcc_n_anchors", "rcc_n_valid",
    "rcc_ncc_median", "rcc_ncc_p25", "rcc_ncc_min",
    "rcc_consensus_0p75", "rcc_consensus_1p5", "rcc_consensus_2p5",
    "rcc_disp_mad", "rcc_disp_p90",
    "rcc_dx_median", "rcc_dy_median",
    "rcc_peak_margin_median",
]

NAN_FEATURES = {k: float("nan") for k in FEATURES}


def _gradient_energy(img: np.ndarray) -> np.ndarray:
    gy = ndimage.sobel(img, axis=0)
    gx = ndimage.sobel(img, axis=1)
    return np.hypot(gx, gy).astype(np.float32)


@dataclass
class Anchors:
    """Sparse witnesses drawn from one template, in template coordinates.

    ``dr``/``dc`` are per-pixel offsets from the template centre, so a
    candidate at ``(x, y)`` predicts search coordinates ``(x + dc, y + dr)``.
    ``values`` holds the zero-meaned, unit-norm template residual at those
    pixels, which makes the per-displacement NCC a single dot product.
    """

    dr: np.ndarray        # (n_anchors, n_pixels) int32
    dc: np.ndarray        # (n_anchors, n_pixels) int32
    values: np.ndarray    # (n_anchors, n_pixels) float32, zero-mean unit-norm
    centres: np.ndarray   # (n_anchors, 2) float64, (dx, dy) from centre

    @property
    def n(self) -> int:
        return int(self.dr.shape[0])


def select_anchors(
    t_res: np.ndarray,
    t_unique: np.ndarray,
    *,
    n_anchors: int = N_ANCHORS,
    half: int = ANCHOR_HALF,
    stride: int = ANCHOR_STRIDE,
    min_separation: int = MIN_SEPARATION,
    border: int = 0,
) -> Anchors | None:
    """Choose spread-out, high-uniqueness, high-gradient witnesses.

    ``border`` excludes a margin the lattice shift stack could not populate
    honestly; anchors there would testify about reflected padding.
    """
    if t_res.shape != t_unique.shape:
        raise ValueError("residual and uniqueness maps must have one shape")
    th, tw = t_res.shape
    pad = int(max(border, half))
    if th - 2 * pad < 1 or tw - 2 * pad < 1:
        return None

    grad = _gradient_energy(t_res)
    quality = t_unique.astype(np.float32) * grad
    if not np.isfinite(quality).any():
        return None
    quality = np.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)

    valid = np.zeros_like(quality, dtype=bool)
    valid[pad:th - pad, pad:tw - pad] = True
    work = np.where(valid, quality, -1.0)

    # Greedy NMS: strongest surviving location, then suppress its neighbourhood.
    picks: list[tuple[int, int]] = []
    for _ in range(n_anchors):
        idx = int(np.argmax(work))
        r, c = divmod(idx, tw)
        if work[r, c] <= 0.0:
            break
        picks.append((r, c))
        r0, r1 = max(0, r - min_separation), min(th, r + min_separation + 1)
        c0, c1 = max(0, c - min_separation), min(tw, c + min_separation + 1)
        work[r0:r1, c0:c1] = -1.0
    if len(picks) < 3:
        return None

    cy, cx = (th - 1) / 2.0, (tw - 1) / 2.0
    dr_rows, dc_rows, val_rows, centres = [], [], [], []
    for r, c in picks:
        rs = slice(r - half, r + half + 1)
        cs = slice(c - half, c + half + 1)
        patch_v = t_res[rs, cs]
        # regular subgrid, so the sampled set stays spatially coherent
        pr, pc = np.meshgrid(np.arange(0, patch_v.shape[0], stride),
                             np.arange(0, patch_v.shape[1], stride),
                             indexing="ij")
        pr, pc = pr.ravel(), pc.ravel()
        vals = patch_v[pr, pc].astype(np.float32)
        vals = vals - vals.mean()
        norm = float(np.linalg.norm(vals))
        if not np.isfinite(norm) or norm < 1e-6:
            continue                       # a flat witness cannot testify
        dr_rows.append((r - half + pr) - cy)
        dc_rows.append((c - half + pc) - cx)
        val_rows.append(vals / norm)
        centres.append((c - cx, r - cy))

    if len(val_rows) < 3:
        return None
    # Ragged rows are impossible here (take is constant unless a patch is
    # clipped, which `pad` prevents), so a plain stack is safe.
    return Anchors(
        dr=np.rint(np.stack(dr_rows)).astype(np.int32),
        dc=np.rint(np.stack(dc_rows)).astype(np.int32),
        values=np.stack(val_rows).astype(np.float32),
        centres=np.asarray(centres, dtype=np.float64),
    )


class ConstellationScorer:
    """Scene-level context: one search residual, one set of anchors.

    Building this is the only per-scene cost. Scoring a candidate is pure
    gathering and dot products, which is what makes RCC affordable on a
    shortlist of several hundred sites.
    """

    def __init__(
        self,
        search_residual: np.ndarray,
        anchors: Anchors,
        *,
        disp_radius: int = DISP_RADIUS,
    ):
        self.res = np.ascontiguousarray(search_residual, dtype=np.float32)
        self.anchors = anchors
        self.radius = int(disp_radius)
        span = np.arange(-self.radius, self.radius + 1, dtype=np.int32)
        # (n_disp,) offsets, ordered so index 0..n-1 maps to (dy, dx) pairs
        self.off_dy, self.off_dx = (a.ravel() for a in np.meshgrid(span, span, indexing="ij"))
        self.n_disp = int(self.off_dy.size)

    def score(self, x: float, y: float) -> dict[str, float]:
        """Per-anchor displacements at one candidate, reduced to agreement."""
        a = self.anchors
        h, w = self.res.shape
        xi, yi = int(round(x)), int(round(y))

        # (n_anchors, n_disp, n_pixels) sample coordinates
        rows = a.dr[:, None, :] + self.off_dy[None, :, None] + yi
        cols = a.dc[:, None, :] + self.off_dx[None, :, None] + xi
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        np.clip(rows, 0, h - 1, out=rows)
        np.clip(cols, 0, w - 1, out=cols)

        samples = self.res[rows, cols]
        samples = np.where(inside, samples, np.float32(0.0))
        # An anchor is only usable if every one of its pixels is in frame at
        # every displacement; a partially-clipped witness would compare
        # different pixel sets across the window and its argmax would be noise.
        usable = inside.all(axis=(1, 2))

        samples = samples - samples.mean(axis=2, keepdims=True)
        norms = np.linalg.norm(samples, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            ncc = np.einsum("adp,ap->ad", samples, a.values) / norms
        ncc = np.where(np.isfinite(ncc), ncc, -1.0)

        best = np.argmax(ncc, axis=1)
        rows_i = np.arange(a.n)
        best_ncc = ncc[rows_i, best]
        dy = self.off_dy[best].astype(np.float64)
        dx = self.off_dx[best].astype(np.float64)

        # Peak margin: best minus the best displacement at least 1.5 px away,
        # so a broad plateau reads as weak evidence even at a high peak.
        far = (np.hypot(self.off_dy[None, :] - dy[:, None],
                        self.off_dx[None, :] - dx[:, None]) >= 1.5)
        runner = np.where(far, ncc, -np.inf).max(axis=1)
        margin = np.where(np.isfinite(runner), best_ncc - runner, np.nan)

        keep = usable & np.isfinite(best_ncc) & (best_ncc > -1.0)
        n_valid = int(keep.sum())
        out = dict(NAN_FEATURES)
        out["rcc_n_anchors"] = float(a.n)
        out["rcc_n_valid"] = float(n_valid)
        if n_valid < 3:
            return out

        dxk, dyk, nck = dx[keep], dy[keep], best_ncc[keep]
        dx_med, dy_med = float(np.median(dxk)), float(np.median(dyk))
        resid = np.hypot(dxk - dx_med, dyk - dy_med)

        out.update({
            "rcc_ncc_median": float(np.median(nck)),
            "rcc_ncc_p25": float(np.percentile(nck, 25)),
            "rcc_ncc_min": float(nck.min()),
            "rcc_consensus_0p75": float(np.mean(resid <= CONSENSUS_TAUS[0])),
            "rcc_consensus_1p5": float(np.mean(resid <= CONSENSUS_TAUS[1])),
            "rcc_consensus_2p5": float(np.mean(resid <= CONSENSUS_TAUS[2])),
            "rcc_disp_mad": float(np.median(resid)),
            "rcc_disp_p90": float(np.percentile(resid, 90)),
            "rcc_dx_median": dx_med,
            "rcc_dy_median": dy_med,
            "rcc_peak_margin_median": float(np.nanmedian(margin[keep]))
            if np.isfinite(margin[keep]).any() else float("nan"),
        })
        return out


def build_scorer(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    n_anchors: int = N_ANCHORS,
    disp_radius: int = DISP_RADIUS,
) -> ConstellationScorer | None:
    """Assemble a scorer for one (reference, search) pair at one pose.

    Reuses exactly the periodic cancellation and lattice estimation the
    declared method already performs -- RCC changes how the residual is *read*,
    not how it is produced.
    """
    import math

    from .baseline import _robust_contrast, _template_from_reference
    from .lattice import estimate_lattice
    from .residual import SHIFT_SET, periodic_residual

    sf = _robust_contrast(search)
    t = _template_from_reference(reference, scale, rotation)
    lat = estimate_lattice(search)
    B = lat.basis
    v1, v2 = B[:, 0].copy(), B[:, 1].copy()
    if not (abs(float(np.linalg.det(B))) > 1.0
            and np.linalg.norm(v1) >= 2.0 and np.linalg.norm(v2) >= 2.0):
        return None

    s_res, _ = periodic_residual(sf, v1, v2)
    t_res, t_unique = periodic_residual(t, v1, v2)
    margin = int(math.ceil(max(abs(m * v1[i] + n * v2[i])
                               for m, n in SHIFT_SET for i in (0, 1)))) + 1

    anchors = select_anchors(t_res, t_unique, n_anchors=n_anchors, border=margin)
    if anchors is None:
        return None
    return ConstellationScorer(s_res, anchors, disp_radius=disp_radius)
