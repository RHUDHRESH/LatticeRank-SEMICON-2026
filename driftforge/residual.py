"""Periodic-background cancellation -> unique-residual matching.

The measured failure mechanism of every ranker so far: correlation is
dominated by the semiconductor lattice, which every periodic alias shares.
The latent audit showed the aliases ARE physically distinguishable (median
latent NCC 0.86) - by exactly the content the lattice does NOT repeat:
missing contacts, line-edge roughness, width variations, local defects.

So instead of learning which periodic peak is right, cancel the periodic
component and match what remains:

    periodic(img) = median of the 8 copies shifted by +-v1, +-v2, +-(v1+v2),
                    +-(v1-v2)   (the unshifted image is excluded)
    residual(img) = img - periodic(img)

A pixel that looks the same one lattice period away contributes ~0; a pixel
where THIS location differs from its periodic neighbours survives. The same
stack gives a per-pixel uniqueness weight for the template,

    uniqueness = std(shifted template stack, axis=0)

so matching can be restricted to the pixels that actually identify the
physical site. Scoring is dense weighted ZNCC (3 FFT correlations per
channel/mask), evaluated at existing adaptive candidates with a +-3 px local
max - the residual never relocates a candidate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

from .lattice import estimate_lattice

SHIFT_SET = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))
LOCAL_R = 3   # +-px window when reading a dense map at a candidate


def lattice_shift_stack(img: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                        order: int = 3) -> np.ndarray:
    """The 8 lattice-translated copies of `img` (unshifted copy excluded)."""
    a = img.astype(np.float32)
    out = []
    for m, n in SHIFT_SET:
        dx, dy = m * v1[0] + n * v2[0], m * v1[1] + n * v2[1]
        # shift(output)(y,x) = input(y-dy, x-dx): content moves BY (dx,dy)
        out.append(ndimage.shift(a, (dy, dx), order=order, mode="nearest"))
    return np.stack(out)


def periodic_residual(img: np.ndarray, v1: np.ndarray, v2: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(residual, uniqueness). uniqueness = per-pixel std of the shift stack."""
    stack = lattice_shift_stack(img, v1, v2)
    periodic = np.median(stack, axis=0)
    resid = img.astype(np.float32) - periodic
    return resid.astype(np.float32), stack.std(axis=0).astype(np.float32)


def weighted_zncc_valid(image: np.ndarray, template: np.ndarray,
                        weight: np.ndarray) -> np.ndarray:
    """Dense weighted ZNCC in valid coordinates (3 FFT correlations)."""
    w = np.maximum(weight.astype(np.float64), 0.0)
    W = float(w.sum())
    if W < 1e-6:
        w = np.ones_like(w); W = float(w.sum())
    t = template.astype(np.float64)
    tc = t - float((w * t).sum()) / W
    den_t = math.sqrt(float((w * tc * tc).sum()))
    if den_t < 1e-9:
        return np.zeros((image.shape[0] - t.shape[0] + 1,
                         image.shape[1] - t.shape[1] + 1), dtype=np.float32)
    I = image.astype(np.float64)
    wf = w[::-1, ::-1]
    num = signal.fftconvolve(I, (w * tc)[::-1, ::-1], mode="valid")
    mu = signal.fftconvolve(I, wf, mode="valid") / W
    energy = signal.fftconvolve(I * I, wf, mode="valid") - W * mu * mu
    den = np.sqrt(np.maximum(energy, 1e-9) * den_t * den_t / W) * math.sqrt(W)
    return (num / np.maximum(den, 1e-9)).astype(np.float32)


def local_max_at(resp: np.ndarray, x: float, y: float,
                 half_w: float, half_h: float, r: int = LOCAL_R) -> float:
    """Max of a valid-coordinate response map within +-r px of (x, y)."""
    c = int(round(x - half_w)); rr = int(round(y - half_h))
    y0, y1 = max(rr - r, 0), min(rr + r + 1, resp.shape[0])
    x0, x1 = max(c - r, 0), min(c + r + 1, resp.shape[1])
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(resp[y0:y1, x0:x1].max())


class ResidualMatcher:
    """Per-scene residual evidence: 2 channels x 3 uniqueness masks, dense."""

    #: (channel, mask) keys in the fixed evaluation order
    KEYS = [f"res_{c}_{m}" for c in ("int", "grad") for m in ("m0", "m50", "m30")]

    def __init__(self, reference: np.ndarray, search: np.ndarray):
        from .baseline import _robust_contrast, _template_from_reference
        sf = _robust_contrast(search)
        t = _template_from_reference(reference, 1.0, 0.0)
        lat = estimate_lattice(search)
        B = lat.basis
        v1, v2 = B[:, 0].copy(), B[:, 1].copy()
        self.lattice_ok = (abs(float(np.linalg.det(B))) > 1.0
                           and np.linalg.norm(v1) >= 2.0 and np.linalg.norm(v2) >= 2.0)
        # read-out window must stay under half the smallest period, or a
        # candidate one cell away steals the true site's residual peak
        pmin = min(np.linalg.norm(v1), np.linalg.norm(v2))
        self.local_r = int(np.clip((pmin - 1.0) // 2, 1, LOCAL_R))
        self.maps: dict[str, np.ndarray] = {}
        th, tw = t.shape
        if not self.lattice_ok:
            self.half_w = (tw - 1) / 2.0
            self.half_h = (th - 1) / 2.0
            return

        s_res, _ = periodic_residual(sf, v1, v2)
        t_res, t_unique = periodic_residual(t, v1, v2)

        # trim the template border the shift stack could not populate honestly
        margin = int(math.ceil(max(abs(m * v1[i] + n * v2[i])
                                   for m, n in SHIFT_SET for i in (0, 1)))) + 1
        margin = min(margin, (min(th, tw) - 16) // 2)
        tc = t_res[margin:th - margin, margin:tw - margin]
        uc = t_unique[margin:th - margin, margin:tw - margin]
        # cropping is symmetric, so the map's centre convention shifts by margin
        self.half_w = (tw - 1) / 2.0 - margin
        self.half_h = (th - 1) / 2.0 - margin

        gy, gx = ndimage.sobel(s_res, axis=0), ndimage.sobel(s_res, axis=1)
        s_grad = np.hypot(gx, gy).astype(np.float32)
        gy, gx = ndimage.sobel(tc, axis=0), ndimage.sobel(tc, axis=1)
        t_grad = np.hypot(gx, gy).astype(np.float32)

        masks = {"m0": np.ones_like(uc),
                 "m50": (uc > np.percentile(uc, 50)).astype(np.float64),
                 "m30": (uc > np.percentile(uc, 70)).astype(np.float64)}
        for mk, w in masks.items():
            self.maps[f"res_int_{mk}"] = weighted_zncc_valid(s_res, tc, w)
            self.maps[f"res_grad_{mk}"] = weighted_zncc_valid(s_grad, t_grad, w)

    def score(self, x: float, y: float) -> dict[str, float]:
        if not self.lattice_ok:
            return {k: float("nan") for k in self.KEYS}
        return {k: local_max_at(self.maps[k], x, y, self.half_w, self.half_h,
                                r=self.local_r)
                for k in self.KEYS}
