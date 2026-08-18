"""Lattice estimation for repetitive semiconductor layouts.

Estimates the repeat geometry of a Search image *from the image alone* - no
generator metadata is consulted. To avoid locking onto coarse routing periods
instead of the device lattice, the implementation:

  * work on a HIGH-pass residual, so coarse mat/routing structure is removed;
  * bound the period search to the physically possible device range;
  * score a candidate period by summing its harmonics, so the fundamental wins
    even when the 2nd harmonic is individually stronger;
  * cross-check the spectral estimate against autocorrelation, and report a
    confidence that reflects agreement rather than raw peak height.

Contacts sit on a checkerboard in both DRAM and FinFET layouts, so the repeat
that preserves the *full* pattern is the centred-rectangular lattice generated
by (2*pitch_x, 0) and (pitch_x, pitch_y) - only translations with (m+n) even
preserve the contact sub-lattice. `Lattice.translations()` enforces that.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

#: Physically possible device pitch at 10 nm/px, covering every DriftForge
#: preset (2.8-17 px) and every sponsor preset (4.0-26 px) with margin.
PERIOD_MIN_PX = 2.5
PERIOD_MAX_PX = 32.0


@dataclass
class Lattice:
    pitch_x: float
    pitch_y: float
    orientation_deg: float
    confidence_x: float
    confidence_y: float
    agreement: float           # spectral vs autocorrelation agreement, 0..1
    parity_aware: bool

    @property
    def confidence(self) -> float:
        """Single gate value: weakest axis, discounted by cross-method disagreement."""
        return float(min(self.confidence_x, self.confidence_y) * self.agreement)

    @property
    def basis(self) -> np.ndarray:
        t = np.deg2rad(self.orientation_deg)
        a = np.array([self.pitch_x * np.cos(t), self.pitch_x * np.sin(t)])
        b = np.array([-self.pitch_y * np.sin(t), self.pitch_y * np.cos(t)])
        return np.column_stack([a, b])

    def translations(self, max_m: int = 6, max_n: int = 6,
                     parity: bool | None = None, step: float = 0.5) -> np.ndarray:
        """Lattice displacement vectors (px), excluding (0,0).

        parity=True keeps only (m+n) even - the translations that preserve a
        checkerboard contact sub-lattice.

        `step` defaults to 0.5 because the estimator legitimately returns
        2x the line pitch when contacts sit on a checkerboard (the doubled
        value IS the fundamental period of the full pattern). Integer steps on
        a doubled basis would silently skip every intermediate valid site, so
        half-steps are emitted and deduplicated by the caller. This trades
        candidate count for recall, which is the correct direction: a site the
        proposer never emits can never be recovered downstream.
        """
        if parity is None:
            parity = self.parity_aware
        B = self.basis
        ks = np.arange(-max_m, max_m + step / 2, step)
        ls = np.arange(-max_n, max_n + step / 2, step)
        out = []
        for m in ks:
            for n in ls:
                if m == 0 and n == 0:
                    continue
                if parity and step >= 1.0 and (int(m) + int(n)) % 2 != 0:
                    continue
                out.append(B @ np.array([float(m), float(n)]))
        return np.asarray(out) if out else np.zeros((0, 2))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = self.confidence
        return d


def _prep(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    lo, hi = np.percentile(a, (0.5, 99.5))
    a = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)
    # High-pass: strip anything coarser than ~2x the largest device pitch.
    return a - ndimage.gaussian_filter(a, sigma=12.0, mode="reflect")


def _axis_from_spectrum(profile: np.ndarray, length: int) -> tuple[float, float]:
    half = profile[: length // 2]
    periods = np.arange(PERIOD_MIN_PX, PERIOD_MAX_PX, 0.05)
    scores = np.empty(periods.size)
    for i, p in enumerate(periods):
        s = 0.0
        for harm in (1, 2, 3):
            k = harm * length / p
            if k >= len(half) - 1:
                break
            k0 = int(np.floor(k))
            frac = k - k0
            s += (half[k0] * (1 - frac) + half[k0 + 1] * frac) / harm
        scores[i] = s
    k = int(np.argmax(scores))
    conf = float(scores[k] / (np.median(scores) + 1e-30))
    return float(periods[k]), conf


def _axis_from_autocorr(sig: np.ndarray) -> float:
    """Dominant period from the 1-D autocorrelation of a projection."""
    x = sig - sig.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    lo, hi = int(PERIOD_MIN_PX), int(PERIOD_MAX_PX) + 1
    seg = ac[lo:hi]
    if seg.size < 3:
        return float("nan")
    k = int(np.argmax(seg)) + lo
    # parabolic refinement on the autocorrelation peak
    if 0 < k < n - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return float(k + 0.5 * (a - c) / den)
    return float(k)


def _orientation(hp: np.ndarray) -> float:
    """Dominant Manhattan orientation via the doubled-angle gradient mean.

    Layouts here are Manhattan, so structure orientation is only defined mod
    90 deg; we report the small residual tilt in (-45, 45].
    """
    gy = ndimage.sobel(hp, axis=0, mode="reflect")
    gx = ndimage.sobel(hp, axis=1, mode="reflect")
    m = np.hypot(gx, gy)
    keep = m > np.percentile(m, 80)          # strong edges only
    if keep.sum() < 50:
        return 0.0
    th = np.arctan2(gy[keep], gx[keep])
    # 4-fold symmetry -> average exp(4i*theta)
    z = np.exp(4j * th).mean()
    ang = np.angle(z) / 4.0
    return float(np.rad2deg(ang))


def estimate_lattice(search: np.ndarray, parity_aware: bool = True) -> Lattice:
    """Estimate the repeat geometry of `search` from the image alone."""
    hp = _prep(search)
    h, w = hp.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    P = np.abs(np.fft.fft2(hp * win)) ** 2

    px_s, cx = _axis_from_spectrum(P.sum(axis=0), w)   # horizontal freq -> x pitch
    py_s, cy = _axis_from_spectrum(P.sum(axis=1), h)   # vertical  freq -> y pitch

    # Independent check: autocorrelation of the axis projections.
    px_a = _axis_from_autocorr(hp.mean(axis=0))
    py_a = _axis_from_autocorr(hp.mean(axis=1))

    def agree(a: float, b: float) -> float:
        """1.0 when the two methods agree on the fundamental or a 2x harmonic."""
        if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
            return 0.5
        r = max(a, b) / min(a, b)
        for target in (1.0, 2.0, 3.0):
            if abs(r - target) < 0.10 * target:
                return 1.0 if target == 1.0 else 0.75
        return 0.35

    agreement = 0.5 * (agree(px_s, px_a) + agree(py_s, py_a))
    return Lattice(pitch_x=px_s, pitch_y=py_s, orientation_deg=_orientation(hp),
                   confidence_x=cx, confidence_y=cy, agreement=float(agreement),
                   parity_aware=parity_aware)
