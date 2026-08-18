"""Three cheap, independent response maps + adaptive candidate harvesting.

The channels provide complementary evidence for periodic scenes:

  raw            ZNCC on robust-contrast intensity
  midband        ZNCC on a sigma3-sigma15 band-pass  (mat/boundary context)
  directionality ZNCC on a polarity-invariant orientation field

The orientation field uses the doubled-angle representation

    m = |grad|,  theta = atan2(gy, gx),  D = (m*cos 2theta, m*sin 2theta)

which is invariant under grad -> -grad. That matters here because SEM edge
brightening lights BOTH sides of a feature, and Reference/Search draw
`edge_strength` (0.14-0.42) and `gamma` (0.82-1.22) from independent streams -
so intensity polarity and contrast differ between the two captures while
orientation does not.

Candidates are then harvested by an ADAPTIVE rule, S_i >= S_max - delta, taken
as a union across channels, rather than a fixed top-K. A fixed K is wrong on
both ends: on a wallpaper tile hundreds of sites are genuinely equivalent, and
on a distinctive tile a handful are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .baseline import _ncc_valid, _robust_contrast, _template_from_reference

CHANNELS = ("raw", "midband", "directionality")


def midband(a: np.ndarray) -> np.ndarray:
    return (ndimage.gaussian_filter(a, 3.0, mode="reflect")
            - ndimage.gaussian_filter(a, 15.0, mode="reflect")).astype(np.float32)


def directionality(a: np.ndarray, smooth: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Polarity-invariant orientation field (Dx, Dy) = m*(cos 2t, sin 2t)."""
    b = ndimage.gaussian_filter(a, smooth, mode="reflect") if smooth > 0 else a
    gy = ndimage.sobel(b, axis=0, mode="reflect")
    gx = ndimage.sobel(b, axis=1, mode="reflect")
    m = np.hypot(gx, gy)
    # m*cos(2t) = (gx^2-gy^2)/m and m*sin(2t) = 2*gx*gy/m, computed stably.
    denom = np.maximum(m, 1e-6)
    dx = (gx * gx - gy * gy) / denom
    dy = (2.0 * gx * gy) / denom
    return dx.astype(np.float32), dy.astype(np.float32)


@dataclass
class ChannelMaps:
    maps: dict[str, np.ndarray]     # channel -> response map (valid coords)
    half_w: float
    half_h: float
    scale: float
    rotation: float

    def score_at(self, ch: str, x: float, y: float) -> float:
        r = self.maps[ch]
        c, rr = int(round(x - self.half_w)), int(round(y - self.half_h))
        if 0 <= rr < r.shape[0] and 0 <= c < r.shape[1]:
            return float(r[rr, c])
        return float("nan")


def response_maps(reference: np.ndarray, search: np.ndarray,
                  scale: float = 1.0, rotation: float = 0.0) -> ChannelMaps:
    """All three channels for one transform hypothesis. 6 NCC calls."""
    sf = _robust_contrast(search)
    sm = midband(sf)
    sdx, sdy = directionality(sf)

    t = _template_from_reference(reference, scale, rotation)
    tm = midband(t)
    tdx, tdy = directionality(t)

    raw = _ncc_valid(sf, t)
    mid = _ncc_valid(sm, tm)
    # Two orthogonal orientation components are averaged into one channel;
    # they are the real and imaginary parts of the same complex field.
    dirn = (0.5 * (_ncc_valid(sdx, tdx) + _ncc_valid(sdy, tdy))).astype(np.float32)

    return ChannelMaps({"raw": raw, "midband": mid, "directionality": dirn},
                       (t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0, scale, rotation)


def local_maxima(resp: np.ndarray, footprint: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loc = resp == ndimage.maximum_filter(resp, size=footprint, mode="nearest")
    ys, xs = np.nonzero(loc)
    return xs, ys, resp[ys, xs]


def harvest(cm: ChannelMaps, delta: float = 0.10, footprint: int = 5,
            cap_per_channel: int = 4000) -> list[dict]:
    """Union of per-channel local maxima passing S_i >= S_max_channel - delta.

    The threshold is per channel, because the three maps have different score
    scales and a shared absolute cut would silently mute the weakest one.
    """
    pool: dict[tuple[int, int], dict] = {}
    for ch, resp in cm.maps.items():
        xs, ys, vals = local_maxima(resp, footprint)
        if vals.size == 0:
            continue
        keep = vals >= (float(vals.max()) - delta)
        if keep.sum() > cap_per_channel:                 # keep the strongest
            idx = np.argsort(vals)[::-1][:cap_per_channel]
            keep = np.zeros_like(keep); keep[idx] = True
        for x, y in zip(xs[keep] + cm.half_w, ys[keep] + cm.half_h):
            key = (int(round(x)), int(round(y)))
            rec = pool.get(key)
            if rec is None:
                rec = {"x": float(x), "y": float(y), "votes": 0,
                       "scale": cm.scale, "rotation": cm.rotation}
                for c in CHANNELS:
                    rec[c] = float("nan")
                pool[key] = rec
            rec["votes"] += 1
    # Fill every channel's score at every harvested location, so the feature
    # table is dense and a candidate found by one channel carries the other
    # two channels' opinion of it - which is the whole point of the ensemble.
    for rec in pool.values():
        for c in CHANNELS:
            rec[c] = cm.score_at(c, rec["x"], rec["y"])
    return list(pool.values())
