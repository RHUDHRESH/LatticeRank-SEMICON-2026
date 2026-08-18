"""Local structural identity descriptor for candidate discrimination.

Periodic copies can have nearly identical global correlation. The descriptor
therefore compares where microstructure appears inside each candidate patch,
using blockwise NCC grids, row/column profiles, local spectra, and phase
coherence after a tiny (+-3 px) local alignment.

The descriptor never relocates a candidate beyond the tiny alignment radius
and never sees distance to the Search centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .baseline import _robust_contrast, _template_from_reference
from .channels import directionality, midband

SHIFT_RADIUS = 3  # tiny local alignment only; relocation is not the job here


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-shape patches."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a * b).sum() / (na * nb))


def _zn(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def _block_nccs(t: np.ndarray, p: np.ndarray, grid: int) -> np.ndarray:
    """Local NCC on a grid x grid partition - the spatial-correspondence core."""
    h, w = t.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = np.empty(grid * grid)
    k = 0
    for i in range(grid):
        for j in range(grid):
            out[k] = _ncc(t[ys[i]:ys[i + 1], xs[j]:xs[j + 1]],
                          p[ys[i]:ys[i + 1], xs[j]:xs[j + 1]])
            k += 1
    return out


def _block_stats(vals: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_std": float(vals.std()),
        f"{prefix}_min": float(vals.min()),
        f"{prefix}_p10": float(np.percentile(vals, 10)),
        f"{prefix}_p25": float(np.percentile(vals, 25)),
        f"{prefix}_med": float(np.median(vals)),
        f"{prefix}_p75": float(np.percentile(vals, 75)),
        f"{prefix}_max": float(vals.max()),
        f"{prefix}_frac05": float(np.mean(vals > 0.5)),
        f"{prefix}_frac07": float(np.mean(vals > 0.7)),
    }


def _spec_profiles(patch: np.ndarray, win: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windowed FFT magnitude + axis profiles (local frequency signature)."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(_zn(patch) * win)))
    return f, f.mean(axis=0), f.mean(axis=1)


def _dominant(profile: np.ndarray) -> int:
    """Dominant off-DC frequency of a centred spectral profile."""
    n = profile.size
    c = n // 2
    half = profile[c + 2:].copy()  # skip DC +-1
    return int(np.argmax(half)) + 2 if half.size else 0


@dataclass
class SceneStructContext:
    """Per-scene precomputation: template representations + padded search maps."""

    th: int
    tw: int
    half_w: float
    half_h: float
    pad: int
    tpl: dict[str, np.ndarray]
    srch: dict[str, np.ndarray]     # padded by `pad` (reflect)
    tpl_spec: dict[str, np.ndarray]
    win: np.ndarray
    win50: np.ndarray


def build_context(reference: np.ndarray, search: np.ndarray,
                  scale: float = 1.0, rotation: float = 0.0,
                  shift_radius: int = SHIFT_RADIUS) -> SceneStructContext:
    t = _template_from_reference(reference, scale, rotation)
    sf = _robust_contrast(search)

    tm = midband(t)
    tdx, tdy = directionality(t)
    tgy = ndimage.sobel(t, axis=0, mode="reflect")
    tgx = ndimage.sobel(t, axis=1, mode="reflect")

    sm = midband(sf)
    sdx, sdy = directionality(sf)
    sgy = ndimage.sobel(sf, axis=0, mode="reflect")
    sgx = ndimage.sobel(sf, axis=1, mode="reflect")

    pad = shift_radius
    srch = {k: np.pad(v, pad, mode="reflect") for k, v in
            {"raw": sf, "mid": sm, "dx": sdx, "dy": sdy,
             "gx": sgx, "gy": sgy}.items()}

    th, tw = t.shape
    win = np.outer(np.hanning(th), np.hanning(tw))
    c50 = min(50, th, tw)
    win50 = np.outer(np.hanning(c50), np.hanning(c50))
    spec, hprof, vprof = _spec_profiles(t, win)
    tpl_spec = {"mag": spec, "h": hprof, "v": vprof}

    return SceneStructContext(
        th=th, tw=tw, half_w=(tw - 1) / 2.0, half_h=(th - 1) / 2.0, pad=pad,
        tpl={"raw": t, "mid": tm, "dx": tdx, "dy": tdy, "gx": tgx, "gy": tgy,
             "gmag": np.hypot(tgx, tgy)},
        srch=srch, tpl_spec=tpl_spec, win=win, win50=win50)


def describe(ctx: SceneStructContext, x: float, y: float) -> dict[str, float]:
    """Structural identity features for the candidate centred at (x, y)."""
    R = ctx.pad
    th, tw = ctx.th, ctx.tw
    H, W = ctx.srch["raw"].shape
    # template top-left in unpadded coords; +R converts to padded coords, and
    # the clip keeps the extended (th+2R, tw+2R) window inside the padded map
    tly = int(np.clip(round(y - ctx.half_h), -R, H - th - R)) + R
    tlx = int(np.clip(round(x - ctx.half_w), -R, W - tw - R)) + R

    ext = {k: v[tly - R:tly + th + R, tlx - R:tlx + tw + R] for k, v in ctx.srch.items()}
    t_raw, t_mid = ctx.tpl["raw"], ctx.tpl["mid"]

    # ---- tiny-shift alignment on raw+midband (never a relocation mechanism)
    tn_raw, tn_mid = _zn(t_raw).ravel(), _zn(t_mid).ravel()
    n = tn_raw.size
    best, best_sc, raw0, mid0, raw_best, mid_best = (0, 0), -9.0, 0.0, 0.0, -9.0, -9.0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            pr = ext["raw"][R + dy:R + dy + th, R + dx:R + dx + tw]
            pm = ext["mid"][R + dy:R + dy + th, R + dx:R + dx + tw]
            zr = float(tn_raw @ _zn(pr).ravel()) / n
            zm = float(tn_mid @ _zn(pm).ravel()) / n
            sc = 0.5 * (zr + zm)
            if dx == 0 and dy == 0:
                raw0, mid0 = zr, zm
            if sc > best_sc:
                best_sc, best, raw_best, mid_best = sc, (dx, dy), zr, zm
    dx0, dy0 = best
    P = {k: v[R + dy0:R + dy0 + th, R + dx0:R + dx0 + tw] for k, v in ext.items()}

    f: dict[str, float] = {
        "s_raw_zncc0": raw0, "s_raw_zncc": raw_best,
        "s_mid_zncc0": mid0, "s_mid_zncc": mid_best,
        "s_tiny_dx": float(dx0), "s_tiny_dy": float(dy0),
        "s_tiny_mag": float(math.hypot(dx0, dy0)),
    }

    # ---- residuals on normalized intensity
    resid = _zn(t_raw) - _zn(P["raw"])
    f["s_raw_resid_rms"] = float(np.sqrt(np.mean(resid ** 2)))
    f["s_raw_resid_mae"] = float(np.mean(np.abs(resid)))
    rm = _zn(t_mid) - _zn(P["mid"])
    f["s_mid_resid_rms"] = float(np.sqrt(np.mean(rm ** 2)))

    # ---- gradient similarity
    pmag = np.hypot(P["gx"], P["gy"])
    f["s_gx_ncc"] = _ncc(ctx.tpl["gx"], P["gx"])
    f["s_gy_ncc"] = _ncc(ctx.tpl["gy"], P["gy"])
    f["s_gmag_ncc"] = _ncc(ctx.tpl["gmag"], pmag)
    f["s_grad_resid_rms"] = float(np.sqrt(np.mean((_zn(ctx.tpl["gmag"]) - _zn(pmag)) ** 2)))

    # ---- polarity-insensitive directionality
    f["s_dirx_ncc"] = _ncc(ctx.tpl["dx"], P["dx"])
    f["s_diry_ncc"] = _ncc(ctx.tpl["dy"], P["dy"])
    f["s_dir_ncc"] = 0.5 * (f["s_dirx_ncc"] + f["s_diry_ncc"])

    # ---- spatial block correspondence (the core discriminator)
    b4 = _block_nccs(t_raw, P["raw"], 4)
    f.update(_block_stats(b4, "s_b4"))
    f["s_b4_worst_loc"] = float(int(np.argmin(b4)))
    b5 = _block_nccs(t_raw, P["raw"], 5)
    f["s_b5_mean"] = float(b5.mean())
    f["s_b5_min"] = float(b5.min())
    f["s_b5_p25"] = float(np.percentile(b5, 25))
    # blockwise directionality agreement (FinFET fins/gates live here)
    bd = 0.5 * (_block_nccs(ctx.tpl["dx"], P["dx"], 4) + _block_nccs(ctx.tpl["dy"], P["dy"], 4))
    f["s_bdir4_mean"] = float(bd.mean())
    f["s_bdir4_min"] = float(bd.min())
    f["s_bdir4_p25"] = float(np.percentile(bd, 25))

    # ---- row / column profile similarity
    f["s_row_corr"] = _ncc(t_raw.mean(axis=1), P["raw"].mean(axis=1))
    f["s_col_corr"] = _ncc(t_raw.mean(axis=0), P["raw"].mean(axis=0))
    te = ctx.tpl["gmag"] ** 2
    pe = pmag ** 2
    f["s_rowE_corr"] = _ncc(te.mean(axis=1), pe.mean(axis=1))
    f["s_colE_corr"] = _ncc(te.mean(axis=0), pe.mean(axis=0))

    # ---- local frequency signature
    pf, ph, pv = _spec_profiles(P["raw"], ctx.win)
    f["s_spec_h_corr"] = _ncc(ctx.tpl_spec["h"], ph)
    f["s_spec_v_corr"] = _ncc(ctx.tpl_spec["v"], pv)
    f["s_spec_resid"] = float(np.sqrt(np.mean((_zn(ctx.tpl_spec["mag"]) - _zn(pf)) ** 2)))
    f["s_domfreq_h"] = float(math.exp(-abs(_dominant(ctx.tpl_spec["h"]) - _dominant(ph))))
    f["s_domfreq_v"] = float(math.exp(-abs(_dominant(ctx.tpl_spec["v"]) - _dominant(pv))))

    # ---- phase / translation consistency (descriptor only, not a mover)
    Ft = np.fft.fft2(_zn(t_raw) * ctx.win)
    Fp = np.fft.fft2(_zn(P["raw"]) * ctx.win)
    cross = Ft * np.conj(Fp)
    cross /= np.maximum(np.abs(cross), 1e-9)
    surf = np.abs(np.fft.ifft2(cross))
    pk = int(np.argmax(surf))
    py_, px_ = divmod(pk, tw)
    peak = float(surf.flat[pk])
    mask = surf.copy()
    mask[max(0, py_ - 2):py_ + 3, max(0, px_ - 2):px_ + 3] = 0.0
    second = float(mask.max())
    sy = py_ - th if py_ > th // 2 else py_
    sx = px_ - tw if px_ > tw // 2 else px_
    f["s_phase_peak"] = peak
    f["s_phase_ratio"] = peak / max(second, 1e-9)
    f["s_phase_dx"] = float(sx)
    f["s_phase_dy"] = float(sy)
    f["s_phase_mag"] = float(math.hypot(sx, sy))

    # ---- multi-scale: 50x50 centre crop separates microtexture from layout
    c = min(50, th, tw)
    oy, ox = (th - c) // 2, (tw - c) // 2
    tc = t_raw[oy:oy + c, ox:ox + c]
    pc = P["raw"][oy:oy + c, ox:ox + c]
    f["s_ms50_zncc"] = _ncc(tc, pc)
    f["s_ms50_dir"] = 0.5 * (_ncc(ctx.tpl["dx"][oy:oy + c, ox:ox + c], P["dx"][oy:oy + c, ox:ox + c])
                             + _ncc(ctx.tpl["dy"][oy:oy + c, ox:ox + c], P["dy"][oy:oy + c, ox:ox + c]))
    f["s_ms50_b4_mean"] = float(_block_nccs(tc, pc, 4).mean())
    return f


STRUCT_FEATURES = [
    "s_raw_zncc0", "s_raw_zncc", "s_mid_zncc0", "s_mid_zncc",
    "s_tiny_dx", "s_tiny_dy", "s_tiny_mag",
    "s_raw_resid_rms", "s_raw_resid_mae", "s_mid_resid_rms",
    "s_gx_ncc", "s_gy_ncc", "s_gmag_ncc", "s_grad_resid_rms",
    "s_dirx_ncc", "s_diry_ncc", "s_dir_ncc",
    "s_b4_mean", "s_b4_std", "s_b4_min", "s_b4_p10", "s_b4_p25", "s_b4_med",
    "s_b4_p75", "s_b4_max", "s_b4_frac05", "s_b4_frac07", "s_b4_worst_loc",
    "s_b5_mean", "s_b5_min", "s_b5_p25",
    "s_bdir4_mean", "s_bdir4_min", "s_bdir4_p25",
    "s_row_corr", "s_col_corr", "s_rowE_corr", "s_colE_corr",
    "s_spec_h_corr", "s_spec_v_corr", "s_spec_resid", "s_domfreq_h", "s_domfreq_v",
    "s_phase_peak", "s_phase_ratio", "s_phase_dx", "s_phase_dy", "s_phase_mag",
    "s_ms50_zncc", "s_ms50_dir", "s_ms50_b4_mean",
]

#: Transparent fixed-weight combination for descriptor-only ranking.
#: Features are z-scored per scene first.
STRUCT_SCORE_WEIGHTS = {
    "s_b4_mean": 0.25, "s_b4_p25": 0.20, "s_dir_ncc": 0.15, "s_mid_zncc": 0.15,
    "s_rowcol": 0.10, "s_phase_peak": 0.10, "s_raw_zncc": 0.05,
}


def struct_score(rows: list[dict[str, float]]) -> np.ndarray:
    """Fixed-weight structural score over one scene's candidates (z-scored)."""
    if not rows:
        return np.zeros(0)
    cols = {}
    for k in ("s_b4_mean", "s_b4_p25", "s_dir_ncc", "s_mid_zncc",
              "s_phase_peak", "s_raw_zncc", "s_row_corr", "s_col_corr"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        sd = v.std()
        cols[k] = (v - v.mean()) / (sd if sd > 1e-9 else 1.0)
    rowcol = 0.5 * (cols["s_row_corr"] + cols["s_col_corr"])
    w = STRUCT_SCORE_WEIGHTS
    return (w["s_b4_mean"] * cols["s_b4_mean"] + w["s_b4_p25"] * cols["s_b4_p25"]
            + w["s_dir_ncc"] * cols["s_dir_ncc"] + w["s_mid_zncc"] * cols["s_mid_zncc"]
            + w["s_rowcol"] * rowcol + w["s_phase_peak"] * cols["s_phase_peak"]
            + w["s_raw_zncc"] * cols["s_raw_zncc"])
