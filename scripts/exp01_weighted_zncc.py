#!/usr/bin/env python3
"""Wave A: uniqueness-weighted proposal correlation (experiments 1-5).

Experiment 1 killed the idea of subtracting periodicity from the *response*
surface: a correlation value is high at the true site because the lattice
matches, so its lattice translates are high for the same reason and the
subtraction removes the signal along with the nuisance.

This attacks the same flatness from the other side, and survives that argument
because it never touches the response. It changes **which reference pixels are
allowed to vote**. The lattice carrier stays intact -- the correlation still
knows it is looking at DRAM -- but pixels that repeat under lattice translation
contribute less than pixels that do not.

Experiments 1 through 5 are one pipeline with five families of weight, so they
share every expensive intermediate (pose-correct template, prepared search
channels, lattice basis, ``t_unique``) and are evaluated in one pass:

    W*   uniqueness alone, at several exponents            (exp 1)
    G*   uniqueness x gradient reliability                 (exp 2)
    S*   spatially balanced uniqueness, 4x4 reference grid (exp 3)
    E*   uniqueness x local histogram entropy              (exp 4)
    V*   uniqueness x local standard deviation             (exp 5)

Scoring uses ``residual.weighted_zncc_valid``, which is a *proper* weighted
ZNCC: the template mean and energy are taken under the same weights as the
search patch, via three FFT correlations. Masking the template and running an
ordinary NCC would leave the statistics unweighted and quietly measure
something else.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import _template_from_reference
from driftforge.channels import (CHANNELS, ChannelMaps, directionality, harvest,
                                 midband, prepare_search, response_maps)
from driftforge.lattice import estimate_lattice
from driftforge.residual import periodic_residual, weighted_zncc_valid

HIT_PX = 1.5
DELTAS = (0.10, 0.15)
K_GRID = (64, 128, 256, 512)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def stratified(records: list[dict], per_cell: int | None) -> list[dict]:
    if per_cell is None:
        return records
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        cells[(rec["architecture"], rec["severity"])].append(rec)
    out: list[dict] = []
    for key in sorted(cells):
        out.extend(cells[key][:per_cell])
    return out


def _norm01(a: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 90.0) -> np.ndarray:
    """Percentile-normalise to [0, 1]; robust to the odd saturated pixel."""
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-6:
        return np.ones_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _local_std(t: np.ndarray, win: int) -> np.ndarray:
    m = ndimage.uniform_filter(t.astype(np.float32), win, mode="reflect")
    m2 = ndimage.uniform_filter(t.astype(np.float32) ** 2, win, mode="reflect")
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


def _local_entropy(t: np.ndarray, win: int = 9, bins: int = 16) -> np.ndarray:
    """Local histogram entropy, computed as a sum over quantised bin planes.

    One uniform filter per bin is far cheaper than a sliding histogram at this
    template size, and the template is only ~100 px on a side.
    """
    q = np.clip(((t - t.min()) / max(float(t.max() - t.min()), 1e-6) * bins).astype(int),
                0, bins - 1)
    H = np.zeros(t.shape, dtype=np.float32)
    for b in range(bins):
        p = ndimage.uniform_filter((q == b).astype(np.float32), win, mode="reflect")
        H -= np.where(p > 1e-6, p * np.log(np.maximum(p, 1e-6)), 0.0)
    return H.astype(np.float32)


def _balanced(u: np.ndarray, grid: int, keep_frac: float) -> np.ndarray:
    """Top-fraction uniqueness *within each cell* of a grid over the template.

    A single defect can dominate a global uniqueness threshold, concentrating
    every voting pixel in one place -- which is exactly the evidence Set B
    degradation is most likely to destroy. Per-cell quotas force the vote to be
    spread over the reference.
    """
    h, w = u.shape
    out = np.zeros_like(u, dtype=np.float32)
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            cell = u[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if cell.size == 0:
                continue
            thr = np.percentile(cell, 100.0 * (1.0 - keep_frac))
            out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = (cell >= thr).astype(np.float32)
    return out


def build_weights(t: np.ndarray, t_unique: np.ndarray) -> dict[str, np.ndarray]:
    """Every weight family, on template coordinates, in one place."""
    u = _norm01(t_unique)
    gy = ndimage.sobel(t, axis=0, mode="reflect")
    gx = ndimage.sobel(t, axis=1, mode="reflect")
    g = _norm01(np.hypot(gx, gy), 5.0, 95.0)
    W: dict[str, np.ndarray] = {}
    # exp 1 -- uniqueness alone
    W["W0_m30_binary"] = (t_unique > np.percentile(t_unique, 70)).astype(np.float32)
    W["W1_u_linear"] = u
    W["W2_u_p05"] = u ** 0.5
    W["W3_u_p15"] = u ** 1.5
    W["W4_u_p20"] = u ** 2.0
    # exp 2 -- uniqueness x gradient reliability
    W["G1_u_g"] = u * g
    W["G2_u05_g"] = (u ** 0.5) * g
    W["G3_u_g05"] = u * (g ** 0.5)
    W["G4_u15_g"] = (u ** 1.5) * g
    # exp 3 -- spatially balanced
    W["S1_bal4_20"] = _balanced(t_unique, 4, 0.20)
    W["S2_bal4_30"] = _balanced(t_unique, 4, 0.30)
    # exp 5 -- local variance (cheap surrogate for entropy)
    W["V1_u_std9"] = u * _norm01(_local_std(t, 9), 5.0, 95.0)
    # exp 4 -- local entropy
    W["E1_u_ent16"] = u * _norm01(_local_entropy(t, 9, 16), 5.0, 95.0)
    return W


def weighted_maps(prepared, t: np.ndarray, w: np.ndarray,
                  scale: float, rotation: float) -> ChannelMaps:
    """The three declared channels, each scored under the same weight map."""
    tm = midband(t)
    tdx, tdy = directionality(t)
    raw = weighted_zncc_valid(prepared.raw, t, w)
    mid = weighted_zncc_valid(prepared.midband, tm, w)
    dirn = (0.5 * (weighted_zncc_valid(prepared.dx, tdx, w)
                   + weighted_zncc_valid(prepared.dy, tdy, w))).astype(np.float32)
    return ChannelMaps({"raw": raw, "midband": mid, "directionality": dirn},
                       (t.shape[1] - 1) / 2.0, (t.shape[0] - 1) / 2.0, scale, rotation)


def pool_stats(cands: list[dict], gx: float, gy: float) -> dict:
    if not cands:
        return {"n": 0, "d_min": float("inf"), "rank": -1}
    d = np.array([math.hypot(c["x"] - gx, c["y"] - gy) for c in cands])
    best = np.array([max((c[ch] for ch in CHANNELS
                          if isinstance(c.get(ch), float) and math.isfinite(c[ch])),
                         default=-np.inf) for c in cands])
    order = np.argsort(-best, kind="mergesort")
    hit = np.nonzero(d[order] <= HIT_PX)[0]
    return {"n": len(cands), "d_min": float(d.min()),
            "rank": int(hit[0]) if hit.size else -1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="p2_val")
    ap.add_argument("--root", default=str(PROJECT / "data" / "phase2"))
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--only", default=None,
                    help="comma-separated weight keys, for the confirmation run")
    ap.add_argument("--out", default=str(PROJECT / "results" / "exp01_weighted_zncc.json"))
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    records = [json.loads(line) for line in open(split_dir / "manifest.jsonl")]
    subset = stratified([r for r in records if r["present"]], args.per_cell)
    keep = set(args.only.split(",")) if args.only else None
    print(f"{args.split}: {len(subset)} present pairs", flush=True)

    rows: list[dict] = []
    for i, rec in enumerate(subset, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
        tscale = 10.0 / float(rec["gt_scale"])
        trot = float(rec["gt_theta"])

        prepared = prepare_search(search)
        t = _template_from_reference(ref, tscale, trot)
        cm = response_maps(ref, search, tscale, trot, prepared=prepared, template=t)

        row = {"id": rec["id"], "architecture": rec["architecture"],
               "severity": rec["severity"], "v": {}}
        for d in DELTAS:
            row["v"][f"baseline_d{d:g}"] = pool_stats(harvest(cm, delta=d), gx, gy)

        lat = estimate_lattice(search)
        B = lat.basis
        v1, v2 = B[:, 0].copy(), B[:, 1].copy()
        if (abs(float(np.linalg.det(B))) > 1.0
                and np.linalg.norm(v1) >= 2.0 and np.linalg.norm(v2) >= 2.0):
            _, t_unique = periodic_residual(t, v1, v2)
            weights = build_weights(t, t_unique)
            for name, w in weights.items():
                if keep is not None and name not in keep:
                    continue
                t0 = time.perf_counter()
                cmw = weighted_maps(prepared, t, w, tscale, trot)
                dt = time.perf_counter() - t0
                for d in DELTAS:
                    st = pool_stats(harvest(cmw, delta=d), gx, gy)
                    st["t_maps"] = dt
                    row["v"][f"{name}_d{d:g}"] = st
        rows.append(row)
        b = row["v"]["baseline_d0.1"]
        w1 = row["v"].get("W1_u_linear_d0.1", {})
        print(f"[{i}/{len(subset)}] {rec['id']} {rec['architecture']} sev{rec['severity']} "
              f"base n={b['n']} rank={b['rank']} | W1 n={w1.get('n')} rank={w1.get('rank')}",
              flush=True)

    names = sorted({k for r in rows for k in r["v"]})

    def agg(name: str, sel=None) -> dict:
        rs = [r for r in rows if name in r["v"] and (sel is None or sel(r))]
        if not rs:
            return {}
        n = np.array([r["v"][name]["n"] for r in rs], dtype=float)
        hit = np.array([r["v"][name]["d_min"] <= HIT_PX for r in rs])
        rank = np.array([r["v"][name]["rank"] for r in rs], dtype=float)
        good = rank >= 0
        out = {"pairs": len(rs), "harvest_recall": round(float(hit.mean()), 4),
               "n_median": float(np.median(n)), "n_p95": float(np.percentile(n, 95)),
               "true_rank_median": float(np.median(rank[good])) if good.any() else None,
               "true_rank_p90": float(np.percentile(rank[good], 90)) if good.any() else None}
        for k in K_GRID:
            out[f"recall_at_{k}"] = round(float(np.mean(good & (rank < k))), 4)
        ts = [r["v"][name].get("t_maps") for r in rs if r["v"][name].get("t_maps")]
        out["t_maps_median"] = round(float(np.median(ts)), 3) if ts else None
        return out

    hard = {r["id"] for r in rows if r["v"]["baseline_d0.1"]["n"] > 2000}
    summary = {
        "split": args.split, "pairs": len(rows), "hit_px": HIT_PX,
        "overall": {n: agg(n) for n in names},
        "hard_regime": {"pairs": len(hard),
                        **{n: agg(n, lambda r: r["id"] in hard) for n in names}},
        "by_architecture": {a: {n: agg(n, lambda r, a=a: r["architecture"] == a)
                                for n in names} for a in ("dram", "finfet")},
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    for n in names:
        print(f"{n:24s}", json.dumps(summary['overall'][n]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
