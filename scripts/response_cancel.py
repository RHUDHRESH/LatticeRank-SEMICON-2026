#!/usr/bin/env python3
"""Experiment 1: cancel the lattice-periodic component of the RESPONSE map.

Phase 1 already cancels the periodic background -- but it does so in image
space, per candidate, after thousands of aliases have been harvested. The
ambiguity is created first and dismantled afterwards, one expensive candidate
at a time. Measured on p2_val: 23% of scenes emit more than 2,000 candidates,
and in that regime runtime is 31.5 s and selection is 9.6%.

The correlation surface is itself periodic. If the reference matches one
lattice cell, it matches every lattice cell, so ``C(x)`` repeats at
``x + m*v1 + n*v2``. That repetition is nuisance for site identity by exactly
the argument ``residual.py`` already makes about pixels::

    P_C(x) = median over the ring of C(x + k)      k in SHIFT_SET
    C_u(x) = C(x) - P_C(x)

A response peak that is only there because the whole field repeats cancels; a
peak that is higher than its own lattice translates survives. Same operator,
same lattice basis, same shift set as the declared method -- moved upstream of
harvesting instead of downstream of it.

Ring 2 is offered because some difference from the immediate neighbours is
itself periodic: FinFET gate CD can vary with fin pitch, so a ring-1 median can
mistake coherent process modulation for site-specific evidence.

Everything runs at ground-truth pose, and the baseline pool is harvested from
the identical maps in the same call, so the comparison is paired per scene.
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

from driftforge.channels import CHANNELS, ChannelMaps, harvest, response_maps
from driftforge.lattice import estimate_lattice
from driftforge.residual import SHIFT_SET

HIT_PX = 1.5

#: Ring-2 lattice translations, added to SHIFT_SET for the "ring2" variant.
SHIFT_SET_2 = ((2, 0), (-2, 0), (0, 2), (0, -2),
               (2, 1), (-2, -1), (1, 2), (-1, -2),
               (2, -1), (-2, 1), (1, -2), (-1, 2))

#: Cancellation collapses the pool, which buys back the budget to harvest at a
#: looser threshold. The wide end of this grid is only meaningful on the
#: scale-matched maps; on the raw cancelled maps it just floods.
DELTAS = (0.10, 0.15, 0.20, 0.30)


def rescale_like(C_u: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Put a cancelled map back on the original map's dynamic range.

    ``harvest`` cuts at ``S >= S_max - delta``, an ABSOLUTE margin. Cancellation
    shrinks the response range -- the periodic peaks that set the old maximum
    are exactly what was subtracted -- so the same 0.10 becomes a far looser cut
    and the pool can grow instead of shrink. Measured on the smoke set: ring-1
    cancellation moved the median pool from 678 to 289 but the p95 from 3,624 to
    6,132, which is a threshold artefact, not a property of the operator.

    Matching ``max - median`` makes delta mean the same fraction of dynamic
    range in both maps, so the comparison comes out paired.
    """
    src = float(np.nanmax(C) - np.nanmedian(C))
    dst = float(np.nanmax(C_u) - np.nanmedian(C_u))
    if not np.isfinite(src) or not np.isfinite(dst) or dst < 1e-9:
        return C_u
    return ((C_u - np.nanmedian(C_u)) * (src / dst)).astype(np.float32)


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


def cancel_map(C: np.ndarray, v1: np.ndarray, v2: np.ndarray,
               shifts) -> np.ndarray:
    """Subtract the median of the map's own lattice translates.

    ``mode="nearest"`` rather than reflection: a reflected edge would invent a
    response that the correlation never produced, and edge candidates are
    exactly where a spurious survivor is most expensive.
    """
    stack = np.empty((len(shifts),) + C.shape, dtype=np.float32)
    for i, (m, n) in enumerate(shifts):
        dy = float(m * v1[1] + n * v2[1])
        dx = float(m * v1[0] + n * v2[0])
        ndimage.shift(C, (dy, dx), output=stack[i], order=1,
                      mode="nearest", prefilter=False)
    return (C - np.median(stack, axis=0)).astype(np.float32)


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
    ap.add_argument("--per-cell", type=int, default=None)
    ap.add_argument("--out", default=str(PROJECT / "results" / "response_cancel.json"))
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    records = [json.loads(line) for line in open(split_dir / "manifest.jsonl")]
    subset = stratified([r for r in records if r["present"]], args.per_cell)
    print(f"{args.split}: {len(subset)} present pairs", flush=True)

    variants = ["baseline"] + [f"{r}{t}_d{d:g}"
                               for r in ("ring1", "ring2")
                               for t in ("", "n") for d in DELTAS]
    rows: list[dict] = []

    for i, rec in enumerate(subset, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
        tscale = 10.0 / float(rec["gt_scale"])
        trot = float(rec["gt_theta"])

        cm = response_maps(ref, search, tscale, trot)
        lat = estimate_lattice(search)
        B = lat.basis
        v1, v2 = B[:, 0].copy(), B[:, 1].copy()
        ok = (abs(float(np.linalg.det(B))) > 1.0
              and np.linalg.norm(v1) >= 2.0 and np.linalg.norm(v2) >= 2.0)

        row = {"id": rec["id"], "architecture": rec["architecture"],
               "severity": rec["severity"], "lattice_ok": bool(ok), "v": {}}
        row["v"]["baseline"] = pool_stats(harvest(cm, delta=0.10), gx, gy)

        if ok:
            for ring, shifts in (("ring1", SHIFT_SET),
                                 ("ring2", tuple(SHIFT_SET) + SHIFT_SET_2)):
                t0 = time.perf_counter()
                cancelled = {ch: cancel_map(cm.maps[ch], v1, v2, shifts)
                             for ch in cm.maps}
                dt = time.perf_counter() - t0
                scaled = {ch: rescale_like(cancelled[ch], cm.maps[ch])
                          for ch in cancelled}
                for tag, maps in (("", cancelled), ("n", scaled)):
                    cm_u = ChannelMaps(maps=maps, half_w=cm.half_w,
                                       half_h=cm.half_h, scale=cm.scale,
                                       rotation=cm.rotation)
                    for d in DELTAS:
                        st = pool_stats(harvest(cm_u, delta=d), gx, gy)
                        st["t_cancel"] = dt
                        row["v"][f"{ring}{tag}_d{d:g}"] = st
        else:
            for name in variants[1:]:
                row["v"][name] = {"n": -1, "d_min": float("inf"), "rank": -1}
        rows.append(row)
        b = row["v"]["baseline"]
        r1 = row["v"].get("ring1_d0.1", {})
        print(f"[{i}/{len(subset)}] {rec['id']} {rec['architecture']} sev{rec['severity']} "
              f"base n={b['n']} rank={b['rank']} | ring1 n={r1.get('n')} "
              f"rank={r1.get('rank')}", flush=True)

    def agg(name: str, sel=None) -> dict:
        rs = [r for r in rows if (sel is None or sel(r))]
        rs = [r for r in rs if r["v"][name]["n"] >= 0]
        if not rs:
            return {}
        n = np.array([r["v"][name]["n"] for r in rs], dtype=float)
        hit = np.array([r["v"][name]["d_min"] <= HIT_PX for r in rs])
        rank = np.array([r["v"][name]["rank"] for r in rs], dtype=float)
        good = rank >= 0
        return {
            "pairs": len(rs),
            "harvest_recall": round(float(hit.mean()), 4),
            "n_median": float(np.median(n)),
            "n_p95": float(np.percentile(n, 95)),
            "n_max": int(n.max()),
            "true_rank_median": float(np.median(rank[good])) if good.any() else None,
            "true_rank_p90": float(np.percentile(rank[good], 90)) if good.any() else None,
            "recall_at_256": round(float(np.mean(good & (rank < 256))), 4),
        }

    hard = set(r["id"] for r in rows if r["v"]["baseline"]["n"] > 2000)
    summary = {
        "split": args.split, "pairs": len(rows), "hit_px": HIT_PX,
        "lattice_ok": int(sum(r["lattice_ok"] for r in rows)),
        "overall": {v: agg(v) for v in variants},
        "hard_regime_baseline_n_gt_2000": {
            "pairs": len(hard),
            **{v: agg(v, lambda r: r["id"] in hard) for v in variants},
        },
        "by_architecture": {
            a: {v: agg(v, lambda r, a=a: r["architecture"] == a) for v in variants}
            for a in ("dram", "finfet")
        },
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print("\nHARD REGIME (baseline n>2000):")
    print(json.dumps(summary["hard_regime_baseline_n_gt_2000"], indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
