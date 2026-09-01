#!/usr/bin/env python3
"""Cache the cheap per-candidate feature table used by every pre-ranker test.

The question all of these experiments circle is one thing: can cheap evidence
compress a 2,000-6,000 candidate pool to a few hundred while keeping the true
site? Answering it needs a per-candidate table over frozen pools, and building
that table is the entire cost -- once it exists, RCC-only ranking, hybrid
ranking and adaptive-K are minutes of analysis on the same file rather than
three separate hour-long runs.

**Nothing expensive is computed here.** ``compute_candidate_rows`` costs about
9,800 us per candidate because it runs the structural descriptor; this uses
``response_maps`` + ``harvest`` directly, which yields the three channel scores
per candidate for the price of the maps, and adds RCC at 550 us. That is the
whole point: a pre-ranker may only use evidence cheap enough to afford on every
candidate in the pool.

Ground-truth pose is used throughout. The pre-ranker is being asked whether it
can preserve the true site under perfect pose; if it cannot do that, no pose
front end will save it, and mixing in pose error would confound the measurement.

Splits are kept as the generator made them -- ``p2_train`` for fitting,
``p2_val`` for reporting -- because they are scene-disjoint by construction.
Re-slicing one split would leak scene geometry across the boundary.
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, response_maps
from driftforge.lattice import estimate_lattice
from driftforge.pipeline import DELTA
from driftforge.rcc import FEATURES as RCC_FEATURES
from driftforge.rcc import build_scorer

HIT_PX = 1.5

#: Cheap per-candidate columns derived from the three channel response maps.
#: Ranks and deltas are included because absolute channel scores are not
#: comparable across scenes -- a 0.72 means something different in a clean DRAM
#: field than under severity 3 -- while a within-scene rank is.
CHEAP_FEATURES = (
    [f"ch_{c}" for c in CHANNELS]
    + [f"rank_{c}" for c in CHANNELS]
    + [f"delta_{c}" for c in CHANNELS]
    + ["ch_best", "ch_best_rank", "ch_best_delta", "ch_spread", "votes",
       "lat_dx_periods", "lat_dy_periods", "lat_phase_res",
       "lat_parity_x", "lat_parity_y", "dist_to_argmax", "dist_to_centre"]
)

#: Scene-level columns, repeated on every candidate of that scene. A ranker may
#: legitimately learn "trust RCC when the pool is huge", which needs the pool
#: size visible per row.
SCENE_FEATURES = ["n_pool", "lat_pitch_min", "lat_pitch_max", "lat_det"]

ALL_FEATURES = CHEAP_FEATURES + SCENE_FEATURES + RCC_FEATURES


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


def _ranks(values: np.ndarray) -> np.ndarray:
    """Dense descending rank; NaN sorts last and receives the worst rank."""
    v = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(-v, kind="mergesort")
    out = np.empty(v.size, dtype=np.float64)
    out[order] = np.arange(v.size, dtype=np.float64)
    return out


def extract_pair(ref: np.ndarray, search: np.ndarray, rec: dict,
                 delta: float) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Return (features, distances-to-truth, timing) for one pair's pool."""
    gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
    tscale = 10.0 / float(rec["gt_scale"])
    trot = float(rec["gt_theta"])

    t0 = time.perf_counter()
    cm = response_maps(ref, search, tscale, trot)
    cands = harvest(cm, delta=delta)
    t_cheap = time.perf_counter() - t0
    if not cands:
        return None

    n = len(cands)
    xs = np.array([c["x"] for c in cands], dtype=np.float64)
    ys = np.array([c["y"] for c in cands], dtype=np.float64)
    chan = {c: np.array([r[c] for r in cands], dtype=np.float64) for c in CHANNELS}

    lat = estimate_lattice(search)
    B = lat.basis
    det = float(abs(np.linalg.det(B)))
    p1, p2 = float(np.linalg.norm(B[:, 0])), float(np.linalg.norm(B[:, 1]))
    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        Binv = None

    best_stack = np.vstack([chan[c] for c in CHANNELS])
    ch_best = np.nanmax(np.where(np.isfinite(best_stack), best_stack, -np.inf), axis=0)
    ch_best = np.where(np.isfinite(ch_best), ch_best, np.nan)
    argmax_i = int(np.nanargmax(ch_best)) if np.isfinite(ch_best).any() else 0
    ax, ay = xs[argmax_i], ys[argmax_i]

    cols: dict[str, np.ndarray] = {}
    for c in CHANNELS:
        v = chan[c]
        cols[f"ch_{c}"] = v
        cols[f"rank_{c}"] = _ranks(v)
        cols[f"delta_{c}"] = v - np.nanmax(v) if np.isfinite(v).any() else np.full(n, np.nan)
    cols["ch_best"] = ch_best
    cols["ch_best_rank"] = _ranks(ch_best)
    cols["ch_best_delta"] = ch_best - np.nanmax(ch_best)
    stack = np.where(np.isfinite(best_stack), best_stack, np.nan)
    cols["ch_spread"] = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    cols["votes"] = np.array([float(r.get("votes", 0)) for r in cands])

    # Displacement from the global argmax, expressed in lattice periods. A
    # periodic alias sits at an integer number of periods from any other
    # alias; a genuinely different site generally does not.
    if Binv is not None:
        d = np.vstack([xs - ax, ys - ay])
        uv = Binv @ d
        cols["lat_dx_periods"] = uv[0]
        cols["lat_dy_periods"] = uv[1]
        frac = np.abs(uv - np.rint(uv))
        cols["lat_phase_res"] = np.hypot(frac[0], frac[1])
        cols["lat_parity_x"] = np.abs(np.rint(uv[0])) % 2.0
        cols["lat_parity_y"] = np.abs(np.rint(uv[1])) % 2.0
    else:
        for k in ("lat_dx_periods", "lat_dy_periods", "lat_phase_res",
                  "lat_parity_x", "lat_parity_y"):
            cols[k] = np.full(n, np.nan)

    cols["dist_to_argmax"] = np.hypot(xs - ax, ys - ay)
    h, w = search.shape[:2]
    cols["dist_to_centre"] = np.hypot(xs - (w - 1) / 2.0, ys - (h - 1) / 2.0)

    cols["n_pool"] = np.full(n, float(n))
    cols["lat_pitch_min"] = np.full(n, min(p1, p2))
    cols["lat_pitch_max"] = np.full(n, max(p1, p2))
    cols["lat_det"] = np.full(n, det)

    t1 = time.perf_counter()
    scorer = build_scorer(ref, search, scale=tscale, rotation=trot)
    t_build = time.perf_counter() - t1
    t2 = time.perf_counter()
    if scorer is None:
        for k in RCC_FEATURES:
            cols[k] = np.full(n, np.nan)
    else:
        feats = [scorer.score(x, y) for x, y in zip(xs, ys)]
        for k in RCC_FEATURES:
            cols[k] = np.array([f[k] for f in feats], dtype=np.float64)
    t_rcc = time.perf_counter() - t2

    table = np.column_stack([cols[k] for k in ALL_FEATURES]).astype(np.float32)
    dist = np.hypot(xs - gx, ys - gy).astype(np.float32)
    timing = {"t_cheap": t_cheap, "t_rcc_build": t_build, "t_rcc_score": t_rcc,
              "n": n, "has_rcc": scorer is not None}
    return table, dist, timing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="p2_val")
    ap.add_argument("--root", default=str(PROJECT / "data" / "phase2"))
    ap.add_argument("--per-cell", type=int, default=None)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    records = [json.loads(line) for line in open(split_dir / "manifest.jsonl")]
    subset = stratified([r for r in records if r["present"]], args.per_cell)
    tag = f"{args.split}_d{args.delta:g}"
    out = Path(args.out) if args.out else PROJECT / "results" / f"prerank_{tag}.npz"
    print(f"{args.split}: {len(subset)} present pairs, delta={args.delta}", flush=True)

    tables, dists, offsets, meta = [], [], [0], []
    for i, rec in enumerate(subset, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        got = extract_pair(ref, search, rec, args.delta)
        if got is None:
            print(f"[{i}/{len(subset)}] {rec['id']} empty pool -> skipped", flush=True)
            continue
        table, dist, timing = got
        tables.append(table)
        dists.append(dist)
        offsets.append(offsets[-1] + table.shape[0])
        meta.append({"id": rec["id"], "architecture": rec["architecture"],
                     "severity": rec["severity"], "preset_family": rec["preset_family"],
                     "d_min": float(dist.min()), **timing})
        print(f"[{i}/{len(subset)}] {rec['id']} {rec['architecture']} sev{rec['severity']} "
              f"n={timing['n']} d_min={dist.min():.2f} "
              f"cheap={timing['t_cheap']:.2f}s rcc={timing['t_rcc_build'] + timing['t_rcc_score']:.2f}s",
              flush=True)

    if not tables:
        print("no pairs extracted")
        return 1
    X = np.vstack(tables)
    d = np.concatenate(dists)
    np.savez_compressed(
        out, X=X, d=d, offsets=np.array(offsets, dtype=np.int64),
        features=np.array(ALL_FEATURES), meta=np.array(json.dumps(meta)),
        hit_px=np.float32(HIT_PX), delta=np.float32(args.delta),
    )
    recall = float(np.mean([m["d_min"] <= HIT_PX for m in meta]))
    print(f"\npairs={len(meta)} rows={X.shape[0]} features={X.shape[1]} "
          f"pool_recall={recall:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
