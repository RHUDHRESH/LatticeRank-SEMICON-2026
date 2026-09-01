#!/usr/bin/env python3
"""Hard runtime gate for one physical-evidence module. Run before claiming a cost.

Two modules have now reported per-candidate costs that were wrong in the
direction that matters. One claimed "well under 3 ms" while measuring 3,561 us;
another claimed 600-650 us while measuring 603,576 us -- a factor of a thousand,
which turned a 4-minute evaluation into a 4-hour one. Both errors came from
timing a handful of hand-picked calls instead of a real workload.

So the cost of a module is not what its author believes. It is what this script
prints, on real harvested candidates, warmed up, repeated, and reported at the
median AND the p95 -- because a module whose median is cheap and whose tail is
not will still blow a per-pair deadline on the scenes that emit 6,000
candidates.

    <150 us        excellent
    150-300 us     acceptable
    300-600 us     must show significant accuracy to justify
    600-1000 us    only exceptional effects survive
    >1000 us       non-shippable unless redesigned

Usage:  python scripts/bench_module.py exp07_edge_profile
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, response_maps
from driftforge.phys.context import build_context

N_ROWS = 2000
WARMUP = 3
MEASURED = 5

BANDS = ((150, "excellent"), (300, "acceptable"),
         (600, "must show significant accuracy to justify"),
         (1000, "only exceptional effects survive"),
         (float("inf"), "NON-SHIPPABLE unless redesigned"))


def classify(us: float) -> str:
    for limit, label in BANDS:
        if us < limit:
            return label
    return "NON-SHIPPABLE unless redesigned"


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("module", help="module name under driftforge.phys, e.g. exp07_edge_profile")
    ap.add_argument("--root", default=str(PROJECT / "data" / "phase2" / "p2_val"))
    ap.add_argument("--rows", type=int, default=N_ROWS)
    args = ap.parse_args()

    mod = importlib.import_module(f"driftforge.phys.{args.module}")
    for attr in ("FEATURES", "build", "score"):
        if not hasattr(mod, attr):
            print(f"FAIL: module is missing {attr}")
            return 1

    root = Path(args.root)
    records = [json.loads(l) for l in open(root / "manifest.jsonl")]
    present = [r for r in records if r["present"]]

    # Gather real harvested candidates across several scenes until we have
    # enough rows; one scene's candidates are not a representative workload.
    work: list[tuple] = []
    scenes = 0
    for rec in present:
        if len(work) >= args.rows:
            break
        ref = load_gray(root / rec["ref_image"])
        search = load_gray(root / rec["search_image"])
        s, rot = float(rec["gt_scale"]), float(rec["gt_theta"])
        cm = response_maps(ref, search, 10.0 / s, rot)
        cands = harvest(cm, delta=0.10)
        if not cands:
            continue
        for c in cands:
            c["_b"] = max((c[ch] for ch in CHANNELS
                           if isinstance(c.get(ch), float) and math.isfinite(c[ch])),
                          default=-np.inf)
        cands.sort(key=lambda c: -c["_b"])
        t0 = time.perf_counter()
        ctx = build_context(ref, search, scale=s, rotation=rot)
        t_ctx = time.perf_counter() - t0
        t0 = time.perf_counter()
        state = mod.build(ctx)
        t_build = time.perf_counter() - t0
        if state is None:
            print(f"  {rec['id']}: build abstained")
            continue
        scenes += 1
        work.append((state, cands[:512], t_build, t_ctx))

    if not work:
        print("FAIL: module abstained on every scene; nothing to benchmark")
        return 1
    total_rows = sum(len(c) for _, c, _, _ in work)
    print(f"benchmarking {args.module} on {total_rows} real candidates "
          f"from {scenes} scenes")

    def one_pass() -> float:
        t0 = time.perf_counter()
        n = 0
        for state, cands, _, _ in work:
            for c in cands:
                mod.score(state, c["x"], c["y"])
                n += 1
        return (time.perf_counter() - t0) / n * 1e6

    for _ in range(WARMUP):
        one_pass()
    samples = np.array([one_pass() for _ in range(MEASURED)])

    # Per-candidate spread, measured once, for the tail figure.
    state, cands, _, _ = work[0]
    per = []
    for c in cands[:400]:
        t0 = time.perf_counter()
        mod.score(state, c["x"], c["y"])
        per.append((time.perf_counter() - t0) * 1e6)
    per = np.array(per)

    med = float(np.median(samples))
    builds = [b for _, _, b, _ in work]
    report = {
        "module": args.module,
        "features": list(mod.FEATURES),
        "scenes": scenes,
        "candidates": total_rows,
        "median_us_per_candidate": round(med, 1),
        "p95_us_per_candidate": round(float(np.percentile(per, 95)), 1),
        "pass_spread_us": round(float(samples.max() - samples.min()), 1),
        "build_median_s": round(float(np.median(builds)), 4),
        "verdict": classify(med),
        "cost_at_256_shortlist_s": round(med * 256 / 1e6, 3),
        "cost_at_4000_pool_s": round(med * 4000 / 1e6, 2),
    }
    print(json.dumps(report, indent=2))
    if med >= 1000:
        print("\nGATE FAILED: >1000 us/candidate is non-shippable. Redesign "
              "before requesting evaluation.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
