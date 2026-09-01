#!/usr/bin/env python3
"""Empirically verify the Phase 2 ground-truth conventions (prompt §2.3).

For a set of present pairs, brute-force

- the rotation theta, over a grid that covers the full net span, and
- the zoom s, over [7.5, 12.5],

by maximizing ZNCC at the *known* ground-truth location, then assert the
recovered values match the manifest labels (0.15 deg / 0.5%). The label is
whatever this script says it is: if the assertion fails, fix the label in
``driftforge/phase2.py::net_rotation_label`` (or the generator), not the
script.

Usage: python scripts/verify_conventions.py [--pairs 20] [--seed-base 1100000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_phase2_sample
from driftforge.pose import rotation_oracle, scale_oracle


def verify(pairs: int, seed_base: int, theta_tol: float = 0.15, scale_rel_tol: float = 0.005) -> bool:
    rng = np.random.default_rng(2026_08_29)
    seeds = rng.choice(np.arange(seed_base, seed_base + 4000), size=pairs, replace=False)
    failures = 0
    print(f"{'seed':>9} {'label θ':>9} {'oracle θ':>9} {'dθ':>7} {'label s':>8} "
          f"{'oracle s':>9} {'ds%':>7} {'score':>6}")
    for seed in sorted(int(s) for s in seeds):
        sample = generate_phase2_sample(seed, split="p2_train", present=True)
        if sample.gt_x is None:
            failures += 1
            print(f"{seed:>9}  ABSENT although requested present")
            continue
        rec_theta, theta_score = rotation_oracle(
            sample.reference, sample.search, sample.gt_x, sample.gt_y, sample.gt_scale
        )
        rec_scale, scale_score = scale_oracle(
            sample.reference, sample.search, sample.gt_x, sample.gt_y, rec_theta,
            shape_scale=sample.gt_scale,
        )
        d_theta = rec_theta - sample.gt_theta
        d_scale_pct = (rec_scale - sample.gt_scale) / sample.gt_scale * 100.0
        ok_theta = abs(d_theta) <= theta_tol
        ok_scale = abs(d_scale_pct) <= scale_rel_tol * 100.0
        if not (ok_theta and ok_scale):
            failures += 1
        print(f"{seed:>9} {sample.gt_theta:>9.3f} {rec_theta:>9.3f} {d_theta:>7.3f} "
              f"{sample.gt_scale:>8.3f} {rec_scale:>9.3f} {d_scale_pct:>7.3f} "
              f"{max(theta_score, scale_score):>6.3f}" + ("" if ok_theta and ok_scale else "  FAIL"))
    print(f"\n{pairs - failures}/{pairs} pairs within tolerance "
          f"(theta {theta_tol} deg, scale {scale_rel_tol:.1%})")
    return failures == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1_100_000)
    args = parser.parse_args(argv)
    ok = verify(args.pairs, args.seed_base)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
