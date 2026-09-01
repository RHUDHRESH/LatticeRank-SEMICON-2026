#!/usr/bin/env python3
"""Sweep the harvest threshold ``delta`` against oracle recall and pool size.

The 8-pair smoke put detection recall at 100% and harvest recall at 87.5%, with
``MAX_CAND`` never binding. That places the loss squarely on the per-channel
``S >= S_max - delta`` cut: the true site *is* a local maximum in some channel,
and the threshold throws it away before any selector can rank it. No amount of
downstream discrimination recovers a candidate that was never harvested, so
this one constant may be worth more than a new feature family.

The response maps are computed once per pair and re-harvested at every delta,
which makes the sweep nearly free relative to the maps themselves. Everything
runs at ground-truth pose so that pose error cannot confound the recall figure.

Cost is reported alongside recall because the two trade directly: a larger
delta admits more of the response surface, and the candidate count drives both
the feature table and the selector's runtime, which is already over the 20 s
timeout on part of the distribution.
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

from driftforge.channels import harvest, response_maps
from driftforge.model import MAX_CANDIDATES

DELTAS = (0.10, 0.125, 0.15, 0.175, 0.20)
HIT_PX = 1.5


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="p2_val")
    ap.add_argument("--root", default=str(PROJECT / "data" / "phase2"))
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--out", default=str(PROJECT / "results" / "delta_sweep.json"))
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    records = [json.loads(line) for line in open(split_dir / "manifest.jsonl")]
    subset = stratified([r for r in records if r["present"]], args.per_cell)
    print(f"{args.split}: {len(subset)} present pairs", flush=True)

    per_delta: dict[float, dict[str, list]] = {
        d: {"hit": [], "n": [], "capped": []} for d in DELTAS
    }
    rows: list[dict] = []

    for i, rec in enumerate(subset, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
        tscale = 10.0 / float(rec["gt_scale"])
        trot = float(rec["gt_theta"])

        t0 = time.perf_counter()
        cm = response_maps(ref, search, tscale, trot)
        t_maps = time.perf_counter() - t0

        row = {"id": rec["id"], "architecture": rec["architecture"],
               "severity": rec["severity"], "t_maps": t_maps, "by_delta": {}}
        for d in DELTAS:
            cands = harvest(cm, delta=d)
            best = min((math.hypot(c["x"] - gx, c["y"] - gy) for c in cands),
                       default=float("inf"))
            # MAX_CANDIDATES is applied after harvest, sorted by best channel
            # score, so a larger delta can only cost recall through that cap.
            per_delta[d]["hit"].append(best <= HIT_PX)
            per_delta[d]["n"].append(len(cands))
            per_delta[d]["capped"].append(len(cands) > MAX_CANDIDATES)
            row["by_delta"][str(d)] = {"n": len(cands), "d_best": best}
        rows.append(row)
        line = "  ".join(f"d{d}:{row['by_delta'][str(d)]['n']}" for d in DELTAS)
        print(f"[{i}/{len(subset)}] {rec['id']} {rec['architecture']} "
              f"sev{rec['severity']} maps={t_maps:.1f}s  {line}", flush=True)

    summary = {
        "split": args.split,
        "pairs": len(rows),
        "hit_px": HIT_PX,
        "max_candidates": MAX_CANDIDATES,
        "by_delta": {
            str(d): {
                "harvest_recall": round(float(np.mean(v["hit"])), 4),
                "n_median": float(np.median(v["n"])),
                "n_p95": float(np.percentile(v["n"], 95)),
                "n_max": int(np.max(v["n"])),
                "pairs_over_max_cand": int(np.sum(v["capped"])),
            }
            for d, v in per_delta.items()
        },
        "t_maps_median": round(float(np.median([r["t_maps"] for r in rows])), 3),
    }
    out = Path(args.out)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
