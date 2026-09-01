#!/usr/bin/env python3
"""Measure the Phase 1 selector's ceiling on Phase 2 data at GROUND-TRUTH pose.

The question this answers is narrow and deliberate: with the zoom and rotation
handed to it for free, how often does the declared Phase 1 chain put the answer
inside 1 px? That number is the ceiling any pose estimator can aim at, and the
baseline any new selector (RCC) has to beat. Pose error is excluded by
construction, so what remains is periodic identity -- the thing under test.

Three recall gates are instrumented separately, because a selector can only
re-rank what survived harvesting:

    detect   a local maximum in at least one channel response
    harvest  survives the per-channel ``S >= S_max - delta`` cut and the
             4000-per-channel harvest cap
    pool     survives ``MAX_CAND`` in ``compute_candidate_rows``

Candidates are integer-pixel local maxima, so a true site sitting at a subpixel
offset is at best ~0.71 px from the nearest grid point. Recall is therefore
read at 1.5 px; the 1 px column is the selection metric, not the recall metric.
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

from driftforge.channels import harvest, response_maps
from driftforge.model import load_model_bundle
from driftforge.pipeline import DELTA, compute_candidate_rows, locate_v2

BANDS = (1.0, 1.5, 2.0, 3.0, 5.0)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def nearest(rows, gx: float, gy: float) -> float:
    """Distance from ground truth to the closest candidate, or inf."""
    best = float("inf")
    for r in rows:
        d = math.hypot(r["x"] - gx, r["y"] - gy)
        if d < best:
            best = d
    return best


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
    ap.add_argument("--per-cell", type=int, default=None,
                    help="pairs per (architecture, severity) cell; omit for all")
    ap.add_argument("--out", default=str(PROJECT / "results" / "gt_pose_ceiling.json"))
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    records = [json.loads(line) for line in open(split_dir / "manifest.jsonl")]
    present = [r for r in records if r["present"]]
    subset = stratified(present, args.per_cell)
    print(f"{args.split}: {len(present)} present pairs, evaluating {len(subset)}",
          flush=True)

    bundle = load_model_bundle()
    rows_out: list[dict] = []

    for i, rec in enumerate(subset, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
        # _template_from_reference's convention: internal zoom = 0.1 * template_scale,
        # so a true down-scaling of gt_scale needs template_scale = 10 / gt_scale.
        tscale = 10.0 / float(rec["gt_scale"])
        trot = float(rec["gt_theta"])

        t0 = time.perf_counter()
        cm = response_maps(ref, search, tscale, trot)
        detect = float("inf")
        for resp in cm.maps.values():
            loc = resp == ndimage.maximum_filter(resp, size=5, mode="nearest")
            ys, xs = np.nonzero(loc)
            if xs.size:
                d = float(np.min(np.hypot(xs + cm.half_w - gx, ys + cm.half_h - gy)))
                detect = min(detect, d)
        harvested = harvest(cm, delta=DELTA)
        t_harvest = time.perf_counter() - t0

        t1 = time.perf_counter()
        pool = compute_candidate_rows(ref, search, template_scale=tscale,
                                      template_rotation=trot)
        t_pool = time.perf_counter() - t1

        t2 = time.perf_counter()
        located = locate_v2(ref, search, model_bundle=bundle,
                            candidate_rows=pool,
                            template_scale=tscale, template_rotation=trot)
        t_select = time.perf_counter() - t2

        row = {
            "id": rec["id"],
            "architecture": rec["architecture"],
            "severity": rec["severity"],
            "preset_family": rec["preset_family"],
            "gt_scale": rec["gt_scale"],
            "gt_theta": rec["gt_theta"],
            "n_harvested": len(harvested),
            "n_pool": len(pool),
            "d_detect": detect,
            "d_harvest": nearest(harvested, gx, gy),
            "d_pool": nearest(pool, gx, gy),
            "d_select": math.hypot(located.x - gx, located.y - gy),
            "eq_set_size": located.eq_set_size,
            "selection_mode": located.diagnostics.get("selection_mode"),
            "t_harvest": t_harvest,
            "t_pool": t_pool,
            "t_select": t_select,
        }
        rows_out.append(row)
        print(f"[{i}/{len(subset)}] {rec['id']} {rec['architecture']} sev{rec['severity']} "
              f"pool={row['n_pool']} d_pool={row['d_pool']:.2f} "
              f"d_select={row['d_select']:.2f} t={t_pool + t_select:.1f}s", flush=True)

    def frac(key: str, band: float, rs=None) -> float:
        rs = rows_out if rs is None else rs
        return round(float(np.mean([r[key] <= band for r in rs])), 4) if rs else float("nan")

    totals = [r["t_pool"] + r["t_select"] for r in rows_out]
    in_pool = [r for r in rows_out if r["d_pool"] <= 1.5]
    summary = {
        "split": args.split,
        "pairs": len(rows_out),
        "recall": {
            stage: {f"le{b}px": frac(f"d_{stage}", b) for b in BANDS}
            for stage in ("detect", "harvest", "pool")
        },
        "selection": {f"le{b}px": frac("d_select", b) for b in BANDS},
        "selection_given_pool_1p5": {
            "n": len(in_pool),
            **{f"le{b}px": frac("d_select", b, in_pool) for b in BANDS},
        },
        "runtime_s": {
            "median_total": round(float(np.median(totals)), 3),
            "p95_total": round(float(np.percentile(totals, 95)), 3),
            "max_total": round(float(np.max(totals)), 3),
            "over_20s": int(sum(t > 20.0 for t in totals)),
        },
        "by_cell": {},
    }
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_out:
        cells[(r["architecture"], r["severity"])].append(r)
    for key in sorted(cells):
        rs = cells[key]
        summary["by_cell"][f"{key[0]}_sev{key[1]}"] = {
            "n": len(rs),
            "pool_le1.5px": frac("d_pool", 1.5, rs),
            "select_le1px": frac("d_select", 1.0, rs),
            "select_le5px": frac("d_select", 5.0, rs),
        }

    out = Path(args.out)
    out.write_text(json.dumps({"summary": summary, "rows": rows_out}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
