#!/usr/bin/env python3
"""RCC-v0: does anchor-displacement *agreement* predict the true site?

The experiment is deliberately the smallest one that can kill the idea. It
freezes the candidate pool, so both selectors see exactly the same sites and no
change in harvesting can masquerade as a change in discrimination:

    pool                 compute_candidate_rows at ground-truth pose
    shortlist            top-K of that pool by best cheap channel score
    label                candidate within 1.5 px of ground truth
    baseline selector    rank the shortlist by best channel score
    RCC selector         rank the shortlist by one RCC feature
    reference point      what locate_v2 actually answered on the same pool

Ground-truth pose is used throughout, because the question under test is
periodic identity, not pose estimation. Adding pose error here would confound
the only thing being measured.

Two numbers decide it. Per-feature AUC says whether the signal exists at all;
top-1 selection-given-pool says whether it survives being turned into a
decision. A feature can have a respectable AUC and still never win the argmax
on the pairs that matter, which is why both are reported.

The dense ``m0/m50/m30`` residual path is deliberately NOT run. RCC is being
tested as a *replacement* for it, so its cost is reported next to the cost of
the stage it would displace.
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

from driftforge.channels import CHANNELS
from driftforge.pipeline import compute_candidate_rows
from driftforge.rcc import FEATURES, build_scorer

#: Shortlist sizes whose oracle recall is reported. K must be chosen from
#: measured recall, not convenience -- the coarse rank of the true site has
#: previously been observed in the hundreds.
K_GRID = (32, 64, 128, 256, 512, 1024)

#: A candidate this close to ground truth is the true site. Candidates are
#: integer-pixel local maxima, so anything under ~0.71 px is unreachable.
HIT_PX = 1.5

#: Features where a LOWER value should indicate the true site.
LOWER_IS_BETTER = {"rcc_disp_mad", "rcc_disp_p90"}


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def channel_best(row: dict) -> float:
    vals = [row[c] for c in CHANNELS
            if isinstance(row.get(c), float) and math.isfinite(row[c])]
    return max(vals) if vals else float("-inf")


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC; 0.5 means the feature carries no ordering signal."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(both.size, dtype=np.float64)
    ranks[order] = np.arange(1, both.size + 1, dtype=np.float64)
    # average ranks over ties, or tied features report spurious separation
    srt = both[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = pos.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * neg.size))


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
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--shortlist", type=int, default=512,
                    help="K used for the RCC selector comparison")
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--out", default=str(PROJECT / "results" / "rcc_v0.json"))
    args = ap.parse_args()

    split_dir = Path(args.root) / args.split
    records = [json.loads(line) for line in open(split_dir / "manifest.jsonl")]
    subset = stratified([r for r in records if r["present"]], args.per_cell)
    print(f"{args.split}: evaluating {len(subset)} present pairs", flush=True)

    pos_feat: dict[str, list[float]] = defaultdict(list)
    neg_feat: dict[str, list[float]] = defaultdict(list)
    per_pair: list[dict] = []
    shortlist_recall = {k: [] for k in K_GRID}
    t_build_all, t_score_all, n_scored_all = [], [], []

    for i, rec in enumerate(subset, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
        tscale = 10.0 / float(rec["gt_scale"])
        trot = float(rec["gt_theta"])

        pool = compute_candidate_rows(ref, search, template_scale=tscale,
                                      template_rotation=trot)
        if not pool:
            continue
        for r in pool:
            r["_d"] = math.hypot(r["x"] - gx, r["y"] - gy)
            r["_best"] = channel_best(r)
        pool.sort(key=lambda r: -r["_best"])
        d_pool = min(r["_d"] for r in pool)

        for k in K_GRID:
            head = pool[:k]
            shortlist_recall[k].append(min(r["_d"] for r in head) <= HIT_PX)

        short = pool[:args.shortlist]
        in_short = min(r["_d"] for r in short) <= HIT_PX

        t0 = time.perf_counter()
        scorer = build_scorer(ref, search, scale=tscale, rotation=trot,
                              n_anchors=args.anchors)
        t_build = time.perf_counter() - t0
        if scorer is None:
            per_pair.append({"id": rec["id"], "architecture": rec["architecture"],
                             "severity": rec["severity"], "n_pool": len(pool),
                             "d_pool": d_pool, "in_shortlist": bool(in_short),
                             "scorer": None})
            print(f"[{i}/{len(subset)}] {rec['id']} no lattice -> skipped", flush=True)
            continue

        t1 = time.perf_counter()
        feats = [scorer.score(r["x"], r["y"]) for r in short]
        t_score = time.perf_counter() - t1
        t_build_all.append(t_build)
        t_score_all.append(t_score)
        n_scored_all.append(len(short))

        # Accumulate separability only over pairs where the true site is
        # actually present in the shortlist; elsewhere there is no positive.
        row = {"id": rec["id"], "architecture": rec["architecture"],
               "severity": rec["severity"], "n_pool": len(pool),
               "n_short": len(short), "d_pool": d_pool,
               "in_shortlist": bool(in_short),
               "t_build": t_build, "t_score": t_score,
               "baseline_d": float(short[0]["_d"]), "rcc": {}}
        if in_short:
            hit = np.array([r["_d"] <= HIT_PX for r in short])
            for key in FEATURES:
                vals = np.array([f[key] for f in feats], dtype=np.float64)
                ok = np.isfinite(vals)
                if not ok.any():
                    continue
                sign = -1.0 if key in LOWER_IS_BETTER else 1.0
                pos_feat[key].extend((sign * vals[ok & hit]).tolist())
                neg_feat[key].extend((sign * vals[ok & ~hit]).tolist())
                # top-1 under this feature alone, ties broken by channel score
                order = np.lexsort((
                    -np.array([r["_best"] for r in short]),
                    -np.where(ok, sign * vals, -np.inf),
                ))
                row["rcc"][key] = float(short[int(order[0])]["_d"])
        per_pair.append(row)
        print(f"[{i}/{len(subset)}] {rec['id']} {rec['architecture']} sev{rec['severity']} "
              f"pool={len(pool)} short={len(short)} in_short={in_short} "
              f"build={t_build:.2f}s score={t_score:.2f}s", flush=True)

    scored = [r for r in per_pair if r.get("in_shortlist") and r.get("rcc")]
    summary = {
        "split": args.split,
        "pairs": len(per_pair),
        "pairs_with_positive_in_shortlist": len(scored),
        "shortlist_oracle_recall": {
            f"K={k}": round(float(np.mean(v)), 4) for k, v in shortlist_recall.items()
            if v
        },
        "feature_auc": {
            k: round(auc(np.array(pos_feat[k]), np.array(neg_feat[k])), 4)
            for k in FEATURES if pos_feat.get(k)
        },
        "top1_given_positive_in_shortlist": {
            "baseline_channel_score": round(
                float(np.mean([r["baseline_d"] <= HIT_PX for r in scored])), 4
            ) if scored else None,
            **{
                k: round(float(np.mean([r["rcc"][k] <= HIT_PX
                                        for r in scored if k in r["rcc"]])), 4)
                for k in FEATURES
                if scored and any(k in r["rcc"] for r in scored)
            },
        },
        "runtime_s": {
            "rcc_build_median": round(float(np.median(t_build_all)), 3) if t_build_all else None,
            "rcc_score_median": round(float(np.median(t_score_all)), 3) if t_score_all else None,
            "candidates_scored_median": float(np.median(n_scored_all)) if n_scored_all else None,
            "us_per_candidate": round(
                float(np.sum(t_score_all) / max(1, np.sum(n_scored_all)) * 1e6), 1
            ) if t_score_all else None,
        },
    }

    out = Path(args.out)
    out.write_text(json.dumps({"summary": summary, "pairs": per_pair}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
