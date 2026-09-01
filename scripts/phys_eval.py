#!/usr/bin/env python3
"""Evaluate physical-evidence modules on frozen candidate shortlists.

Every ``driftforge/phys/exp*.py`` module is discovered automatically and asked
the same question: do its columns help the ranker pick the true site out of the
candidates the declared method already produced? Nothing here generates or
reorders candidates, so a module can only ever be judged on discrimination --
which is the doctrine three killed experiments earned:

    New physical evidence never vetoes baseline candidate generation unless it
    independently proves higher recall.

Two numbers per module. **Per-feature AUC** says whether the signal exists at
all, separating true sites from the wrong candidates that actually threaten
them. **Incremental top-1** says whether it survives being turned into a
decision, by refitting the same ranker family with and without the module's
columns. A feature can carry respectable AUC and still never move an argmax,
so a module is only interesting when both move.

Candidates are shortlisted to the top K by channel score before physical
scoring, because the extractors cost milliseconds and a pool can hold 6,000
sites. Pairs whose true site does not survive into the shortlist are excluded
from the discrimination metrics rather than counted as losses -- shortlist
recall is a proposal-stage property, already measured, and mixing it in here
would let a recall change masquerade as a discrimination change.

The true site is never injected into a shortlist it did not earn.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pkgutil
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
from driftforge.phys.context import build_context
from driftforge.rcc import FEATURES as RCC_FEATURES
from driftforge.rcc import build_scorer

HIT_PX = 1.5
SHORTLIST = 256


def discover() -> dict:
    """Import every exp* module in driftforge.phys that honours the contract."""
    import driftforge.phys as pkg
    mods = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("exp"):
            continue
        try:
            m = importlib.import_module(f"driftforge.phys.{info.name}")
        except Exception as exc:                      # a broken module is data
            print(f"  ! {info.name}: import failed: {exc}", flush=True)
            continue
        if all(hasattr(m, a) for a in ("FEATURES", "build", "score")):
            mods[info.name] = m
        else:
            print(f"  ! {info.name}: missing FEATURES/build/score", flush=True)
    return mods


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


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-corrected rank AUC. 0.5 is no ordering signal."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(both.size)
    ranks[order] = np.arange(1, both.size + 1)
    srt = both[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n = pos.size
    return float((ranks[:n].sum() - n * (n + 1) / 2.0) / (n * neg.size))


def within_scene_auc(pack: dict, k: int) -> tuple[float, int]:
    """Macro-averaged AUC: compute per scene, then average across scenes.

    A pooled AUC over every shortlist row can be earned entirely by
    between-scene differences -- a column that is constant inside each scene but
    differs across scenes will score well and rank nothing. ``uc_sigma_median``
    did exactly that at a pooled 0.638. Only within-scene ordering is what a
    candidate feature is asked for, so only within-scene ordering is scored.
    """
    off = pack["offsets"]
    vals = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        dd = pack["d"][a:b]
        hit = dd <= HIT_PX
        if not hit.any() or hit.all():
            continue
        v = pack["X"][a:b, k].astype(np.float64)
        a_ = auc(v[hit], v[~hit])
        if np.isfinite(a_):
            vals.append(a_)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def within_scene_std(pack: dict, k: int) -> float:
    """Median across scenes of the feature's spread inside one shortlist."""
    off = pack["offsets"]
    stds = []
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        v = pack["X"][a:b, k].astype(np.float64)
        v = v[np.isfinite(v)]
        if v.size > 1:
            stds.append(float(np.std(v)))
    return float(np.median(stds)) if stds else 0.0


def extract(split_dir: Path, records: list[dict], mods: dict,
            shortlist: int) -> dict:
    """One pass: pools, RCC, and every discovered physical module."""
    phys_cols = [c for m in mods.values() for c in m.FEATURES]
    cols = [f"ch_{c}" for c in CHANNELS] + ["ch_best"] + list(RCC_FEATURES) + phys_cols
    X, d, offsets, meta = [], [], [0], []
    timing: dict[str, list[float]] = defaultdict(list)

    for i, rec in enumerate(records, 1):
        ref = load_gray(split_dir / rec["ref_image"])
        search = load_gray(split_dir / rec["search_image"])
        gx, gy = float(rec["gt_x"]), float(rec["gt_y"])
        s = float(rec["gt_scale"])
        rot = float(rec["gt_theta"])

        cm = response_maps(ref, search, 10.0 / s, rot)
        cands = harvest(cm, delta=0.10)
        if not cands:
            continue
        for c in cands:
            c["_b"] = max((c[ch] for ch in CHANNELS
                           if isinstance(c.get(ch), float) and math.isfinite(c[ch])),
                          default=-np.inf)
        cands.sort(key=lambda c: -c["_b"])
        short = cands[:shortlist]
        dist = np.array([math.hypot(c["x"] - gx, c["y"] - gy) for c in short])

        ctx = build_context(ref, search, scale=s, rotation=rot)
        rcc = build_scorer(ref, search, scale=10.0 / s, rotation=rot)
        states = {}
        for name, m in mods.items():
            t0 = time.perf_counter()
            try:
                states[name] = m.build(ctx)
            except Exception as exc:
                print(f"  ! {name}.build failed on {rec['id']}: {exc}", flush=True)
                states[name] = None
            timing[f"{name}_build"].append(time.perf_counter() - t0)

        rows = np.full((len(short), len(cols)), np.nan, dtype=np.float32)
        for j, c in enumerate(short):
            k = 0
            for ch in CHANNELS:
                rows[j, k] = c.get(ch, np.nan); k += 1
            rows[j, k] = c["_b"]; k += 1
            f = rcc.score(c["x"], c["y"]) if rcc is not None else {}
            for key in RCC_FEATURES:
                rows[j, k] = f.get(key, np.nan); k += 1
            for name, m in mods.items():
                st = states.get(name)
                got = {}
                if st is not None:
                    try:
                        got = m.score(st, c["x"], c["y"])
                    except Exception as exc:
                        if j == 0:
                            print(f"  ! {name}.score failed on {rec['id']}: {exc}",
                                  flush=True)
                for key in m.FEATURES:
                    rows[j, k] = got.get(key, np.nan); k += 1

        X.append(rows)
        d.append(dist.astype(np.float32))
        offsets.append(offsets[-1] + rows.shape[0])
        meta.append({"id": rec["id"], "architecture": rec["architecture"],
                     "severity": rec["severity"], "n_pool": len(cands),
                     "d_min": float(dist.min())})
        print(f"[{i}/{len(records)}] {rec['id']} {rec['architecture']} "
              f"sev{rec['severity']} pool={len(cands)} in_short={dist.min() <= HIT_PX}",
              flush=True)

    return {"X": np.vstack(X), "d": np.concatenate(d),
            "offsets": np.array(offsets), "cols": cols, "meta": meta,
            "timing": {k: float(np.median(v)) for k, v in timing.items()}}


def top1(pack: dict, scores: np.ndarray) -> float:
    """Fraction of shortlists whose argmax is the true site."""
    hits = []
    off = pack["offsets"]
    for i in range(len(off) - 1):
        a, b = int(off[i]), int(off[i + 1])
        dd = pack["d"][a:b]
        if dd.min() > HIT_PX:
            continue                       # true site never reached the shortlist
        s = scores[a:b]
        j = int(np.argmax(np.where(np.isfinite(s), s, -np.inf)))
        hits.append(dd[j] <= HIT_PX)
    return float(np.mean(hits)) if hits else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(PROJECT / "data" / "phase2"))
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--shortlist", type=int, default=SHORTLIST)
    ap.add_argument("--exclude", default="",
                    help="comma-separated module names to skip (e.g. one whose "
                         "per-candidate cost makes it infeasible to measure)")
    ap.add_argument("--only", default="",
                    help="comma-separated module names; overrides --exclude")
    ap.add_argument("--out", default=str(PROJECT / "results" / "phys_eval.json"))
    args = ap.parse_args()

    print("discovering physical-evidence modules...")
    mods = discover()
    if args.only:
        keep = {k.strip() for k in args.only.split(",") if k.strip()}
        mods = {k: v for k, v in mods.items() if k in keep}
    elif args.exclude:
        drop = {k.strip() for k in args.exclude.split(",") if k.strip()}
        for k in drop & set(mods):
            print(f"  - {k}: excluded by request")
        mods = {k: v for k, v in mods.items() if k not in drop}
    if not mods:
        print("no modules found; nothing to evaluate")
        return 1
    for name, m in mods.items():
        print(f"  + {name}: {len(m.FEATURES)} features")

    packs = {}
    for split in ("p2_train", "p2_val"):
        sd = Path(args.root) / split
        recs = stratified([json.loads(l) for l in open(sd / "manifest.jsonl")
                           if json.loads(l)["present"]], args.per_cell)
        print(f"\n=== {split}: {len(recs)} pairs ===", flush=True)
        packs[split] = extract(sd, recs, mods, args.shortlist)

    tr, va = packs["p2_train"], packs["p2_val"]
    cols = va["cols"]

    # -- drop columns that cannot carry information -------------------------
    # A constant column and an all-NaN column are both silent failures: the
    # module reported a value it never actually measured. Detect them here
    # rather than trusting each module's own FEATURES list, so the same guard
    # covers every extractor added later.
    dead: dict[str, str] = {}
    for k, name in enumerate(cols):
        v = va["X"][:, k].astype(np.float64)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            dead[name] = "all NaN"
        elif finite.size < 0.5 * v.size:
            dead[name] = f"NaN on {100 * (1 - finite.size / v.size):.0f}% of rows"
        elif np.unique(finite).size <= 1:
            dead[name] = "constant"
        else:
            # constant WITHIN every scene carries no within-shortlist ordering,
            # which is the only thing a candidate feature is asked for
            per = []
            for i in range(len(va["offsets"]) - 1):
                a, b = int(va["offsets"][i]), int(va["offsets"][i + 1])
                seg = v[a:b][np.isfinite(v[a:b])]
                per.append(np.unique(seg).size <= 1 if seg.size else True)
            if all(per):
                dead[name] = "constant within every scene"
    if dead:
        print("")
        print("dead columns (excluded from all fits):")
        for name, why in dead.items():
            print(f"  - {name:28s} {why}")
    live = [c for c in cols if c not in dead]
    ytr = (tr["d"] <= HIT_PX).astype(int)
    hit_va = va["d"] <= HIT_PX

    # -- per-feature AUC on val, true site vs the wrong candidates that
    #    actually reached the same shortlist ------------------------------
    aucs, pooled_aucs, wstd = {}, {}, {}
    for k, name in enumerate(cols):
        v = va["X"][:, k].astype(np.float64)
        pa = auc(v[hit_va], v[~hit_va])
        pooled_aucs[name] = round(float(pa), 4) if np.isfinite(pa) else None
        wa, n_sc = within_scene_auc(va, k)
        aucs[name] = round(float(wa), 4) if np.isfinite(wa) else None
        wstd[name] = round(within_scene_std(va, k), 6)

    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_score(keys: list[str]) -> float:
        idx = [cols.index(k) for k in keys]
        w = np.where(ytr == 1, 200.0, 1.0)
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                           min_samples_leaf=40, random_state=0)
        m.fit(np.nan_to_num(tr["X"][:, idx], nan=0.0), ytr, sample_weight=w)
        return top1(va, m.predict_proba(np.nan_to_num(va["X"][:, idx], nan=0.0))[:, 1])

    base_keys = [c for c in ([f"ch_{c}" for c in CHANNELS] + ["ch_best"]
                             + list(RCC_FEATURES)) if c in live]
    base = fit_score(base_keys)
    print(f"\nbaseline (cheap + RCC) top-1 given in-shortlist: {base:.4f}")

    incremental = {}
    for name, m in mods.items():
        keys = [c for c in m.FEATURES if c in live]
        if not keys:
            incremental[name] = {"top1": None, "delta_pp": None,
                                 "note": "every feature dead"}
            print(f"  + {name:22s} SKIPPED: every feature dead")
            continue
        v = fit_score(base_keys + keys)
        incremental[name] = {"top1": round(v, 4),
                             "delta_pp": round(100.0 * (v - base), 2)}
        print(f"  + {name:22s} top1={v:.4f}  ({100 * (v - base):+.2f} pp)")
    # Per-feature conditional information. A feature at AUC 0.80 that adds
    # nothing over existing evidence is dead; one at 0.66 that adds 5 pp is
    # valuable. exp16_irls scored -0.752 and reproduced rcc_disp_p90 at -0.754,
    # which is exactly the case this column exists to expose.
    per_feature = {}
    print("")
    print("per-feature incremental over baseline:")
    for name, m in mods.items():
        for c in m.FEATURES:
            if c not in live:
                continue
            v = fit_score(base_keys + [c])
            per_feature[c] = round(100.0 * (v - base), 2)
            print(f"    {c:30s} {per_feature[c]:+6.2f} pp")

    allk = base_keys + [c for m in mods.values() for c in m.FEATURES if c in live]
    combined = fit_score(allk)
    print(f"  + ALL modules           top1={combined:.4f}  "
          f"({100 * (combined - base):+.2f} pp)")

    report = {
        "shortlist": args.shortlist, "hit_px": HIT_PX,
        "train_pairs": len(tr["meta"]), "val_pairs": len(va["meta"]),
        "val_in_shortlist": int(sum(m["d_min"] <= HIT_PX for m in va["meta"])),
        "modules": {n: list(m.FEATURES) for n, m in mods.items()},
        "baseline_top1": round(base, 4),
        "incremental": incremental,
        "combined_top1": round(combined, 4),
        "combined_delta_pp": round(100.0 * (combined - base), 2),
        "feature_auc_within_scene": aucs,
        "feature_auc_pooled": pooled_aucs,
        "within_scene_std": wstd,
        "per_feature_incremental_pp": per_feature,
        "dead_columns": dead,
        "build_timing_median_s": va["timing"],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
