#!/usr/bin/env python3
"""Pre-ranker experiments 1, 2 and 4, all off one cached feature table.

The measurement that matters is **oracle recall at K**: after ranking a scene's
whole candidate pool, is the true site still inside the top K? That is the only
thing a pre-ranker owes the expensive stage downstream. Classification accuracy
is not reported because it is not the job -- a pre-ranker that is wrong about
which candidate is best but keeps the right one in the top 256 has done
everything asked of it.

Four rankers are compared on identical frozen pools:

    channel      rank by best raw/midband/directionality score (today's order)
    cheap        learned on the cheap channel + lattice features only
    rcc          learned on RCC features only
    hybrid       learned on both

``cheap`` exists as the control that decides the whole question. If ``hybrid``
beats ``channel`` but not ``cheap``, then the gain came from learning to rank
rather than from constellation evidence, and RCC has not earned its runtime.

Training uses ``p2_train`` and reporting uses ``p2_val``; the generator built
them scene-disjoint, so no re-slicing is done here. Negatives are reweighted
toward the ones that actually threaten -- candidates a channel ranker already
puts near the top -- because a pre-ranker trained against random background
learns a problem we do not have.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]

K_GRID = (32, 64, 128, 256, 384, 512, 1024)

RCC_KEYS = ("rcc_disp_p90", "rcc_disp_mad", "rcc_consensus_1p5",
            "rcc_consensus_2p5", "rcc_ncc_median", "rcc_ncc_p25",
            "rcc_ncc_min", "rcc_peak_margin_median")


class Cache:
    """One extracted split: stacked rows plus the per-pair row boundaries."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.X = z["X"]
        self.d = z["d"]
        self.offsets = z["offsets"]
        self.features = [str(f) for f in z["features"]]
        self.meta = json.loads(str(z["meta"]))
        self.hit_px = float(z["hit_px"])
        self.delta = float(z["delta"])

    @property
    def n_pairs(self) -> int:
        return len(self.offsets) - 1

    def pair(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.X[a:b], self.d[a:b]

    def columns(self, keys) -> np.ndarray:
        idx = [self.features.index(k) for k in keys]
        return np.asarray(idx, dtype=int)

    @property
    def y(self) -> np.ndarray:
        return (self.d <= self.hit_px).astype(np.int32)


def group_keys(cache: Cache, group: str) -> list[str]:
    rcc = [f for f in cache.features if f.startswith("rcc_")]
    cheap = [f for f in cache.features if not f.startswith("rcc_")]
    if group == "cheap":
        return cheap
    if group == "rcc":
        return [k for k in RCC_KEYS if k in cache.features]
    if group == "hybrid":
        return cheap + rcc
    raise ValueError(group)


def sample_weights(cache: Cache) -> np.ndarray:
    """Upweight positives, and negatives a channel ranker already likes.

    The dangerous negative is the one that already sits near the top of the
    order we are trying to improve on. Weighting uniformly would spend almost
    all of the model's capacity on the thousands of candidates no ranker was
    ever going to choose.
    """
    y = cache.y
    rank = cache.X[:, cache.features.index("ch_best_rank")]
    w = np.ones(y.size, dtype=np.float64)
    w[rank < 256] = 4.0
    w[rank < 32] = 12.0
    # Positives are ~1 in 1000 rows; without this the loss ignores them.
    pos = y == 1
    w[pos] = 200.0
    return w


def fit(model_name: str, Xtr: np.ndarray, ytr: np.ndarray,
        wtr: np.ndarray):
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    if model_name == "logistic":
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        )
        m.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
        return m
    if model_name == "hgb":
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m
    raise ValueError(model_name)


def recall_at_k(cache: Cache, scores: np.ndarray) -> dict:
    """Per-pair: is the true site inside the top K of this ordering?"""
    hits = {k: [] for k in K_GRID}
    min_k, pool_hit, tags = [], [], []
    for i in range(cache.n_pairs):
        a, b = int(cache.offsets[i]), int(cache.offsets[i + 1])
        d = cache.d[a:b]
        s = scores[a:b]
        truth = d <= cache.hit_px
        tags.append(cache.meta[i])
        if not truth.any():
            pool_hit.append(False)
            for k in K_GRID:
                hits[k].append(False)
            min_k.append(np.nan)
            continue
        pool_hit.append(True)
        order = np.argsort(-np.where(np.isfinite(s), s, -np.inf), kind="mergesort")
        pos = int(np.nonzero(truth[order])[0][0])   # 0-based rank of the best true row
        min_k.append(float(pos + 1))
        for k in K_GRID:
            hits[k].append(pos < k)
    return {
        "pool_recall": float(np.mean(pool_hit)),
        "recall": {f"K{k}": round(float(np.mean(hits[k])), 4) for k in K_GRID},
        "min_k": np.array(min_k, dtype=np.float64),
        "hits": {k: np.array(v) for k, v in hits.items()},
        "tags": tags,
    }


def stratify(res: dict, key: str, k: int = 256) -> dict:
    out: dict[str, dict] = {}
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(res["tags"]):
        if key == "regime":
            n = m["n"]
            g = "n<=100" if n <= 100 else ("n<=1000" if n <= 1000 else
                                           ("n<=2000" if n <= 2000 else "n>2000"))
        else:
            g = str(m[key])
        groups.setdefault(g, []).append(i)
    for g, idx in sorted(groups.items()):
        idx = np.array(idx)
        out[g] = {"n": int(idx.size),
                  f"recall_K{k}": round(float(np.mean(res["hits"][k][idx])), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=str(PROJECT / "results" / "prerank_p2_train_d0.1.npz"))
    ap.add_argument("--val", default=str(PROJECT / "results" / "prerank_p2_val_d0.1.npz"))
    ap.add_argument("--out", default=str(PROJECT / "results" / "prerank_eval.json"))
    args = ap.parse_args()

    tr, va = Cache(Path(args.train)), Cache(Path(args.val))
    print(f"train {tr.n_pairs} pairs / {tr.X.shape[0]} rows "
          f"({int(tr.y.sum())} positives)")
    print(f"val   {va.n_pairs} pairs / {va.X.shape[0]} rows")

    report: dict = {"train_pairs": tr.n_pairs, "val_pairs": va.n_pairs,
                    "delta": va.delta, "hit_px": va.hit_px, "rankers": {}}

    # -- baseline: today's ordering, best channel score ----------------------
    base = recall_at_k(va, va.X[:, va.features.index("ch_best")].astype(np.float64))
    report["pool_recall"] = round(base["pool_recall"], 4)
    report["rankers"]["channel_score"] = {
        "recall": base["recall"],
        "by_architecture": stratify(base, "architecture"),
        "by_regime": stratify(base, "regime"),
    }
    print("\nchannel_score", json.dumps(base["recall"]))

    wtr = sample_weights(tr)
    for group in ("cheap", "rcc", "hybrid"):
        keys = group_keys(tr, group)
        ctr, cva = tr.columns(keys), va.columns(keys)
        for model_name in ("logistic", "hgb"):
            name = f"{group}_{model_name}"
            model = fit(model_name, tr.X[:, ctr], tr.y, wtr)
            Xva = np.nan_to_num(va.X[:, cva], nan=0.0, posinf=0.0, neginf=0.0)
            scores = model.predict_proba(Xva)[:, 1]
            res = recall_at_k(va, scores)
            report["rankers"][name] = {
                "n_features": len(keys),
                "recall": res["recall"],
                "median_min_k": float(np.nanmedian(res["min_k"])),
                "p90_min_k": float(np.nanpercentile(res["min_k"], 90)),
                "by_architecture": stratify(res, "architecture"),
                "by_regime": stratify(res, "regime"),
            }
            print(f"{name:18s}", json.dumps(res["recall"]),
                  f" median_min_k={report['rankers'][name]['median_min_k']:.0f}")
            if group == "hybrid" and model_name == "hgb":
                best = res
                # Persisted so the end-to-end funnel test can reuse the exact
                # ranker these numbers describe rather than refitting one.
                import joblib
                joblib.dump({"model": model, "features": keys},
                            PROJECT / "results" / "prerank_hybrid_hgb.joblib")

    # -- experiment 4: how small can K be, per scene? ------------------------
    # The floor is the minimum K that retains the true site. A policy can only
    # be judged against that floor, not against a fixed K chosen in advance.
    mk = best["min_k"]
    finite = mk[np.isfinite(mk)]
    n_pool = np.array([m["n"] for m in best["tags"]], dtype=np.float64)
    report["adaptive_k"] = {
        "min_k_median": float(np.median(finite)),
        "min_k_p90": float(np.percentile(finite, 90)),
        "min_k_p99": float(np.percentile(finite, 99)),
        "min_k_max": float(finite.max()),
        "fixed_k_needed_for_99pct_of_in_pool": float(np.percentile(finite, 99)),
        "corr_min_k_vs_pool_size": round(float(np.corrcoef(
            np.log1p(n_pool[np.isfinite(mk)]), np.log1p(finite))[0, 1]), 4),
    }
    # A pool-size-proportional policy is the cheapest possible gate: it needs
    # no model, only the candidate count the harvest already produced.
    for frac, floor, cap in ((0.10, 64, 512), (0.20, 64, 768), (0.25, 128, 1024)):
        policy_k = np.clip(np.ceil(n_pool * frac), floor, cap)
        keep = np.isfinite(mk)
        rec = float(np.mean(mk[keep] <= policy_k[keep]))
        report["adaptive_k"][f"policy_{frac:g}_{floor}_{cap}"] = {
            "recall_of_in_pool": round(rec, 4),
            "k_median": float(np.median(policy_k)),
            "k_p95": float(np.percentile(policy_k, 95)),
        }

    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print("\n" + json.dumps({"pool_recall": report["pool_recall"],
                             "adaptive_k": report["adaptive_k"]}, indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
