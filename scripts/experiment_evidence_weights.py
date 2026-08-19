#!/usr/bin/env python3
"""Cache scene-level evidence and test score fusion without rerunning FFTs.

This is an ablation harness, not the production evaluator.  It deliberately
keeps scene boundaries and reports a first-half tuning / second-half holdout
split so score-weight experiments cannot silently become candidate-row leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample
from driftforge.model import MODEL_PATH, load_model_bundle
from driftforge.pipeline import (
    _add_structural_features,
    compute_candidate_rows,
    normalize_evidence,
    rank_candidate_rows,
    wallpaper_ambiguity_diagnostic,
)
from driftforge.residual import ResidualMatcher
from driftforge.splits import read_manifest

SIGNALS = (
    "model",
    "raw",
    "midband",
    "directionality",
    *ResidualMatcher.KEYS,
)


def _z(values: np.ndarray) -> np.ndarray:
    normalized = normalize_evidence(values)
    return values.astype(float) if normalized is None else normalized


def collect(args: argparse.Namespace) -> None:
    records = read_manifest(args.manifest)[args.start : args.start + args.limit]
    bundle = load_model_bundle(args.model)
    scenes = []
    for index, record in enumerate(records, 1):
        sample = generate_sample(
            int(record["seed"]), record["architecture"], record["profile"], 2
        )
        started = time.perf_counter()
        rows = compute_candidate_rows(sample.reference, sample.search, struct=False)
        ambiguity = wallpaper_ambiguity_diagnostic(rows, sample.search)
        _add_structural_features(sample.reference, sample.search, rows)
        probabilities, _ = rank_candidate_rows(rows, bundle)
        matcher = ResidualMatcher(sample.reference, sample.search)
        residual_rows = []
        for row in rows:
            scores = matcher.score(row["x"], row["y"])
            residual_rows.append([scores[key] for key in matcher.KEYS])
        residual = np.asarray(residual_rows, dtype=np.float32)
        signal = np.column_stack(
            [
                _z(probabilities),
                _z(np.asarray([row["raw"] for row in rows], dtype=float)),
                _z(np.asarray([row["midband"] for row in rows], dtype=float)),
                _z(np.asarray([row["directionality"] for row in rows], dtype=float)),
                *[_z(residual[:, column]) for column in range(residual.shape[1])],
            ]
        ).astype(np.float32)
        errors = np.hypot(
            np.asarray([row["x"] for row in rows]) - sample.gt_x,
            np.asarray([row["y"] for row in rows]) - sample.gt_y,
        ).astype(np.float32)
        centre_error = math.hypot(sample.gt_x - 499.5, sample.gt_y - 499.5)
        scenes.append(
            {
                "id": record["id"],
                "profile": record["profile"],
                "architecture": record["architecture"],
                "signals": signal,
                "errors": errors,
                "centre_error": centre_error,
                "ambiguity": ambiguity,
            }
        )
        print(
            f"[{index}/{len(records)}] {record['id']} rows={len(rows)} "
            f"best={float(errors.min()):.2f}px {time.perf_counter()-started:.2f}s",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"signals": SIGNALS, "scenes": scenes}, args.output, compress=3)


def _evaluate(scenes: list[dict], weights: np.ndarray) -> tuple[int, int, float]:
    success = catastrophic = 0
    errors = []
    for scene in scenes:
        # Exact-wallpaper labels are used only to isolate fusion quality here;
        # production detection remains image-derived and is evaluated elsewhere.
        if scene["profile"] == "ambiguous":
            error = float(scene["centre_error"])
        else:
            score = scene["signals"] @ weights
            error = float(scene["errors"][int(np.argmax(score))])
        errors.append(error)
        success += error <= 5.0
        catastrophic += error > 25.0
    return success, catastrophic, float(np.median(errors))


def _meta_training_rows(scenes: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scene-balanced positives plus difficult and diverse wrong aliases."""
    rng = np.random.default_rng(seed)
    matrices, labels = [], []
    for scene in scenes:
        if scene["profile"] == "ambiguous":
            continue
        signal = scene["signals"]
        positive = np.flatnonzero(scene["errors"] <= 5.0)
        negative = np.flatnonzero(scene["errors"] > 5.0)
        keep = set(map(int, positive))
        for column in range(signal.shape[1]):
            keep.update(map(int, negative[np.argsort(-signal[negative, column])[:20]]))
        remaining = np.asarray([index for index in negative if int(index) not in keep])
        if remaining.size:
            keep.update(
                map(
                    int,
                    rng.choice(remaining, min(100, remaining.size), replace=False),
                )
            )
        chosen = np.asarray(sorted(keep), dtype=int)
        matrices.append(signal[chosen])
        labels.append((scene["errors"][chosen] <= 5.0).astype(np.int8))
    return np.vstack(matrices), np.concatenate(labels)


def _evaluate_model(scenes: list[dict], model) -> tuple[int, int, float]:
    success = catastrophic = 0
    errors = []
    for scene in scenes:
        if scene["profile"] == "ambiguous":
            error = float(scene["centre_error"])
        else:
            score = model.predict_proba(scene["signals"])[:, 1]
            error = float(scene["errors"][int(np.argmax(score))])
        errors.append(error)
        success += error <= 5.0
        catastrophic += error > 25.0
    return success, catastrophic, float(np.median(errors))


def search(args: argparse.Namespace) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    payload = joblib.load(args.cache)
    scenes = payload["scenes"]
    split = len(scenes) // 2
    tune, holdout = scenes[:split], scenes[split:]
    candidates = []
    # Named, interpretable ablations first.
    for column, name in enumerate(SIGNALS):
        weights = np.zeros(len(SIGNALS)); weights[column] = 1.0
        candidates.append((name, weights))
    current = np.zeros(len(SIGNALS)); current[0] = current[SIGNALS.index("res_int_m50")] = 1.0
    candidates.append(("current_model_plus_res_int_m50", current))
    rng = np.random.default_rng(args.seed)
    for index in range(args.trials):
        weights = rng.normal(0.0, 1.0, len(SIGNALS))
        weights /= max(float(np.linalg.norm(weights)), 1e-9)
        candidates.append((f"random_{index}", weights))
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_evaluate(tune, item[1])[0],
            _evaluate(tune, item[1])[1],
            _evaluate(tune, item[1])[2],
        ),
    )
    report = []
    for name, weights in ranked[: args.top]:
        report.append(
            {
                "name": name,
                "weights": dict(zip(SIGNALS, map(float, weights))),
                "tune": _evaluate(tune, weights),
                "holdout": _evaluate(holdout, weights),
                "all": _evaluate(scenes, weights),
            }
        )
    matrix, labels = _meta_training_rows(tune, args.seed)
    models = {
        "logistic_meta": LogisticRegression(
            class_weight="balanced", max_iter=2_000, C=0.2, random_state=args.seed
        ),
        "hist_gradient_meta": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=250, learning_rate=0.06,
            l2_regularization=2.0, random_state=args.seed,
        ),
        "random_forest_meta": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            max_features=0.8, n_jobs=-1, random_state=args.seed,
        ),
    }
    model_report = []
    for name, model in models.items():
        model.fit(matrix, labels)
        model_report.append(
            {
                "name": name,
                "training_rows": int(labels.size),
                "positive_rows": int(labels.sum()),
                "tune": _evaluate_model(tune, model),
                "holdout": _evaluate_model(holdout, model),
                "all": _evaluate_model(scenes, model),
            }
        )
    print(json.dumps({"linear_fusions": report, "meta_models": model_report}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("collect")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--start", type=int, default=200)
    cache.add_argument("--limit", type=int, default=80)
    cache.add_argument("--model", type=Path, default=MODEL_PATH)
    cache.add_argument("--output", type=Path, required=True)
    grid = commands.add_parser("search")
    grid.add_argument("--cache", type=Path, required=True)
    grid.add_argument("--trials", type=int, default=20_000)
    grid.add_argument("--top", type=int, default=10)
    grid.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    collect(args) if args.command == "collect" else search(args)


if __name__ == "__main__":
    main()
