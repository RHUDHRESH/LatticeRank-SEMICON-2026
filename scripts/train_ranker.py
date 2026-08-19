#!/usr/bin/env python3
"""Train the packaged candidate ranker through the inference feature path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import (
    generate_sample,
    normalize_architecture,
    normalize_profile,
)
from driftforge.model import (
    CANDIDATE_DELTA,
    EQUIVALENCE_MARGIN,
    MODEL_FEATURES,
    MODEL_PATH,
    MODEL_FORMAT_VERSION,
    POSITIVE_TOLERANCE_PX,
    model_metadata,
    package_versions,
    validate_feature_names,
)
from driftforge.pipeline import (
    compute_candidate_rows,
    rank_candidate_rows,
    select_equivalent_candidate,
)
from driftforge.splits import read_manifest

DEFAULT_TRAIN_MANIFEST = PROJECT / "manifests" / "train.jsonl"
DEFAULT_VALIDATION_MANIFEST = (
    PROJECT / "manifests" / "validation_benchmark.jsonl"
)
DEFAULT_TRAIN_LIMIT = 300
DEFAULT_VALIDATION_START = 200
DEFAULT_VALIDATION_LIMIT = 80
DEFAULT_TRAIN_MAX_CANDIDATES = 800
DEFAULT_NEGATIVES_PER_SCENE = 65


@dataclass
class SceneTable:
    sample_id: str
    scene_id: str
    seed: int
    architecture: str
    profile: str
    gt_x: float
    gt_y: float
    rows: list[dict]


def _manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_records(records: list[dict], source: Path) -> list[dict]:
    required = ("id", "scene_id", "seed", "architecture", "profile")
    normalized: list[dict] = []
    for index, source_record in enumerate(records):
        missing = [name for name in required if name not in source_record]
        if missing:
            raise ValueError(
                f"{source}: record {index} is missing {', '.join(missing)}"
            )
        record = dict(source_record)
        record["id"] = str(record["id"])
        record["scene_id"] = str(record["scene_id"])
        record["seed"] = int(record["seed"])
        if record["seed"] < 0:
            raise ValueError(f"{source}: record {record['id']} has a negative seed")
        record["architecture"] = normalize_architecture(record["architecture"])
        record["profile"] = normalize_profile(record["profile"])
        normalized.append(record)
    return normalized


def load_record_slice(
    path: Path, *, start: int, limit: int | None
) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    records = read_manifest(path)
    records = records[start:] if limit is None else records[start : start + limit]
    records = _normalize_records(records, path)
    if not records:
        raise ValueError(f"manifest slice is empty: {path}")
    return records


def assert_scene_disjoint(
    train_records: list[dict], validation_records: list[dict]
) -> None:
    """Reject scene, seed, or generated-scene collisions across the split."""
    for label, records in (
        ("training", train_records),
        ("validation", validation_records),
    ):
        scene_ids = [str(record["scene_id"]) for record in records]
        seeds = [int(record["seed"]) for record in records]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError(f"{label} manifest contains duplicate scene_id values")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{label} manifest contains duplicate seeds")

    train_scenes = {str(record["scene_id"]) for record in train_records}
    validation_scenes = {
        str(record["scene_id"]) for record in validation_records
    }
    overlap = train_scenes & validation_scenes
    if overlap:
        raise ValueError(
            "training/validation scene_id overlap: " + ", ".join(sorted(overlap)[:3])
        )

    train_seeds = {int(record["seed"]) for record in train_records}
    validation_seeds = {int(record["seed"]) for record in validation_records}
    seed_overlap = train_seeds & validation_seeds
    if seed_overlap:
        raise ValueError(
            "training/validation seed overlap: "
            + ", ".join(str(value) for value in sorted(seed_overlap)[:3])
        )

    # The latent scene generator receives seed * 17 + 3. Check that identity
    # directly so future manifest naming cannot mask a realization collision.
    train_realizations = {seed * 17 + 3 for seed in train_seeds}
    validation_realizations = {seed * 17 + 3 for seed in validation_seeds}
    if train_realizations & validation_realizations:
        raise ValueError("training/validation generated-scene realization overlap")


def build_scene_table(
    record: dict,
    *,
    training: bool,
    search_supersample: int,
    max_candidates: int,
) -> SceneTable:
    """Build one scene by calling the exact production feature function."""
    sample = generate_sample(
        int(record["seed"]),
        record["architecture"],
        record["profile"],
        search_supersample,
    )
    keep_xy = (sample.gt_x, sample.gt_y) if training else None
    rows = compute_candidate_rows(
        sample.reference,
        sample.search,
        delta=CANDIDATE_DELTA,
        max_cand=max_candidates,
        keep_xy=keep_xy,
        keep_tol=POSITIVE_TOLERANCE_PX,
        struct=True,
    )
    for row in rows:
        row["error_px"] = math.hypot(
            float(row["x"]) - sample.gt_x,
            float(row["y"]) - sample.gt_y,
        )
    return SceneTable(
        sample_id=str(record["id"]),
        scene_id=str(record["scene_id"]),
        seed=int(record["seed"]),
        architecture=sample.architecture,
        profile=sample.profile,
        gt_x=sample.gt_x,
        gt_y=sample.gt_y,
        rows=rows,
    )


def _build_scene_job(payload: tuple[dict, bool, int, int]) -> SceneTable:
    record, training, search_supersample, max_candidates = payload
    return build_scene_table(
        record,
        training=training,
        search_supersample=search_supersample,
        max_candidates=max_candidates,
    )


def collect_scene_tables(
    records: list[dict],
    *,
    training: bool,
    search_supersample: int,
    max_candidates: int,
    workers: int,
) -> list[SceneTable]:
    payloads = [
        (record, training, search_supersample, max_candidates)
        for record in records
    ]
    tables: list[SceneTable] = []
    if workers == 1:
        iterator = map(_build_scene_job, payloads)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_build_scene_job, payloads)
    try:
        for index, table in enumerate(iterator, start=1):
            tables.append(table)
            if index % 10 == 0 or index == len(payloads):
                print(
                    f"[{index}/{len(payloads)}] {sum(len(t.rows) for t in tables)} "
                    "candidate rows",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown()
    return tables


def hard_negative_sample(
    rows: list[dict], *, negative_limit: int, seed: int
) -> list[dict]:
    """Keep positives and deterministic, diverse hard negatives per scene."""
    rng = np.random.default_rng(seed)
    positives = [
        row for row in rows if row["error_px"] <= POSITIVE_TOLERANCE_PX
    ]
    negatives = [
        row for row in rows if row["error_px"] > POSITIVE_TOLERANCE_PX
    ]
    if not negatives:
        return positives

    keep: set[int] = set()

    def take(order: Sequence[int], count: int) -> None:
        for index in list(order)[:count]:
            if len(keep) >= negative_limit:
                break
            keep.add(int(index))

    for channel in ("raw", "midband", "directionality"):
        values = np.asarray(
            [
                row[channel] if math.isfinite(row[channel]) else -9.0
                for row in negatives
            ]
        )
        take(np.argsort(-values), 15)

    if positives:
        positive_raw = max(
            (
                row["raw"]
                for row in positives
                if math.isfinite(row["raw"])
            ),
            default=0.0,
        )
        confusion = np.abs(
            np.asarray(
                [
                    row["raw"] if math.isfinite(row["raw"]) else -9.0
                    for row in negatives
                ]
            )
            - positive_raw
        )
        take(np.argsort(confusion), 10)
        px, py = positives[0]["x"], positives[0]["y"]
        remote = [
            index
            for index, row in enumerate(negatives)
            if math.hypot(row["x"] - px, row["y"] - py) > 100.0
        ]
        if remote:
            values = np.asarray(
                [
                    negatives[index]["raw"]
                    if math.isfinite(negatives[index]["raw"])
                    else -9.0
                    for index in remote
                ]
            )
            take([remote[index] for index in np.argsort(-values)], 10)

    remaining = [
        index for index in range(len(negatives)) if index not in keep
    ]
    if remaining and len(keep) < negative_limit:
        chosen = rng.choice(
            remaining,
            size=min(negative_limit - len(keep), len(remaining)),
            replace=False,
        )
        keep.update(int(index) for index in chosen)
    return positives + [negatives[index] for index in sorted(keep)]


def feature_matrix(rows: list[dict]) -> np.ndarray:
    validate_feature_names(MODEL_FEATURES)
    missing = sorted(
        {
            feature
            for row in rows
            for feature in MODEL_FEATURES
            if feature not in row
        }
    )
    if missing:
        raise ValueError(
            "candidate rows are missing model features: " + ", ".join(missing)
        )
    values = np.asarray(
        [[row[feature] for feature in MODEL_FEATURES] for row in rows],
        dtype=np.float64,
    )
    return np.nan_to_num(values, nan=0.0, posinf=9.0, neginf=-9.0)


def ranker_validation_metrics(
    model, tables: list[SceneTable], *, scene_normalize: bool = False
) -> dict:
    predictions: list[dict] = []
    candidate_hits = 0
    for table in tables:
        if table.rows:
            errors = np.asarray(
                [row["error_px"] for row in table.rows], dtype=np.float64
            )
            candidate_hits += int(
                np.any(errors <= POSITIVE_TOLERANCE_PX)
            )
            _, scores = rank_candidate_rows(
                table.rows,
                {
                    "model": model,
                    "features": list(MODEL_FEATURES),
                    "metadata": {
                        "scene_feature_normalization": scene_normalize,
                    },
                },
            )
            selected, _ = select_equivalent_candidate(
                table.rows,
                scores,
                (1000, 1000),
            )
            row = table.rows[int(selected)]
            pred_x, pred_y = float(row["x"]), float(row["y"])
        else:
            pred_x = pred_y = 499.5
        error = math.hypot(pred_x - table.gt_x, pred_y - table.gt_y)
        predictions.append(
            {
                "sample_id": table.sample_id,
                "architecture": table.architecture,
                "error_px": error,
            }
        )

    errors = np.asarray(
        [prediction["error_px"] for prediction in predictions],
        dtype=np.float64,
    )
    localization = {
        "metric_type": "ranker_only_localization_accuracy",
        "selection": (
            "production_no_residual_score_normalization_then_equivalence_"
            "and_nearest_search_centre"
        ),
        "count": len(predictions),
        "accuracy_at_1px": float(np.mean(errors <= 1.0)),
        "accuracy_at_2px": float(np.mean(errors <= 2.0)),
        "accuracy_at_5px": float(np.mean(errors <= 5.0)),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "catastrophic_rate_over_25px": float(np.mean(errors > 25.0)),
    }
    for architecture in ("dram", "finfet"):
        subset = np.asarray(
            [
                prediction["error_px"]
                for prediction in predictions
                if prediction["architecture"] == architecture
            ],
            dtype=np.float64,
        )
        if subset.size:
            localization[f"accuracy_at_5px_{architecture}"] = float(
                np.mean(subset <= 5.0)
            )
    return {
        "ranker_only_localization": localization,
        "candidate_pool_diagnostic": {
            "metric_type": "candidate_recall_not_localization_accuracy",
            "tolerance_px": POSITIVE_TOLERANCE_PX,
            "hits": candidate_hits,
            "count": len(tables),
            "recall": candidate_hits / len(tables),
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the packaged ranker on scene-disjoint manifests."
    )
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=DEFAULT_VALIDATION_MANIFEST,
    )
    parser.add_argument("--train-start", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--train-limit", type=_positive_int, default=DEFAULT_TRAIN_LIMIT
    )
    parser.add_argument(
        "--validation-start",
        type=_nonnegative_int,
        default=DEFAULT_VALIDATION_START,
    )
    parser.add_argument(
        "--validation-limit",
        type=_positive_int,
        default=DEFAULT_VALIDATION_LIMIT,
    )
    parser.add_argument(
        "--train-max-candidates",
        type=_positive_int,
        default=DEFAULT_TRAIN_MAX_CANDIDATES,
    )
    parser.add_argument(
        "--validation-max-candidates",
        type=_positive_int,
        default=8_000,
    )
    parser.add_argument(
        "--negatives-per-scene",
        type=_positive_int,
        default=DEFAULT_NEGATIVES_PER_SCENE,
    )
    parser.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, min(6, os.cpu_count() or 1)),
    )
    parser.add_argument("--seed", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--scene-normalize",
        action="store_true",
        help="normalize varying feature columns within each candidate group",
    )
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="optional JSON sidecar; metrics are always embedded in the model",
    )
    return parser


def build_training_provenance(
    args: argparse.Namespace,
    train_records: list[dict],
    validation_records: list[dict],
    estimator_parameters: dict,
) -> dict:
    """Capture every material setting needed to interpret a trained bundle."""
    return {
        "seed": args.seed,
        "search_supersample": args.search_supersample,
        "train_manifest": {
            "name": args.train_manifest.name,
            "sha256": _manifest_digest(args.train_manifest),
            "start": args.train_start,
            "count": len(train_records),
        },
        "validation_manifest": {
            "name": args.validation_manifest.name,
            "sha256": _manifest_digest(args.validation_manifest),
            "start": args.validation_start,
            "count": len(validation_records),
        },
        "scene_disjoint": True,
        "scene_feature_normalization": args.scene_normalize,
        "candidate_generation": {
            "candidate_delta": CANDIDATE_DELTA,
            "training_candidate_cap": args.train_max_candidates,
            "validation_candidate_cap": args.validation_max_candidates,
            "structural_features_enabled": True,
            "training_ground_truth_keep_enabled": True,
            "validation_ground_truth_keep_enabled": False,
            "ground_truth_keep_tolerance_px": POSITIVE_TOLERANCE_PX,
        },
        "validation_selection": {
            "production_equivalent": True,
            "residual_enabled": False,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "search_shape": [1000, 1000],
        },
        "estimator_parameters": dict(estimator_parameters),
        "negative_rows_per_scene_cap": args.negatives_per_scene,
        "hard_negative_selection": {
            "deterministic": True,
            "seed_offset_per_scene": True,
            "channel_top_quota": 15,
            "score_confusion_quota": 10,
            "remote_alias_quota": 10,
        },
        "worker_count": args.workers,
    }


def run(args: argparse.Namespace) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib

    train_records = load_record_slice(
        args.train_manifest,
        start=args.train_start,
        limit=args.train_limit,
    )
    validation_records = load_record_slice(
        args.validation_manifest,
        start=args.validation_start,
        limit=args.validation_limit,
    )
    assert_scene_disjoint(train_records, validation_records)
    validate_feature_names(MODEL_FEATURES)

    started = time.perf_counter()
    print("building training candidate rows", file=sys.stderr, flush=True)
    train_tables = collect_scene_tables(
        train_records,
        training=True,
        search_supersample=args.search_supersample,
        max_candidates=args.train_max_candidates,
        workers=args.workers,
    )
    print("building validation candidate rows", file=sys.stderr, flush=True)
    validation_tables = collect_scene_tables(
        validation_records,
        training=False,
        search_supersample=args.search_supersample,
        max_candidates=args.validation_max_candidates,
        workers=args.workers,
    )

    balanced_rows: list[dict] = []
    balanced_matrices: list[np.ndarray] = []
    for index, table in enumerate(train_tables):
        sampled = hard_negative_sample(
            table.rows,
            negative_limit=args.negatives_per_scene,
            seed=args.seed + index,
        )
        balanced_rows.extend(sampled)
        if args.scene_normalize and sampled:
            full_matrix = feature_matrix(table.rows)
            mean = full_matrix.mean(axis=0)
            standard_deviation = full_matrix.std(axis=0)
            varying = standard_deviation > 1e-9
            full_matrix[:, varying] = (
                full_matrix[:, varying] - mean[varying]
            ) / standard_deviation[varying]
            indices = {id(row): position for position, row in enumerate(table.rows)}
            balanced_matrices.append(
                full_matrix[[indices[id(row)] for row in sampled]]
            )
    if not balanced_rows:
        raise RuntimeError("training produced no candidate rows")
    labels = np.asarray(
        [
            row["error_px"] <= POSITIVE_TOLERANCE_PX
            for row in balanced_rows
        ],
        dtype=np.int8,
    )
    if labels.min() == labels.max():
        raise RuntimeError("training requires both positive and negative candidates")

    estimator_parameters = {
        "max_iter": 300,
        "learning_rate": 0.08,
        "l2_regularization": 1.0,
        "random_state": args.seed,
        "class_weight": "balanced",
    }
    model = HistGradientBoostingClassifier(**estimator_parameters)
    fit_started = time.perf_counter()
    training_matrix = (
        np.vstack(balanced_matrices)
        if args.scene_normalize
        else feature_matrix(balanced_rows)
    )
    model.fit(training_matrix, labels)
    fit_seconds = time.perf_counter() - fit_started

    validation_metrics = ranker_validation_metrics(
        model,
        validation_tables,
        scene_normalize=args.scene_normalize,
    )
    metrics = {
        "training": {
            "scene_count": len(train_tables),
            "candidate_rows_before_sampling": sum(
                len(table.rows) for table in train_tables
            ),
            "candidate_rows_after_sampling": len(balanced_rows),
            "positive_rows": int(labels.sum()),
            "fit_seconds": fit_seconds,
        },
        "validation": validation_metrics,
        "total_seconds": time.perf_counter() - started,
    }
    versions = package_versions()
    provenance = build_training_provenance(
        args,
        train_records,
        validation_records,
        estimator_parameters,
    )
    metadata = {
        **model_metadata(),
        "format_version": MODEL_FORMAT_VERSION,
        "seed": args.seed,
        "package_versions": versions,
        "training_provenance": provenance,
        "metrics": metrics,
        "scene_feature_normalization": args.scene_normalize,
    }
    bundle = {
        "model": model,
        "features": list(MODEL_FEATURES),
        "seed": args.seed,
        "package_versions": versions,
        "metrics": metrics,
        "metadata": metadata,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(args.output)
    sidecar = args.metrics_output
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "training_provenance": provenance,
                    "package_versions": versions,
                    "features": list(MODEL_FEATURES),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
