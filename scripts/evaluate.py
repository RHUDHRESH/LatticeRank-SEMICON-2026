#!/usr/bin/env python3
"""Regenerate final-pipeline predictions, metrics, and measured runtime."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
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
    MODEL_PATH,
    POSITIVE_TOLERANCE_PX,
    load_model_bundle,
    model_file_provenance,
)
from driftforge.pipeline import compute_candidate_rows, locate_v2
from driftforge.splits import read_manifest

DEFAULT_VALIDATION_MANIFEST = (
    PROJECT / "manifests" / "validation_benchmark.jsonl"
)
DEFAULT_VALIDATION_START = 200
DEFAULT_VALIDATION_LIMIT = 80
DEFAULT_RANDOMIZED_COUNT = 40
RANDOMIZED_MINIMUM = 30


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


def _normalize_records(records: list[dict]) -> list[dict]:
    required = ("id", "scene_id", "seed", "architecture", "profile")
    output: list[dict] = []
    seen: set[str] = set()
    for index, source in enumerate(records):
        missing = [field for field in required if field not in source]
        if missing:
            raise ValueError(
                f"evaluation record {index} is missing {', '.join(missing)}"
            )
        record = dict(source)
        record["id"] = str(record["id"])
        record["scene_id"] = str(record["scene_id"])
        record["seed"] = int(record["seed"])
        if record["seed"] < 0:
            raise ValueError(f"evaluation record {record['id']} has a negative seed")
        record["architecture"] = normalize_architecture(record["architecture"])
        record["profile"] = normalize_profile(record["profile"])
        if record["scene_id"] in seen:
            raise ValueError(f"duplicate evaluation scene: {record['scene_id']}")
        seen.add(record["scene_id"])
        output.append(record)
    if not output:
        raise ValueError("evaluation set is empty")
    return output


def validation_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if not args.manifest.is_file():
        raise FileNotFoundError(f"validation manifest not found: {args.manifest}")
    all_records = read_manifest(args.manifest)
    records = all_records[args.start : args.start + args.limit]
    provenance = {
        "kind": "fixed_scene_disjoint_validation",
        "manifest": args.manifest.name,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "start": args.start,
        "count": len(records),
    }
    return _normalize_records(records), provenance


def randomized_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.count < RANDOMIZED_MINIMUM:
        raise ValueError(
            f"randomized compliance evaluation requires at least "
            f"{RANDOMIZED_MINIMUM} pairs"
        )
    rng = np.random.default_rng(args.seed)
    seed_base = 1_500_000 + abs(args.seed) * 1_000
    seeds = rng.choice(
        np.arange(seed_base, seed_base + max(10_000, args.count * 20)),
        size=args.count,
        replace=False,
    )
    architectures = np.asarray(
        (["dram", "finfet"] * ((args.count + 1) // 2))[: args.count]
    )
    rng.shuffle(architectures)
    profiles = np.asarray(
        ["standard", "hard", "boundary", "ambiguous"][
            : min(4, args.count)
        ]
        + [
            ("standard", "hard", "boundary", "ambiguous")[
                int(rng.integers(0, 4))
            ]
            for _ in range(max(0, args.count - 4))
        ]
    )
    rng.shuffle(profiles)
    records = [
        {
            "id": f"randomized-{index:06d}",
            "scene_id": f"scene-{int(seed)}",
            "seed": int(seed),
            "architecture": str(architectures[index]),
            "profile": str(profiles[index]),
        }
        for index, seed in enumerate(seeds)
    ]
    provenance = {
        "kind": "deterministic_randomized_compliance",
        "random_seed": args.seed,
        "count": args.count,
        "sample_seeds": [int(seed) for seed in seeds],
    }
    return _normalize_records(records), provenance


def evaluate_record(
    record: dict,
    *,
    model_bundle: dict,
    model_path: Path,
    search_supersample: int,
    use_residual: bool,
) -> dict:
    sample = generate_sample(
        int(record["seed"]),
        record["architecture"],
        record["profile"],
        search_supersample,
    )
    started = time.perf_counter()
    candidate_rows = compute_candidate_rows(
        sample.reference,
        sample.search,
        delta=CANDIDATE_DELTA,
    )
    result = locate_v2(
        sample.reference,
        sample.search,
        use_residual=use_residual,
        model_path=model_path,
        model_bundle=model_bundle,
        candidate_rows=candidate_rows,
    )
    runtime_seconds = time.perf_counter() - started
    error = math.hypot(result.x - sample.gt_x, result.y - sample.gt_y)
    candidate_error = min(
        (
            math.hypot(
                float(row["x"]) - sample.gt_x,
                float(row["y"]) - sample.gt_y,
            )
            for row in candidate_rows
        ),
        default=float("inf"),
    )
    return {
        "id": str(record["id"]),
        "scene_id": str(record["scene_id"]),
        "seed": int(record["seed"]),
        "architecture": sample.architecture,
        "profile": sample.profile,
        "gt_x": sample.gt_x,
        "gt_y": sample.gt_y,
        "pred_x": result.x,
        "pred_y": result.y,
        "error_px": error,
        "localized_within_1px": error <= 1.0,
        "localized_within_2px": error <= 2.0,
        "localized_within_5px": error <= 5.0,
        "candidate_pool_hit_within_5px": (
            candidate_error <= POSITIVE_TOLERANCE_PX
        ),
        "candidate_pool_best_error_px": candidate_error,
        "n_candidates": result.n_candidates,
        "probability": result.probability,
        "equivalence_set_size": result.eq_set_size,
        "used_residual": result.used_residual,
        "runtime_seconds": runtime_seconds,
    }


def _localization_metrics(rows: list[dict]) -> dict:
    errors = np.asarray([row["error_px"] for row in rows], dtype=np.float64)
    output = {
        "metric_type": "final_pipeline_localization_accuracy",
        "count": len(rows),
        "accuracy_at_1px": float(np.mean(errors <= 1.0)),
        "accuracy_at_2px": float(np.mean(errors <= 2.0)),
        "accuracy_at_5px": float(np.mean(errors <= 5.0)),
        "accuracy_at_10px": float(np.mean(errors <= 10.0)),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "p99_error_px": float(np.percentile(errors, 99)),
        "max_error_px": float(errors.max()),
        "catastrophic_count_over_25px": int(np.sum(errors > 25.0)),
        "catastrophic_rate_over_25px": float(np.mean(errors > 25.0)),
    }
    for architecture in ("dram", "finfet"):
        subset = np.asarray(
            [
                row["error_px"]
                for row in rows
                if row["architecture"] == architecture
            ],
            dtype=np.float64,
        )
        if subset.size:
            output[f"count_{architecture}"] = int(subset.size)
            output[f"accuracy_at_5px_{architecture}"] = float(
                np.mean(subset <= 5.0)
            )
    return output


def summarize(
    rows: list[dict],
    *,
    provenance: dict,
    model_path: Path,
    model_bundle: dict,
    model_load_seconds: float,
    evaluation_wall_seconds: float,
    use_residual: bool,
    search_supersample: int,
) -> dict:
    runtimes = np.asarray(
        [row["runtime_seconds"] for row in rows], dtype=np.float64
    )
    candidate_hits = sum(
        bool(row["candidate_pool_hit_within_5px"]) for row in rows
    )
    finite_best = np.asarray(
        [
            row["candidate_pool_best_error_px"]
            for row in rows
            if math.isfinite(row["candidate_pool_best_error_px"])
        ],
        dtype=np.float64,
    )
    model_provenance = model_file_provenance(model_path)
    return {
        "schema_version": 1,
        "dataset": provenance,
        "pipeline": {
            "name": "packaged_final_localizer",
            "candidate_delta": CANDIDATE_DELTA,
            "residual_enabled": use_residual,
            "search_supersample": search_supersample,
            "model": {
                **model_provenance,
                "format_version": model_bundle["metadata"].get(
                    "format_version"
                ),
                "feature_count": len(model_bundle["features"]),
            },
        },
        "localization_accuracy": _localization_metrics(rows),
        "candidate_pool_diagnostic": {
            "metric_type": "candidate_recall_not_localization_accuracy",
            "description": (
                "Fraction whose harvested candidate pool contains a point "
                "within 5 px of ground truth; this is a proposal-stage ceiling, "
                "not final localization accuracy."
            ),
            "tolerance_px": POSITIVE_TOLERANCE_PX,
            "hits": candidate_hits,
            "count": len(rows),
            "recall": candidate_hits / len(rows),
            "median_best_candidate_error_px": (
                float(np.median(finite_best))
                if finite_best.size
                else None
            ),
        },
        "runtime": {
            "metric_type": "measured_final_pipeline_runtime",
            "definition": (
                "compute_candidate_rows + model prediction + residual evidence "
                "+ final selection; synthetic generation and output writing excluded"
            ),
            "model_load_seconds": model_load_seconds,
            "mean_seconds_per_pair": float(runtimes.mean()),
            "median_seconds_per_pair": float(np.median(runtimes)),
            "p95_seconds_per_pair": float(np.percentile(runtimes, 95)),
            "min_seconds_per_pair": float(runtimes.min()),
            "max_seconds_per_pair": float(runtimes.max()),
            "sum_pair_seconds": float(runtimes.sum()),
            "evaluation_wall_seconds_including_generation": evaluation_wall_seconds,
        },
        "composition": {
            "architectures": dict(
                sorted(Counter(row["architecture"] for row in rows).items())
            ),
            "profiles": dict(
                sorted(Counter(row["profile"] for row in rows).items())
            ),
        },
    }


def _write_outputs(output_dir: Path, rows: list[dict], metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    temporary_csv = predictions_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(predictions_path)

    metrics_path = output_dir / "metrics.json"
    temporary_json = metrics_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(metrics_path)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--search-supersample", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="debug-only ranker evaluation without final residual evidence",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate final localization predictions and metrics."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser(
        "validation", help="reproduce the fixed 80-pair final validation"
    )
    validation.add_argument(
        "--manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST
    )
    validation.add_argument(
        "--start", type=_nonnegative_int, default=DEFAULT_VALIDATION_START
    )
    validation.add_argument(
        "--limit", type=_positive_int, default=DEFAULT_VALIDATION_LIMIT
    )
    _add_common_arguments(validation)

    randomized = commands.add_parser(
        "randomized", help="run a deterministic randomized 30+ pair evaluation"
    )
    randomized.add_argument(
        "--count", type=_positive_int, default=DEFAULT_RANDOMIZED_COUNT
    )
    randomized.add_argument("--seed", type=_nonnegative_int, default=2026)
    _add_common_arguments(randomized)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.command == "validation":
        records, provenance = validation_records(args)
    else:
        records, provenance = randomized_records(args)

    evaluation_started = time.perf_counter()
    model_started = time.perf_counter()
    bundle = load_model_bundle(args.model)
    model_load_seconds = time.perf_counter() - model_started

    rows: list[dict] = []
    for index, record in enumerate(records, start=1):
        row = evaluate_record(
            record,
            model_bundle=bundle,
            model_path=args.model,
            search_supersample=args.search_supersample,
            use_residual=not args.no_residual,
        )
        rows.append(row)
        print(
            f"[{index}/{len(records)}] {row['id']} "
            f"error={row['error_px']:.2f}px runtime={row['runtime_seconds']:.2f}s",
            file=sys.stderr,
            flush=True,
        )
    wall_seconds = time.perf_counter() - evaluation_started
    metrics = summarize(
        rows,
        provenance=provenance,
        model_path=args.model,
        model_bundle=bundle,
        model_load_seconds=model_load_seconds,
        evaluation_wall_seconds=wall_seconds,
        use_residual=not args.no_residual,
        search_supersample=args.search_supersample,
    )
    _write_outputs(args.output_dir, rows, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return metrics


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
