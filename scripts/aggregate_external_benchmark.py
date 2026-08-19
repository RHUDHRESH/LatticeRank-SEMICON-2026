#!/usr/bin/env python3
"""Freeze compact evidence from external-starter candidate traces.

The input traces are produced by ``evaluate_external_starter.py``.  Selection
is recomputed here from the production consensus equation so exploratory
method summaries cannot accidentally become the release claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WEIGHTS = {"residual_z": 1.0, "raw_z": 0.05, "midband_z": 0.05}
EQUIVALENCE_MARGIN = 0.025


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(record: dict) -> dict:
    signals = record["candidate_signals"]
    scores = [sum(WEIGHTS[k] * row[k] for k in WEIGHTS) for row in signals]
    maximum = max(scores)
    equivalent = [i for i, value in enumerate(scores) if value >= maximum - EQUIVALENCE_MARGIN]
    chosen = min(equivalent, key=lambda i: (signals[i]["x"] - 499.5) ** 2
                 + (signals[i]["y"] - 499.5) ** 2)
    row = signals[chosen]
    return {
        "pred_x": row["x"], "pred_y": row["y"], "error_px": row["error_px"],
        "equivalence_set_size": len(equivalent), "score": scores[chosen],
    }


def summarize(rows: list[dict]) -> dict:
    errors = sorted(float(row["error_px"]) for row in rows)
    n = len(errors)
    return {
        "pairs": n,
        "successes_within_5px": sum(value <= 5 for value in errors),
        "accuracy_at_5px": sum(value <= 5 for value in errors) / n,
        "catastrophic_count_over_25px": sum(value > 25 for value in errors),
        "catastrophic_rate_over_25px": sum(value > 25 for value in errors) / n,
        "median_error_px": (errors[(n - 1) // 2] + errors[n // 2]) / 2,
        "maximum_error_px": max(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs=5, type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    revisions = set()
    source_hashes = set()
    for file_index, path in enumerate(args.inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = payload["benchmark"]
        revisions.add(benchmark["revision"])
        source_hashes.add((benchmark["pipeline_sha256"], benchmark["presets_sha256"]))
        split = "confirmation" if file_index == len(args.inputs) - 1 else "development"
        for record in payload["records"]:
            selected = select(record)
            rows.append({
                "split": split, "seed": benchmark["seed"], "index": record["index"],
                "preset": record["preset"], "architecture": record["architecture"],
                "gt_x": record["gt_x"], "gt_y": record["gt_y"],
                **selected, "candidate_count": record["candidate_count"],
                "candidate_best_error_px": record["candidate_best_error_px"],
            })
    if len(revisions) != 1 or len(source_hashes) != 1:
        raise SystemExit("input traces do not share one external source revision")
    development = [row for row in rows if row["split"] == "development"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    pipeline_hashes = {
        name: sha256(PROJECT / name) for name in (
            "driftforge/baseline.py", "driftforge/channels.py", "driftforge/lattice.py",
            "driftforge/pipeline.py", "driftforge/residual.py",
            "scripts/evaluate_external_starter.py", "scripts/aggregate_external_benchmark.py",
        )
    }
    source_pipeline, source_presets = next(iter(source_hashes))
    result = {
        "schema_version": 1,
        "status": "measured",
        "benchmark": {
            "kind": "pinned_public_reference_style_generator",
            "repository": "https://github.com/FlankerDev12/drift-sense-ref",
            "revision": next(iter(revisions)),
            "generator_pipeline_sha256": source_pipeline,
            "generator_presets_sha256": source_presets,
            "seeds": [payload for payload in sorted({row["seed"] for row in rows})],
            "presets_exercised_by_generator_default": True,
            "stress_overrides": False,
        },
        "selection": {
            "equation": "residual_z + 0.05*raw_z + 0.05*midband_z",
            "weights": WEIGHTS,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "tie_break": "nearest Search centre among evidence-equivalent candidates",
        },
        "development": summarize(development),
        "confirmation": summarize(confirmation),
        "pooled_descriptive_only": summarize(rows),
        "interpretation": (
            "Seeds 4200/4300/4400/4600 were used while freezing the consensus. "
            "Seed 4700 is the untouched confirmation run. The pooled rate is descriptive, "
            "not a substitute for the confirmation result."
        ),
        "provenance": {
            "row_evidence": "results/external_starter_predictions.csv",
            "production_code_sha256": pipeline_hashes,
            "model_sha256": "3548fa98f50135579e367021f0c577ab8d927a18a0b8ed79e6a10c734cd48ded",
            "source_revision": None,
            "reproduction": [
                "Run scripts/evaluate_external_starter.py for seeds 4200, 4300, 4400, 4600, and 4700 with count 30.",
                "Pass those five JSON traces in seed order to scripts/aggregate_external_benchmark.py.",
            ],
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"development": result["development"], "confirmation": result["confirmation"],
                      "pooled": result["pooled_descriptive_only"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
