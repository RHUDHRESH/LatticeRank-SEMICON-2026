#!/usr/bin/env python3
"""Reproduce the candidate-margin sweep without running the final ranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, response_maps
from driftforge.generator import generate_sample
from driftforge.model import MAX_CANDIDATES
from driftforge.splits import read_manifest

DELTAS = (0.02, 0.05, 0.10, 0.15, 0.25)


def _wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [centre - radius, centre + radius]


def _pool(channel_maps, delta: float) -> list[dict]:
    candidates = harvest(channel_maps, delta=delta)
    candidates.sort(
        key=lambda candidate: -max(
            value
            for key, value in candidate.items()
            if key in CHANNELS and isinstance(value, float) and math.isfinite(value)
        )
    )
    return candidates[:MAX_CANDIDATES]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "manifests" / "validation_benchmark.jsonl",
    )
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_manifest(args.manifest)[args.start : args.start + args.limit]
    measurements = {delta: [] for delta in DELTAS}
    for index, record in enumerate(records, 1):
        sample = generate_sample(
            int(record["seed"]), record["architecture"], record["profile"], 2
        )
        maps = response_maps(sample.reference, sample.search)
        for delta in DELTAS:
            candidates = _pool(maps, delta)
            best = min(
                (
                    math.hypot(
                        candidate["x"] - sample.gt_x,
                        candidate["y"] - sample.gt_y,
                    )
                    for candidate in candidates
                ),
                default=float("inf"),
            )
            measurements[delta].append(
                {
                    "hit": best <= 5.0,
                    "count": len(candidates),
                    "architecture": record["architecture"],
                }
            )
        print(f"[{index}/{len(records)}] {record['id']}", flush=True)

    sweep = []
    for delta in DELTAS:
        rows = measurements[delta]
        hits = sum(row["hit"] for row in rows)
        sweep.append(
            {
                "delta": delta,
                "overall_recall_le_5px": hits / len(rows),
                "hits": hits,
                "median_candidates": float(np.median([row["count"] for row in rows])),
                "p95_candidates": float(np.percentile([row["count"] for row in rows], 95)),
                "wilson_95_interval": _wilson(hits, len(rows)),
                "dram_recall_le_5px": float(
                    np.mean([row["hit"] for row in rows if row["architecture"] == "dram"])
                ),
                "finfet_recall_le_5px": float(
                    np.mean([row["hit"] for row in rows if row["architecture"] == "finfet"])
                ),
            }
        )
    code_files = (
        "driftforge/channels.py",
        "driftforge/generator.py",
        "driftforge/model.py",
        "scripts/evaluate_candidate_recall.py",
    )
    hashes = {
        name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        for name in code_files
    }
    try:
        manifest_name = args.manifest.resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        manifest_name = args.manifest.name
    payload = {
        "schema_version": 1,
        "status": "measured",
        "source_revision": None,
        "benchmark": {
            "manifest": manifest_name,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "slice": [args.start, args.start + len(records)],
            "pairs": len(records),
        },
        "metric": "Fraction of pairs with at least one harvested candidate within 5 search pixels of ground truth.",
        "warning": "Candidate-pool recall is a proposal-stage ceiling, not end-to-end localization accuracy.",
        "sweep": sweep,
        "shipped_pipeline": next(row for row in sweep if row["delta"] == 0.10),
        "diagnostic_wider_pool": next(row for row in sweep if row["delta"] == 0.15),
        "provenance": {
            "type": "regenerated_from_current_content_addressed_code",
            "reproduction_command": "python scripts/evaluate_candidate_recall.py --output results/reproduced-candidate-recall.json",
            "code_sha256": hashes,
            "note": "Record the release commit SHA after committing this measured artifact.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(sweep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
