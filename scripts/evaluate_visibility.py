#!/usr/bin/env python3
"""Reproduce the raw-response true-site visibility diagnostic."""
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

from driftforge.channels import local_maxima, response_maps
from driftforge.generator import generate_sample
from driftforge.splits import read_manifest


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
    local_hits = global_hits = 0
    ranks = []
    for index, record in enumerate(records, 1):
        sample = generate_sample(
            int(record["seed"]), record["architecture"], record["profile"], 2
        )
        maps = response_maps(sample.reference, sample.search)
        raw = maps.maps["raw"]
        xs, ys, values = local_maxima(raw)
        cx = xs.astype(float) + maps.half_w
        cy = ys.astype(float) + maps.half_h
        distances = np.hypot(cx - sample.gt_x, cy - sample.gt_y)
        near = np.flatnonzero(distances <= 5.0)
        if near.size:
            local_hits += 1
            best_near_score = float(values[near].max())
            ranks.append(1 + int(np.sum(values > best_near_score)))
        maximum = int(np.argmax(raw))
        gy, gx = np.unravel_index(maximum, raw.shape)
        global_error = math.hypot(
            gx + maps.half_w - sample.gt_x,
            gy + maps.half_h - sample.gt_y,
        )
        global_hits += global_error <= 5.0
        print(f"[{index}/{len(records)}] {record['id']}", flush=True)

    code_files = (
        "driftforge/baseline.py",
        "driftforge/channels.py",
        "driftforge/generator.py",
        "scripts/evaluate_visibility.py",
    )
    payload = {
        "schema_version": 1,
        "status": "measured_diagnostic",
        "source_revision": None,
        "benchmark": {
            "manifest": "manifests/validation_benchmark.jsonl",
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "slice": [args.start, args.start + len(records)],
            "pairs": len(records),
        },
        "metric_type": "raw_response_visibility_not_localization_accuracy",
        "true_site_local_maximum_within_5px_rate": local_hits / len(records),
        "raw_global_maximum_within_5px_rate": global_hits / len(records),
        "median_true_site_rank_among_local_maxima": float(np.median(ranks)),
        "true_site_rank_at_most_10_rate": sum(rank <= 10 for rank in ranks) / len(records),
        "true_site_rank_at_most_50_rate": sum(rank <= 50 for rank in ranks) / len(records),
        "warning": "This measures raw-response visibility and rank, not final localization accuracy.",
        "provenance": {
            "type": "regenerated_from_current_content_addressed_code",
            "reproduction_command": "python scripts/evaluate_visibility.py --output reproduced-visibility.json",
            "code_sha256": {
                name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                for name in code_files
            },
            "note": "Record the release commit SHA after committing this measured artifact."
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
