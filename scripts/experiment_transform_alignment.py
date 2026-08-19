#!/usr/bin/env python3
"""Measure candidate quality after physically motivated transform alignment."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.channels import CHANNELS, harvest, response_maps
from driftforge.generator import generate_sample
from driftforge.pipeline import lattice_compatibility_diagnostic, normalize_evidence
from driftforge.residual import ResidualMatcher
from driftforge.splits import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    records = read_manifest(args.manifest)[args.start : args.start + args.limit]
    totals = {
        name: {
            "recall": 0,
            **{channel: 0 for channel in CHANNELS},
            "residual": 0,
            "consensus": 0,
        }
        for name in ("identity", "rotation_inverse", "estimated")
    }
    for index, record in enumerate(records, 1):
        sample = generate_sample(
            int(record["seed"]), record["architecture"], record["profile"], 2
        )
        rotation = (
            sample.search_acquisition.rotation_deg
            - sample.reference_acquisition.rotation_deg
        )
        scale = sample.search_acquisition.scale / sample.reference_acquisition.scale
        variants = {
            "identity": (1.0, 0.0),
            "rotation_inverse": (scale, -rotation),
        }
        diagnostic = lattice_compatibility_diagnostic(sample.reference, sample.search)
        variants["estimated"] = (
            diagnostic["suggested_scale"],
            diagnostic["suggested_rotation_deg"],
        )
        line = []
        for name, (trial_scale, trial_rotation) in variants.items():
            candidates = harvest(
                response_maps(
                    sample.reference,
                    sample.search,
                    scale=trial_scale,
                    rotation=trial_rotation,
                ),
                delta=0.10,
            )
            errors = np.asarray(
                [
                    math.hypot(row["x"] - sample.gt_x, row["y"] - sample.gt_y)
                    for row in candidates
                ]
            )
            hit = bool(errors.size and errors.min() <= 5.0)
            totals[name]["recall"] += hit
            for channel in CHANNELS:
                best = max(
                    range(len(candidates)),
                    key=lambda i: (
                        candidates[i][channel]
                        if np.isfinite(candidates[i][channel])
                        else -9.0
                    ),
                )
                totals[name][channel] += errors[best] <= 5.0
            matcher = ResidualMatcher(
                sample.reference,
                sample.search,
                scale=trial_scale,
                rotation=trial_rotation,
            )
            residual = normalize_evidence(
                np.asarray(
                    [
                        matcher.score(row["x"], row["y"])["res_int_m50"]
                        for row in candidates
                    ]
                )
            )
            raw = normalize_evidence(np.asarray([row["raw"] for row in candidates]))
            mid = normalize_evidence(np.asarray([row["midband"] for row in candidates]))
            if residual is not None:
                totals[name]["residual"] += errors[int(np.argmax(residual))] <= 5.0
            if residual is not None and raw is not None and mid is not None:
                score = residual + 0.05 * raw + 0.05 * mid
                totals[name]["consensus"] += errors[int(np.argmax(score))] <= 5.0
            line.append(f"{name}={'hit' if hit else 'miss'}")
        print(f"[{index}/{len(records)}] " + " ".join(line), flush=True)
    print("\nsummary")
    for name, metrics in totals.items():
        rendered = " ".join(f"{key}={value}/{len(records)}" for key, value in metrics.items())
        print(f"{name}: {rendered}")


if __name__ == "__main__":
    main()
