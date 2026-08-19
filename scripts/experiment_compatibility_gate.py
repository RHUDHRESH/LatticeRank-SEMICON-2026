#!/usr/bin/env python3
"""Measure image-only lattice compatibility on a fixed DriftForge manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.generator import generate_sample
from driftforge.pipeline import lattice_compatibility_diagnostic
from driftforge.splits import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = read_manifest(args.manifest)[args.start : args.start + args.limit]
    rows = []
    for index, record in enumerate(selected, 1):
        sample = generate_sample(
            int(record["seed"]), record["architecture"], record["profile"], 2
        )
        rows.append(
            {
                "id": record["id"],
                "architecture": record["architecture"],
                "profile": record["profile"],
                **lattice_compatibility_diagnostic(sample.reference, sample.search),
            }
        )
        print(f"[{index}/{len(selected)}] {record['id']}", flush=True)
    payload = {
        "manifest": str(args.manifest),
        "start": args.start,
        "count": len(rows),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
