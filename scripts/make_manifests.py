#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.splits import assert_disjoint, make_records, summarize, write_manifest


DEFAULT_COUNTS = {"train": 3000, "validation": 600, "test": 1000, "stress": 500}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic, scene-disjoint LatticeRank manifests.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "manifests")
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--validation", type=int, default=DEFAULT_COUNTS["validation"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--stress", type=int, default=DEFAULT_COUNTS["stress"])
    args = parser.parse_args()

    counts = {"train": args.train, "validation": args.validation, "test": args.test, "stress": args.stress}
    groups = {split: make_records(split, count) for split, count in counts.items()}
    assert_disjoint(groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, records in groups.items():
        write_manifest(args.output_dir / f"{split}.jsonl", records)
        report[split] = summarize(records)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

