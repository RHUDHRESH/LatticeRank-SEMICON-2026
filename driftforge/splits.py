from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import PROFILE_MIX


SPLIT_SEED_BASE = {
    "train": 100_000,
    "validation": 300_000,
    "test": 500_000,
    "stress": 700_000,
    # Disjoint from train (100k-102999), validation (300k-300599),
    # test (500k-500999) and stress (700k-700499).
    "validation_benchmark": 900_000,
}


def _weighted_profiles(split: str, count: int, rng: np.random.Generator) -> list[str]:
    names, weights = zip(*PROFILE_MIX[split])
    exact = np.asarray(weights, dtype=np.float64) * count
    counts = np.floor(exact).astype(int)
    for idx in np.argsort(exact - counts)[::-1][: count - int(counts.sum())]:
        counts[idx] += 1
    values = [name for name, n in zip(names, counts) for _ in range(int(n))]
    rng.shuffle(values)
    return values


def make_records(split: str, count: int) -> list[dict]:
    if split not in SPLIT_SEED_BASE:
        raise ValueError(f"unknown split {split!r}")
    if count < 1:
        raise ValueError("count must be positive")
    base = SPLIT_SEED_BASE[split]
    rng = np.random.default_rng(base + count * 37)
    profiles = _weighted_profiles(split, count, rng)
    # Exact or one-sample-near 50/50 architecture balance.
    architectures = np.asarray((["dram", "finfet"] * ((count + 1) // 2))[:count])
    rng.shuffle(architectures)
    return [
        {
            "id": f"{split}-{i:06d}",
            "split": split,
            "seed": base + i,
            "scene_id": f"scene-{base + i}",
            "architecture": str(architectures[i]),
            "profile": profiles[i],
        }
        for i in range(count)
    ]


def write_manifest(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: list[dict]) -> dict:
    return {
        "count": len(records),
        "architectures": dict(sorted(Counter(row["architecture"] for row in records).items())),
        "profiles": dict(sorted(Counter(row["profile"] for row in records).items())),
        "unique_scenes": len({row["scene_id"] for row in records}),
    }


def assert_disjoint(groups: dict[str, list[dict]]) -> None:
    seen: set[str] = set()
    for split, records in groups.items():
        current = {row["scene_id"] for row in records}
        overlap = seen & current
        if overlap:
            raise AssertionError(f"scene leakage into {split}: {sorted(overlap)[:3]}")
        seen |= current

