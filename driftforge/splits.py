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

# Phase 2 bases. Disjoint from every Phase 1 base and from each other; the
# gap between bases exceeds the largest per-split count by two orders of
# magnitude, and assert_disjoint additionally checks derived realizations.
SPLIT_SEED_BASE.update(
    {
        "p2_train": 1_100_000,
        "p2_val": 1_300_000,
        "p2_test": 1_500_000,
        "p2_holdout_fam": 1_700_000,
        "p2_stress": 1_900_000,
        # RGB optical-mode (Set D analogue) validation split, mirroring the
        # p2_val mix on a disjoint seed base.
        "p2_val_rgb": 2_100_000,
        # Bulk production corpus (10M+ base; far from every other range).
        "p2_bulk": 10_000_000,
    }
)


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
    """Reject seed collisions across splits, including derived realizations.

    Besides the raw seeds themselves, every split consumes seeds derived from
    them - the latent scene is drawn from ``seed * 17 + 3`` (Phase 1) or the
    scene seed directly (Phase 2), absent-pair references from
    ``scene_seed ^ 0x5EED``, and acquisitions from the Phase 2 multipliers -
    so a collision in any of those derived spaces would correlate two
    splits' imagery just as effectively. The check mirrors
    ``train_ranker.assert_scene_disjoint``.
    """
    seen: dict[str, set] = {"scene_id": set(), "seed": set(), "derived": set()}
    for split, records in groups.items():
        current_ids = {row["scene_id"] for row in records}
        overlap = seen["scene_id"] & current_ids
        if overlap:
            raise AssertionError(f"scene leakage into {split}: {sorted(overlap)[:3]}")

        seeds = {int(row["seed"]) for row in records}
        derived = set()
        for seed in seeds:
            for value in _derived_realizations(seed):
                derived.add(value)
        # Raw seeds and derived realizations are checked against the union of
        # both spaces from earlier splits: a seed equal to another split's
        # derived realization correlates the two scenes just as effectively.
        for label, current in (("seed", seeds), ("derived", derived)):
            overlap = (seen["seed"] | seen["derived"]) & current
            if overlap:
                raise AssertionError(
                    f"{label} leakage into {split}: {sorted(overlap)[:3]}"
                )
        seen["scene_id"] |= current_ids
        seen["seed"] |= seeds
        seen["derived"] |= derived


def _derived_realizations(seed: int) -> set[int]:
    """Seed values derived from ``seed`` by the generators (G9)."""
    return {
        seed * 17 + 3,            # Phase 1 latent scene
        seed * 7_919 + 101,       # Phase 2 reference acquisition stream
        seed * 1_000_003 + 313,   # Phase 2 search acquisition stream
        seed ^ 0x5EED,            # Phase 2 absent-pair reference scene
    }

