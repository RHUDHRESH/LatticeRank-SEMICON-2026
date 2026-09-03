#!/usr/bin/env python3
"""Recompute every headline rate directly from committed coordinate rows."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["id"] for row in rows}) != len(rows):
        raise AssertionError(f"{name} contains duplicate IDs")
    for row in rows:
        recomputed = math.hypot(
            float(row["pred_x"]) - float(row["gt_x"]),
            float(row["pred_y"]) - float(row["gt_y"]),
        )
        if not math.isclose(recomputed, float(row["error_px"]), abs_tol=1e-9):
            raise AssertionError(f"{row['id']}: error_px does not match coordinates")
    return rows


def _external_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["seed"], row["index"]) for row in rows}
    if len(keys) != len(rows):
        raise AssertionError(f"{name} contains duplicate seed/index rows")
    for row in rows:
        recomputed = math.hypot(
            float(row["pred_x"]) - float(row["gt_x"]),
            float(row["pred_y"]) - float(row["gt_y"]),
        )
        if not math.isclose(recomputed, float(row["error_px"]), abs_tol=1e-9):
            raise AssertionError(
                f"external {row['seed']}/{row['index']}: error does not match coordinates"
            )
    return rows


def _wilson(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    centre = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / count + z * z / (4.0 * count * count)
    ) / denominator
    return centre - radius, centre + radius


def _audit(
    label: str,
    csv_name: str,
    json_name: str,
    metric_path: tuple[str, ...],
) -> None:
    rows = _rows(csv_name)
    errors = [float(row["error_px"]) for row in rows]
    successes = sum(error <= 5.0 for error in errors)
    catastrophic = sum(error > 25.0 for error in errors)
    rate = successes / len(rows)
    payload = json.loads((RESULTS / json_name).read_text(encoding="utf-8"))
    recorded = payload
    for key in metric_path:
        recorded = recorded[key]
    if not math.isclose(rate, float(recorded), abs_tol=1e-12):
        raise AssertionError(
            f"{label}: rows give {rate:.6f}, but {json_name} records {recorded}"
        )
    low, high = _wilson(successes, len(rows))
    print(
        f"{label}: {successes}/{len(rows)} = {rate:.1%} within 5 px | "
        f"95% Wilson {low:.1%}-{high:.1%} | "
        f"{catastrophic}/{len(rows)} = {catastrophic/len(rows):.1%} over 25 px"
    )


def main() -> int:
    _audit(
        "Fixed validation",
        "validation_predictions.csv",
        "validation_metrics.json",
        ("localization_accuracy", "accuracy_at_5px"),
    )
    _audit(
        "Randomized compliance",
        "evaluation_30plus_predictions.csv",
        "evaluation_30plus.json",
        ("localization_accuracy", "accuracy_at_5px"),
    )
    external = _external_rows("external_starter_predictions.csv")
    payload = json.loads(
        (RESULTS / "external_starter_benchmark.json").read_text(encoding="utf-8")
    )
    for split, metric_key in (("development", "development"), ("confirmation", "confirmation")):
        selected = [row for row in external if row["split"] == split]
        successes = sum(float(row["error_px"]) <= 5 for row in selected)
        rate = successes / len(selected)
        recorded = float(payload[metric_key]["accuracy_at_5px"])
        if not math.isclose(rate, recorded, abs_tol=1e-12):
            raise AssertionError(f"external {split}: rows give {rate}, JSON gives {recorded}")
        print(f"External {split}: {successes}/{len(selected)} = {rate:.1%} within 5 px")
    print("PASS | headline rates are derived from the emitted final coordinates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
