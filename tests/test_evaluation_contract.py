from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate


def test_randomized_evaluation_is_deterministic_and_requires_30_pairs() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        evaluate.randomized_records(SimpleNamespace(count=29, seed=7))

    args = SimpleNamespace(count=30, seed=7)
    first, first_provenance = evaluate.randomized_records(args)
    second, second_provenance = evaluate.randomized_records(args)
    assert first == second
    assert first_provenance == second_provenance
    assert {record["architecture"] for record in first} == {"dram", "finfet"}
    assert {"standard", "hard", "boundary", "ambiguous"} <= {
        record["profile"] for record in first
    }


def test_candidate_recall_is_not_reported_as_localization_accuracy(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"portable model provenance")
    rows = [
        {
            "architecture": "dram",
            "profile": "standard",
            "error_px": 100.0,
            "candidate_pool_hit_within_5px": True,
            "candidate_pool_best_error_px": 1.0,
            "runtime_seconds": 2.0,
        },
        {
            "architecture": "finfet",
            "profile": "hard",
            "error_px": 2.0,
            "candidate_pool_hit_within_5px": True,
            "candidate_pool_best_error_px": 2.0,
            "runtime_seconds": 4.0,
        },
    ]
    metrics = evaluate.summarize(
        rows,
        provenance={"kind": "test"},
        model_path=model_path,
        model_bundle={
            "features": ["a"],
            "metadata": {"format_version": 1},
        },
        model_load_seconds=0.1,
        evaluation_wall_seconds=6.5,
        use_residual=True,
        search_supersample=2,
    )
    assert metrics["localization_accuracy"]["accuracy_at_5px"] == 0.5
    diagnostic = metrics["candidate_pool_diagnostic"]
    assert diagnostic["recall"] == 1.0
    assert diagnostic["metric_type"] == (
        "candidate_recall_not_localization_accuracy"
    )
    assert metrics["pipeline"]["model"]["path"] == "model.joblib"
    assert len(metrics["pipeline"]["model"]["sha256"]) == 64
    code = metrics["evaluation_code"]
    assert code["algorithm"] == "sha256"
    assert len(code["aggregate_sha256"]) == 64
    assert "driftforge/pipeline.py" in code["files"]


def test_evaluation_manifest_rejects_invalid_profile() -> None:
    with pytest.raises(ValueError, match="profile must be one of"):
        evaluate._normalize_records(
            [
                {
                    "id": "bad",
                    "scene_id": "scene-bad",
                    "seed": 1,
                    "architecture": "finfet",
                    "profile": "bad",
                }
            ]
        )
