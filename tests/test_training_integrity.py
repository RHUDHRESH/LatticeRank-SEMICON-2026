from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from driftforge.model import MODEL_FEATURES, forbidden_features
from driftforge.pipeline import (
    locate_v2,
    rank_candidate_rows,
    select_equivalent_candidate,
)
from scripts import train_ranker


def test_training_and_inference_share_compute_candidate_rows(
    monkeypatch,
) -> None:
    sample = SimpleNamespace(
        reference=np.zeros((4, 4), dtype=np.uint8),
        search=np.zeros((4, 4), dtype=np.uint8),
        gt_x=10.0,
        gt_y=20.0,
        architecture="dram",
        profile="standard",
    )
    monkeypatch.setattr(train_ranker, "generate_sample", lambda *_args: sample)
    calls: list[dict] = []

    def fake_compute(reference, search, **kwargs):
        assert reference is sample.reference
        assert search is sample.search
        calls.append(kwargs)
        return [{"x": 13.0, "y": 24.0}]

    monkeypatch.setattr(train_ranker, "compute_candidate_rows", fake_compute)
    record = {
        "id": "sample",
        "scene_id": "scene-1",
        "seed": 1,
        "architecture": "dram",
        "profile": "standard",
    }
    training = train_ranker.build_scene_table(
        record,
        training=True,
        search_supersample=1,
        max_candidates=100,
    )
    validation = train_ranker.build_scene_table(
        record,
        training=False,
        search_supersample=1,
        max_candidates=100,
    )
    assert calls[0]["keep_xy"] == (10.0, 20.0)
    assert calls[1]["keep_xy"] is None
    assert calls[0]["struct"] is True
    assert training.rows[0]["error_px"] == 5.0
    assert validation.rows[0]["error_px"] == 5.0


def test_scene_disjoint_guard_rejects_leakage() -> None:
    train = [{"scene_id": "scene-a", "seed": 10}]
    with pytest.raises(ValueError, match="scene_id overlap"):
        train_ranker.assert_scene_disjoint(
            train, [{"scene_id": "scene-a", "seed": 20}]
        )
    with pytest.raises(ValueError, match="seed overlap"):
        train_ranker.assert_scene_disjoint(
            train, [{"scene_id": "scene-b", "seed": 10}]
        )


def test_model_features_exclude_labels_positions_and_centre_distance() -> None:
    prohibited_exact = {
        "x",
        "y",
        "gt_x",
        "gt_y",
        "error_px",
        "sample_id",
        "scene_id",
        "architecture",
        "profile",
    }
    assert not (set(MODEL_FEATURES) & prohibited_exact)
    assert not forbidden_features(MODEL_FEATURES)


def test_hard_negative_sampling_is_seed_deterministic() -> None:
    rows = []
    for index in range(120):
        rows.append(
            {
                "row_id": index,
                "x": float(index * 4),
                "y": float(index * 3),
                "error_px": 0.0 if index == 0 else float(index + 10),
                "raw": index / 120.0,
                "midband": (120 - index) / 120.0,
                "directionality": ((index * 17) % 120) / 120.0,
            }
        )
    first = train_ranker.hard_negative_sample(
        rows, negative_limit=65, seed=123
    )
    second = train_ranker.hard_negative_sample(
        rows, negative_limit=65, seed=123
    )
    assert [row["row_id"] for row in first] == [
        row["row_id"] for row in second
    ]
    assert first[0]["row_id"] == 0
    capped = train_ranker.hard_negative_sample(
        rows, negative_limit=7, seed=123
    )
    assert sum(row["error_px"] > 5.0 for row in capped) <= 7
    assert [row["row_id"] for row in capped] == [
        row["row_id"]
        for row in train_ranker.hard_negative_sample(
            rows, negative_limit=7, seed=123
        )
    ]


def test_feature_matrix_uses_only_fixed_ordered_schema() -> None:
    row = {feature: float(index) for index, feature in enumerate(MODEL_FEATURES)}
    row.update({"gt_x": 999999.0, "error_px": 999999.0})
    matrix = train_ranker.feature_matrix([row])
    assert matrix.shape == (1, len(MODEL_FEATURES))
    assert matrix[0, 0] == 0.0
    assert matrix[0, -1] == float(len(MODEL_FEATURES) - 1)


def test_training_validation_uses_production_no_residual_selection() -> None:
    class FixedModel:
        def predict_proba(self, matrix):
            probabilities = np.asarray([0.10, 0.80, 0.75])
            return np.column_stack((1.0 - probabilities, probabilities))

    rows = [
        {"x": 100.0, "y": 100.0, "raw": 0.0},
        {"x": 700.0, "y": 700.0, "raw": 1.0},
        {"x": 500.0, "y": 500.0, "raw": 2.0},
    ]
    bundle = {"model": FixedModel(), "features": ["raw"]}
    _, scores = rank_candidate_rows(rows, bundle)
    selected, equivalence_size = select_equivalent_candidate(
        rows, scores, (1000, 1000)
    )
    result = locate_v2(
        np.zeros((1000, 1000), dtype=np.uint8),
        np.zeros((1000, 1000), dtype=np.uint8),
        use_residual=False,
        model_bundle=bundle,
        candidate_rows=rows,
    )
    assert (result.x, result.y) == (
        rows[selected]["x"],
        rows[selected]["y"],
    )
    assert result.eq_set_size == equivalence_size


def test_training_manifest_rejects_invalid_profile() -> None:
    with pytest.raises(ValueError, match="profile must be one of"):
        train_ranker._normalize_records(
            [
                {
                    "id": "bad",
                    "scene_id": "scene-bad",
                    "seed": 1,
                    "architecture": "dram",
                    "profile": "bad",
                }
            ],
            train_ranker.DEFAULT_TRAIN_MANIFEST,
        )


def test_training_provenance_records_material_candidate_settings() -> None:
    args = train_ranker._build_parser().parse_args([])
    provenance = train_ranker.build_training_provenance(
        args,
        [{"id": "train"}],
        [{"id": "validation"}],
        {"random_state": args.seed},
    )
    candidate = provenance["candidate_generation"]
    assert candidate["training_candidate_cap"] == args.train_max_candidates
    assert candidate["validation_candidate_cap"] == (
        args.validation_max_candidates
    )
    assert candidate["candidate_delta"] == 0.10
    assert provenance["negative_rows_per_scene_cap"] == (
        args.negatives_per_scene
    )
    assert provenance["validation_selection"] == {
        "production_equivalent": True,
        "residual_enabled": False,
        "equivalence_margin": 0.05,
        "search_shape": [1000, 1000],
    }
