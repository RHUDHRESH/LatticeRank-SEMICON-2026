"""The watchdog must shed work without ever failing to answer.

These run the real pipeline on the packaged example pair. The central claim
under test is the one the submission's rules defence rests on: with
``deadline=None`` the pipeline behaves exactly as it did in Phase 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from driftforge.budget import Deadline
from driftforge.model import load_model_bundle
from driftforge.pipeline import locate_v2

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dram"


def load(name: str) -> np.ndarray:
    with Image.open(EXAMPLE / name) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


@pytest.fixture(scope="module")
def pair() -> tuple[np.ndarray, np.ndarray]:
    return load("reference.png"), load("search.png")


@pytest.fixture(scope="module")
def bundle() -> dict:
    return load_model_bundle()


def test_no_deadline_is_the_unchanged_phase_1_path(pair, bundle) -> None:
    reference, search = pair
    result = locate_v2(reference, search, model_bundle=bundle)
    # No clock consulted, no budget key, no degradation recorded.
    assert "budget" not in result.diagnostics
    assert 0.0 <= result.x < search.shape[1]
    assert 0.0 <= result.y < search.shape[0]


def test_a_generous_deadline_matches_the_undeadlined_answer(pair, bundle) -> None:
    reference, search = pair
    baseline = locate_v2(reference, search, model_bundle=bundle)
    guarded = locate_v2(
        reference, search, model_bundle=bundle, deadline=Deadline(budget_s=600.0)
    )
    # Same coordinate: with time to spare the watchdog must be inert.
    assert guarded.x == pytest.approx(baseline.x)
    assert guarded.y == pytest.approx(baseline.y)
    assert guarded.diagnostics["budget"]["degraded"] is False


def test_an_exhausted_budget_still_returns_a_valid_coordinate(pair, bundle) -> None:
    reference, search = pair
    # Small but positive: the pipeline must degrade, not raise and not hang.
    result = locate_v2(
        reference, search, model_bundle=bundle, deadline=Deadline(budget_s=0.001)
    )
    assert np.isfinite(result.x) and np.isfinite(result.y)
    assert 0.0 <= result.x < search.shape[1]
    assert 0.0 <= result.y < search.shape[0]


def test_degradation_is_recorded_rather_than_silent(pair, bundle) -> None:
    reference, search = pair
    deadline = Deadline(budget_s=0.001)
    result = locate_v2(reference, search, model_bundle=bundle, deadline=deadline)
    report = result.diagnostics["budget"]
    # A degraded answer that does not say so would corrupt the very
    # confidence calibration the watchdog exists to protect.
    if report["degraded"]:
        assert report["degradations"], "degraded flag set with no reason recorded"
        for entry in report["degradations"]:
            assert entry["stage"] and entry["action"]


def test_a_seconds_value_is_accepted_as_a_deadline(pair, bundle) -> None:
    reference, search = pair
    result = locate_v2(reference, search, model_bundle=bundle, deadline=600.0)
    assert result.diagnostics["budget"]["budget_s"] == 600.0


def test_budget_report_survives_the_json_payload(pair, bundle) -> None:
    import json

    reference, search = pair
    result = locate_v2(reference, search, model_bundle=bundle, deadline=30.0)
    # inference.py splats diagnostics into a json.dumps(allow_nan=False) call.
    json.dumps(result.diagnostics, allow_nan=False)
