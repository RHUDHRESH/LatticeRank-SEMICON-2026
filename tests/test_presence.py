"""Presence evidence must be finite, ordered, and measure distinct sites.

The failure that matters here is silent: a NaN or a margin measured against
the peak's own neighbour produces a confidence number that looks reasonable
and means nothing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from driftforge.presence import (
    PRESENCE_FEATURES,
    feature_vector,
    presence_evidence,
)


def row(x: float, y: float, raw: float, mid: float | None = None,
        dir_: float | None = None) -> dict:
    return {
        "x": x, "y": y,
        "raw": raw,
        "midband": raw if mid is None else mid,
        "directionality": raw if dir_ is None else dir_,
    }


def test_empty_pool_is_neutral_and_finite() -> None:
    ev = presence_evidence([])
    assert ev["pool_is_empty"] == 1.0
    assert all(math.isfinite(v) for v in ev.values())
    # An empty pool is maximally ambiguous, not maximally confident.
    assert ev["eq_fraction"] == 1.0
    assert ev["margin_ratio"] == 0.0


def test_margin_ignores_the_peaks_own_neighbourhood() -> None:
    # A near-neighbour of the peak is the same site sampled twice. Counting it
    # as the runner-up would report a margin of ~0 on an unambiguous scene.
    rows = [
        row(500, 500, 0.90),
        row(502, 500, 0.89),   # 2 px away: same site
        row(600, 500, 0.30),   # a genuinely different site
    ]
    ev = presence_evidence(rows, pitch_px=40.0)
    assert ev["margin_raw"] == pytest.approx(0.60, abs=1e-6)


def test_margin_uses_pitch_when_supplied() -> None:
    rows = [
        row(500, 500, 0.90),
        row(530, 500, 0.85),   # 30 px: inside a 40 px pitch, same site
        row(600, 500, 0.20),   # 100 px: distinct
    ]
    tight = presence_evidence(rows, pitch_px=10.0)   # 30 px is now distinct
    loose = presence_evidence(rows, pitch_px=40.0)   # 30 px is not
    assert tight["margin_raw"] == pytest.approx(0.05, abs=1e-6)
    assert loose["margin_raw"] == pytest.approx(0.70, abs=1e-6)


def test_an_unambiguous_scene_outscores_a_flat_one() -> None:
    sharp = [row(500, 500, 0.95)] + [
        row(100 + 40 * i, 100, 0.10) for i in range(20)
    ]
    flat = [row(100 + 40 * i, 100, 0.50) for i in range(21)]
    a = presence_evidence(sharp, pitch_px=20.0)
    b = presence_evidence(flat, pitch_px=20.0)
    assert a["margin_ratio"] > b["margin_ratio"]
    assert a["psr"] > b["psr"]
    assert a["eq_fraction"] < b["eq_fraction"]


def test_non_finite_responses_never_leak_into_the_output() -> None:
    rows = [
        row(500, 500, 0.9),
        row(600, 500, float("nan")),
        row(700, 500, float("inf")),
        row(800, 500, float("-inf")),
    ]
    ev = presence_evidence(rows, pitch_px=20.0)
    assert all(math.isfinite(v) for v in ev.values()), ev


def test_channel_agreement_detects_disagreeing_channels() -> None:
    agree = [
        row(500, 500, 0.9, 0.9, 0.9),
        row(600, 500, 0.1, 0.1, 0.1),
    ]
    disagree = [
        row(500, 500, 0.9, 0.1, 0.1),
        row(600, 500, 0.1, 0.9, 0.9),
    ]
    assert presence_evidence(agree, pitch_px=20.0)["channel_agreement"] == 1.0
    assert presence_evidence(disagree, pitch_px=20.0)["channel_agreement"] < 1.0


def test_psr_is_zero_rather_than_wild_on_a_tiny_pool() -> None:
    # Fewer than 8 background candidates cannot estimate a spread; the honest
    # answer is "no evidence", not a ratio computed from three points.
    ev = presence_evidence([row(500, 500, 0.9), row(600, 500, 0.2)], pitch_px=20.0)
    assert ev["psr"] == 0.0


def test_feature_vector_matches_the_frozen_order() -> None:
    ev = presence_evidence([row(500, 500, 0.9), row(600, 500, 0.2)], pitch_px=20.0)
    vector = feature_vector(ev)
    assert vector.shape == (len(PRESENCE_FEATURES),)
    assert np.isfinite(vector).all()
    for index, name in enumerate(PRESENCE_FEATURES):
        assert vector[index] == pytest.approx(ev[name])


def test_feature_vector_tolerates_a_missing_feature() -> None:
    # A shipped weight vector must never be silently misaligned by a missing
    # key; absent features read as zero, in the frozen order.
    vector = feature_vector({"peak_raw": 1.0})
    assert vector.shape == (len(PRESENCE_FEATURES),)
    assert vector[PRESENCE_FEATURES.index("peak_raw")] == 1.0
    assert vector[PRESENCE_FEATURES.index("psr")] == 0.0
