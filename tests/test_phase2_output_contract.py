"""Focused tests for the Phase 2 output bounds in ``register.solve``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import register
from driftforge.dense import DenseMatch


@dataclass
class NeverRefine:
    """Deadline stub that forces the inexpensive fallback path."""

    def affords(self, _stage: str, estimate_s: float = 0.0) -> bool:
        return False

    def degrade(self, _stage: str, _action: str) -> None:
        return None


def test_found_pose_is_clamped_to_disclosed_output_bounds(monkeypatch) -> None:
    match = DenseMatch(x=17.0, y=23.0, scale=12.5, rotation=-8.5, score=0.9)
    monkeypatch.setattr(register, "dense_pose_search", lambda *_args, **_kwargs: match)

    result = register.solve(
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.uint8),
        NeverRefine(),
        presence=None,
        correctness=None,
    )

    assert result["found"] == 1
    assert result["scale"] == 12.0
    assert result["theta"] == -5.0
    assert result["x"] == 17.0 and result["y"] == 23.0


def test_rejected_pair_zeroes_pose_before_any_output_clamp(monkeypatch) -> None:
    match = DenseMatch(x=17.0, y=23.0, scale=7.5, rotation=8.5, score=0.1)
    monkeypatch.setattr(register, "dense_pose_search", lambda *_args, **_kwargs: match)

    result = register.solve(
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.uint8),
        NeverRefine(),
        presence=None,
        correctness=None,
    )

    assert result["found"] == 0
    assert all(result[key] == 0.0 for key in ("x", "y", "theta", "scale"))
    assert np.isfinite(result["score"])
