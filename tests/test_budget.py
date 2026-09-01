"""Deadline behaviour under a driven clock.

Every test here uses an injected clock rather than real elapsed time, so the
degradation ladder is verified deterministically instead of by making a
machine slow.
"""
from __future__ import annotations

import pytest

from driftforge.budget import (
    DEFAULT_BUDGET_S,
    HARD_TIMEOUT_S,
    Deadline,
    resolve,
)


class FakeClock:
    """A clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make(budget: float = 10.0) -> tuple[Deadline, FakeClock]:
    clock = FakeClock()
    return Deadline(budget_s=budget, clock=clock), clock


def test_default_budget_leaves_headroom_under_the_hard_timeout() -> None:
    # The whole point of the watchdog is that a pair never reaches the
    # organizers' hard limit, so the default must not sit at it.
    assert DEFAULT_BUDGET_S < HARD_TIMEOUT_S
    assert DEFAULT_BUDGET_S <= 0.75 * HARD_TIMEOUT_S


def test_elapsed_and_remaining_track_the_clock() -> None:
    deadline, clock = make(10.0)
    assert deadline.elapsed() == pytest.approx(0.0)
    assert deadline.remaining() == pytest.approx(10.0)
    clock.advance(4.0)
    assert deadline.elapsed() == pytest.approx(4.0)
    assert deadline.remaining() == pytest.approx(6.0)


def test_remaining_clamps_at_zero_and_reports_expired() -> None:
    deadline, clock = make(10.0)
    clock.advance(99.0)
    assert deadline.remaining() == 0.0
    assert deadline.expired()


def test_affords_rejects_a_stage_that_does_not_fit() -> None:
    deadline, clock = make(10.0)
    clock.advance(2.0)                       # 8 s left, below the soft line
    assert deadline.affords("residual", estimate_s=3.0)
    assert not deadline.affords("residual", estimate_s=9.0)


def test_soft_fraction_blocks_new_work_late_in_the_budget() -> None:
    deadline, clock = make(10.0)
    clock.advance(8.0)                       # past the 0.75 soft line
    # 2 s remain and the stage claims to need 1 s, but an estimate that close
    # to the end is exactly the case the soft fraction exists to refuse.
    assert not deadline.affords("structural", estimate_s=1.0)
    # Free work is still allowed.
    assert deadline.affords("bookkeeping", estimate_s=0.0)


def test_degradations_are_recorded_with_context() -> None:
    deadline, clock = make(10.0)
    assert not deadline.degraded
    clock.advance(6.0)
    deadline.degrade("residual", "skipped: insufficient budget")
    assert deadline.degraded
    entry = deadline.degradations[0]
    assert entry.stage == "residual"
    assert entry.elapsed_s == pytest.approx(6.0)
    assert entry.remaining_s == pytest.approx(4.0)


def test_cap_for_scales_candidate_work_to_the_time_left() -> None:
    deadline, clock = make(10.0)
    clock.advance(5.0)                       # 5 s left
    # 1000 candidates at 10 ms each would need 10 s; only 500 fit.
    assert deadline.cap_for(unit_cost_s=0.01, requested=1000) == 500
    # A request that already fits is returned untouched.
    assert deadline.cap_for(unit_cost_s=0.001, requested=100) == 100


def test_cap_for_never_returns_zero_work() -> None:
    deadline, clock = make(10.0)
    clock.advance(10.0)                      # nothing left at all
    # Emitting no answer is the one outcome the budget exists to prevent.
    assert deadline.cap_for(unit_cost_s=1.0, requested=5000) == 1
    assert deadline.cap_for(unit_cost_s=1.0, requested=5000, minimum=25) == 25


def test_cap_for_passes_through_when_unit_cost_is_unknown() -> None:
    deadline, _ = make(10.0)
    assert deadline.cap_for(unit_cost_s=0.0, requested=321) == 321


def test_report_is_json_safe_and_states_degradation() -> None:
    deadline, clock = make(10.0)
    clock.advance(3.0)
    deadline.degrade("structural", "capped to 500 candidates")
    report = deadline.report()
    assert report["degraded"] is True
    assert report["budget_s"] == 10.0
    assert isinstance(report["degradations"], list)
    assert report["degradations"][0]["stage"] == "structural"

    import json

    json.dumps(report, allow_nan=False)      # must survive the JSON payload


def test_resolve_accepts_none_deadline_and_seconds() -> None:
    assert resolve(None) is None
    assert isinstance(resolve(7.5), Deadline)
    assert resolve(7.5).budget_s == 7.5
    existing = Deadline(budget_s=3.0)
    assert resolve(existing) is existing


def test_a_non_positive_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        Deadline(budget_s=0.0)
    with pytest.raises(ValueError):
        Deadline(budget_s=-1.0)


def test_negative_estimate_is_rejected() -> None:
    deadline, _ = make(10.0)
    with pytest.raises(ValueError):
        deadline.affords("stage", estimate_s=-1.0)
