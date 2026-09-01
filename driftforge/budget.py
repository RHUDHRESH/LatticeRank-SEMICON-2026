"""Per-pair time budget for the Phase 2 scored run.

The Phase 2 run environment gives a **median 5 s per pair and a hard 20 s
timeout, and a pair that exceeds the timeout scores zero** -- losing its
localization credit, its pose credit, and contributing a wrong ``found`` flag
to the rejection F1. A pair that overruns is therefore far more expensive than
a pair answered imprecisely but on time.

Phase 1 measured 2.86 s median but a 30.32 s P95 and a 45.20 s maximum over
the 80-pair validation set (``results/runtime.json``), so roughly the slowest
twentieth of pairs already exceed the hard limit before Phase 2 adds any
scale or rotation search. This module exists to convert that tail from "scores
zero" into "answers with less refinement".

Design rules, in priority order:

1. **Always emit an answer.** Degrading is always preferable to overrunning.
2. **Degrade in a defined order**, cheapest evidence lost first, so the
   behaviour under pressure is predictable and reportable rather than
   whatever the profiler happened to hit.
3. **A degraded answer is labelled**, and is distinguishable from both a
   confident answer and an internal error. Silent degradation would corrupt
   the confidence calibration it is meant to protect.
4. **Disabled by default.** ``Deadline`` is opt-in; passing ``None`` anywhere
   restores the exact Phase 1 code path, which keeps the "disable it and you
   get the declared Phase 1 method back" ablation honest.

Nothing here interrupts running code. Preemption in pure Python costs either a
thread or a signal handler, neither of which is safe inside numpy/scipy calls;
instead the pipeline consults the deadline at stage boundaries, which is
sufficient because the expensive stages are per-candidate loops that can be
shortened rather than single indivisible calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

#: Organizer-stated hard timeout, in seconds. A pair beyond this scores zero.
HARD_TIMEOUT_S = 20.0

#: Organizer-stated median target, in seconds.
MEDIAN_TARGET_S = 5.0

#: Default per-pair budget. Deliberately well under ``HARD_TIMEOUT_S``: the
#: reference machine is a 4-core box that may be slower than the host these
#: numbers were measured on, the budget excludes image loading and CSV
#: writing, and a stage boundary can only be checked *between* stages, so the
#: last stage entered still has to finish. The headroom absorbs all three.
DEFAULT_BUDGET_S = 12.0

#: Fraction of the budget after which no new optional evidence stage is
#: started even if it is individually estimated to fit.
SOFT_FRACTION = 0.75


@dataclass
class Degradation:
    """One recorded decision to do less work than the full pipeline would."""

    stage: str
    action: str
    elapsed_s: float
    remaining_s: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "stage": self.stage,
            "action": self.action,
            "elapsed_s": round(self.elapsed_s, 4),
            "remaining_s": round(self.remaining_s, 4),
        }


@dataclass
class Deadline:
    """A per-pair wall-clock budget consulted at pipeline stage boundaries.

    ``clock`` is injectable so tests can drive degradation deterministically
    instead of relying on a machine being slow.

    Typical use::

        deadline = Deadline(budget_s=12.0)
        if deadline.affords("residual", estimate_s=2.0):
            ...
        else:
            deadline.degrade("residual", "skipped: insufficient budget")
    """

    budget_s: float = DEFAULT_BUDGET_S
    clock: Callable[[], float] = time.perf_counter
    soft_fraction: float = SOFT_FRACTION
    _start: float = field(init=False)
    degradations: list[Degradation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.budget_s <= 0.0:
            raise ValueError("budget_s must be positive")
        self._start = self.clock()

    # -- state -------------------------------------------------------------

    def elapsed(self) -> float:
        return float(self.clock() - self._start)

    def remaining(self) -> float:
        return float(max(0.0, self.budget_s - self.elapsed()))

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    @property
    def degraded(self) -> bool:
        return bool(self.degradations)

    # -- decisions ---------------------------------------------------------

    def affords(self, stage: str, estimate_s: float = 0.0) -> bool:
        """Return whether ``stage`` should be started.

        A stage is started only when the estimated cost fits in what is left
        *and* the soft fraction of the budget has not already been spent. The
        second condition matters because a cost estimate is only ever an
        estimate: without it, a stage estimated at 1 s with 1.1 s remaining
        would be started at the worst possible moment.
        """
        if estimate_s < 0.0:
            raise ValueError("estimate_s must be non-negative")
        remaining = self.remaining()
        if remaining <= 0.0:
            return False
        if self.elapsed() >= self.soft_fraction * self.budget_s:
            return estimate_s <= 0.0
        return estimate_s <= remaining

    def degrade(self, stage: str, action: str) -> None:
        """Record that ``stage`` was skipped or shortened, and why."""
        self.degradations.append(
            Degradation(
                stage=stage,
                action=action,
                elapsed_s=self.elapsed(),
                remaining_s=self.remaining(),
            )
        )

    def cap_for(self, unit_cost_s: float, requested: int, *, minimum: int = 1) -> int:
        """Return how many per-candidate evaluations fit in the time left.

        ``unit_cost_s`` is the measured cost of one evaluation. The result is
        clamped to ``[minimum, requested]`` -- the pipeline must always do at
        least some work, because emitting nothing is the one outcome the
        budget exists to prevent.
        """
        if requested <= minimum:
            return max(minimum, requested)
        if unit_cost_s <= 0.0:
            return requested
        affordable = int(self.remaining() / unit_cost_s)
        return int(max(minimum, min(requested, affordable)))

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict[str, object]:
        """Budget diagnostics for the result payload and the run log."""
        return {
            "budget_s": self.budget_s,
            "elapsed_s": round(self.elapsed(), 4),
            "remaining_s": round(self.remaining(), 4),
            "degraded": self.degraded,
            "degradations": [d.as_dict() for d in self.degradations],
        }


def resolve(deadline: Deadline | float | None) -> Deadline | None:
    """Accept a ``Deadline``, a bare budget in seconds, or ``None``.

    ``None`` means no watchdog, and therefore the exact Phase 1 behaviour.
    """
    if deadline is None:
        return None
    if isinstance(deadline, Deadline):
        return deadline
    return Deadline(budget_s=float(deadline))
