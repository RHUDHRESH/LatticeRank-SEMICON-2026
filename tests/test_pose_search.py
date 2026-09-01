"""Coarse pose search: geometry, recall, and the conventions it depends on.

The failure that matters most here is silent: an off-by-a-few coordinate
mapping or a wrong effective scale still returns a plausible pool of
candidates, and the error only shows up as a localization score that is
mysteriously a few pixels worse than it should be.
"""
from __future__ import annotations

import numpy as np
import pytest

from driftforge.channels import prepare_search, response_maps
from driftforge.pose_search import (
    COARSE_ROTATIONS,
    COARSE_SCALES,
    DECIMATION,
    coarse_pose_candidates,
    decimate,
)


def test_disclosed_ranges_are_covered_by_the_coarse_grids() -> None:
    # The addendum discloses s in [8, 12] and theta in +/-5 deg, and says
    # hard-coding those bounds is intended. The grids must span them exactly.
    assert min(COARSE_SCALES) == 8.0
    assert max(COARSE_SCALES) == 12.0
    assert min(COARSE_ROTATIONS) == -5.0
    assert max(COARSE_ROTATIONS) == 5.0


def test_decimation_shrinks_by_the_requested_factor() -> None:
    image = np.zeros((1000, 1000), dtype=np.uint8)
    pyramid = decimate(image, 4)
    assert pyramid.factor == 4.0
    assert pyramid.full_shape == (1000, 1000)
    assert abs(pyramid.image.shape[0] - 250) <= 1
    assert abs(pyramid.image.shape[1] - 250) <= 1


@pytest.mark.parametrize("factor", [3, 4, 5])
@pytest.mark.parametrize("point", [(300, 700), (120, 90), (880, 640)])
def test_coordinate_mapping_round_trips_within_the_quantization_floor(
    factor: int, point: tuple[int, int]
) -> None:
    # One decimated pixel spans `factor` full-resolution pixels, so the best
    # achievable accuracy is +/- factor/2. Anything worse is a mapping bug --
    # in particular, using the nominal factor instead of the (n-1)/(m-1) ratio
    # that ndimage.zoom actually applies drifts steadily across the image.
    x, y = point
    image = np.zeros((1000, 1000), dtype=np.uint8)
    image[y, x] = 255
    pyramid = decimate(image, factor)
    yy, xx = np.unravel_index(int(np.argmax(pyramid.image)), pyramid.image.shape)
    fx, fy = pyramid.to_full(float(xx), float(yy))
    assert np.hypot(fx - x, fy - y) <= factor, (fx, fy, point)


def test_decimation_antialiases_before_resampling() -> None:
    # A fine stripe pattern above the post-decimation Nyquist limit must not
    # survive as a strong low-frequency pattern. Without the Gaussian it
    # aliases -- and on periodic device fields the alias lands exactly where a
    # matcher looks.
    fine = np.zeros((600, 600), dtype=np.float32)
    fine[:, ::2] = 255.0
    plain = fine[::5, ::5]
    filtered = decimate(fine, 5).image
    assert float(filtered.std()) < float(plain.std())


def test_prepared_search_channels_do_not_change_the_maps() -> None:
    rng = np.random.default_rng(20260830)
    search = rng.integers(0, 255, size=(240, 240), dtype=np.uint8)
    reference = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    direct = response_maps(reference, search, 1.0, 0.0)
    hoisted = response_maps(
        reference, search, 1.0, 0.0, prepared=prepare_search(search)
    )
    for channel, values in direct.maps.items():
        assert np.array_equal(values, hoisted.maps[channel]), channel


def test_a_supplied_template_is_used_verbatim() -> None:
    rng = np.random.default_rng(7)
    search = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
    reference = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    template = rng.random((24, 24)).astype(np.float32)
    maps = response_maps(reference, search, 9.0, 1.0, template=template)
    # half_w/half_h are derived from the template, so they prove which one ran.
    assert maps.half_w == pytest.approx((template.shape[1] - 1) / 2.0)
    assert maps.half_h == pytest.approx((template.shape[0] - 1) / 2.0)
    assert maps.scale == 9.0 and maps.rotation == 1.0


def test_candidates_are_in_full_resolution_coordinates() -> None:
    rng = np.random.default_rng(11)
    reference = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    search = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    found = coarse_pose_candidates(
        reference, search, scales=(10.0,), rotations=(0.0,), max_candidates=50
    )
    assert found, "the coarse stage produced no candidates at all"
    for record in found:
        assert 0.0 <= record["x"] <= 1000.0
        assert 0.0 <= record["y"] <= 1000.0
        assert record["scale"] == 10.0
        assert record["rotation"] == 0.0
        assert np.isfinite(record["pose_score"])


def test_a_degenerate_reference_yields_no_candidates_rather_than_nan() -> None:
    # A constant template makes every normalized correlation NaN. The stage
    # must skip it, not propagate NaN into the candidate scores.
    flat = np.full((1000, 1000), 128, dtype=np.uint8)
    rng = np.random.default_rng(3)
    search = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    found = coarse_pose_candidates(flat, search, scales=(10.0,), rotations=(0.0,))
    assert found == []


def test_the_deadline_stops_the_grid_and_records_why() -> None:
    from driftforge.budget import Deadline

    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            self.now += 5.0          # every reading burns 5 s
            return self.now

    rng = np.random.default_rng(5)
    reference = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    search = rng.integers(0, 255, size=(1000, 1000), dtype=np.uint8)
    deadline = Deadline(budget_s=1.0, clock=Clock())
    coarse_pose_candidates(reference, search, deadline=deadline)
    assert deadline.degraded
    assert any(d.stage == "coarse_pose_grid" for d in deadline.degradations)


def test_default_decimation_matches_the_measured_choice() -> None:
    # Recorded so a future change to this constant is a deliberate decision
    # with a fresh measurement behind it, not a silent regression.
    assert DECIMATION == 3
