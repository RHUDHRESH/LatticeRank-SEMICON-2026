from __future__ import annotations

import math

from driftforge.generator import generate_sample
from driftforge.pipeline import locate_v2


def test_exact_wallpaper_uses_centre_rule_before_expensive_ranking() -> None:
    """Independent noise must not turn equivalent wallpaper sites into evidence."""
    sample = generate_sample(
        seed=900204,
        architecture="finfet",
        profile="ambiguous",
        search_supersample=2,
    )

    result = locate_v2(sample.reference, sample.search)
    error = math.hypot(result.x - sample.gt_x, result.y - sample.gt_y)

    assert error <= 5.0
    assert result.n_candidates >= 2_500
    assert result.eq_set_size == result.n_candidates
    assert result.used_residual is False
    assert result.x == 499.5
    assert result.y == 499.5
    assert result.diagnostics["selection_mode"] == (
        "periodic_wallpaper_centre_rule"
    )
    assert result.diagnostics["wallpaper_ambiguity"]["detected"] is True
