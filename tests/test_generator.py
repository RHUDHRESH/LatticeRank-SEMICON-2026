from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.baseline import locate
from driftforge.generator import generate_sample
from driftforge.ml import model_arrays
from driftforge.splits import assert_disjoint, make_records


class GeneratorTests(unittest.TestCase):
    def test_shape_dtype_range_and_label(self) -> None:
        sample = generate_sample(11, "dram", "standard", search_supersample=1, return_debug=True)
        self.assertEqual(sample.reference.shape, (1000, 1000))
        self.assertEqual(sample.search.shape, (1000, 1000))
        self.assertEqual(sample.reference.dtype, np.uint8)
        self.assertEqual(sample.search.dtype, np.uint8)
        self.assertTrue(0 <= sample.gt_x < 1000)
        self.assertTrue(0 <= sample.gt_y < 1000)
        self.assertIsNotNone(sample.target_mask)

    def test_deterministic_but_independent_captures(self) -> None:
        a = generate_sample(19, "finfet", "hard", search_supersample=1)
        b = generate_sample(19, "finfet", "hard", search_supersample=1)
        self.assertTrue(np.array_equal(a.reference, b.reference))
        self.assertTrue(np.array_equal(a.search, b.search))
        # Reference and Search do not share pixels/noise or even pixel scale.
        search_crop = a.search[max(0, int(a.gt_y) - 50):int(a.gt_y) + 50, max(0, int(a.gt_x) - 50):int(a.gt_x) + 50]
        self.assertFalse(np.array_equal(a.reference[: search_crop.shape[0], : search_crop.shape[1]], search_crop))

    def test_architectures_are_not_identical(self) -> None:
        dram = generate_sample(23, "dram", "standard", search_supersample=1)
        finfet = generate_sample(23, "finfet", "standard", search_supersample=1)
        self.assertGreater(float(np.mean(np.abs(dram.search.astype(float) - finfet.search.astype(float)))), 1.0)

    def test_ambiguous_target_obeys_center_rule(self) -> None:
        sample = generate_sample(31, "dram", "ambiguous", search_supersample=1)
        self.assertLess(math.hypot(sample.gt_x - 499.5, sample.gt_y - 499.5), 12.0)

    def test_model_arrays(self) -> None:
        sample = generate_sample(37, "finfet", "standard", search_supersample=1)
        arrays = model_arrays(sample)
        self.assertEqual(arrays["reference"].shape, (1, 1000, 1000))
        self.assertEqual(arrays["reference_at_search_scale"].shape, (1, 100, 100))
        self.assertEqual(arrays["target_heatmap_stride4"].shape, (1, 250, 250))

    def test_scene_disjoint_splits(self) -> None:
        groups = {name: make_records(name, 20) for name in ("train", "validation", "test", "stress")}
        assert_disjoint(groups)
        self.assertEqual(len({row["architecture"] for row in groups["train"]}), 2)

    def test_classical_baseline_smoke(self) -> None:
        """The locator runs end to end and emits a usable in-bounds coordinate.

        A historical sub-2 px assertion was only
        reachable because the generator placed a unique high-contrast landmark
        inside the Reference FOV in every non-ambiguous sample, next to the
        answer; with the leak removed this seed lands ~425 px away. The
        accuracy of the classical baseline is now measured properly on >=300
        leak-free pairs instead of being asserted from one sample, and this
        stays a smoke test.
        """
        sample = generate_sample(300_000, "finfet", "standard", search_supersample=1)
        result = locate(sample.reference, sample.search, rotations=(0.0,), scales=(1.0,))
        self.assertTrue(0.0 <= result.x < 1000.0)
        self.assertTrue(0.0 <= result.y < 1000.0)
        self.assertTrue(math.isfinite(result.score))
        self.assertTrue(result.candidates)


if __name__ == "__main__":
    unittest.main()

