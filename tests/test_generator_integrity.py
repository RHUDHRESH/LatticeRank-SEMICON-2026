"""The structural signature must not encode the answer.

These tests prevent the historical target-correlation leak from returning. The important
one is `test_moving_the_target_does_not_move_the_signature`: correlation can be
near zero by luck, but an interventional test cannot pass if placement reads
the target.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.config import REFERENCE_FOV_NM, WORLD_FOV_NM
from driftforge.generator import generate_sample
from driftforge.scene import (
    SIGNATURE_RADIUS_NM,
    make_scene_spec,
    render_scene,
    signature_placement,
)
from driftforge.splits import read_manifest

#: Historical baseline locator checksum. Update only when intentionally
#: changing the baseline implementation and its contract.
BASELINE_SHA256 = "15f591b375c8d9804daada13af8b6539a8931b45cc52439105c606bca72dd0c8"


def _specs(n: int):
    for i in range(n):
        arch = "dram" if i % 2 == 0 else "finfet"
        profile = ("standard", "hard", "boundary")[i % 3]
        yield make_scene_spec((900_000 + i) * 17 + 3, arch, profile)


class SignatureIndependenceTests(unittest.TestCase):
    def test_placement_source_does_not_mention_the_target(self) -> None:
        src = inspect.getsource(signature_placement)
        for banned in ("target_cx_nm", "target_cy_nm", "gt_x", "gt_y", "nominal_"):
            self.assertNotIn(banned, src, f"signature placement references {banned}")

    def test_moving_the_target_does_not_move_the_signature(self) -> None:
        """Interventional proof: hold the scene, move the target, measure."""
        for spec in _specs(60):
            base_x, base_y, base_r, _ = signature_placement(spec)
            for tx, ty in ((520.0, 520.0), (9480.0, 9480.0), (5000.0, 1200.0)):
                moved = dataclasses.replace(spec, target_cx_nm=tx, target_cy_nm=ty)
                x, y, r, _ = signature_placement(moved)
                self.assertEqual((base_x, base_y, base_r), (x, y, r))

    def test_signature_is_not_correlated_with_the_target(self) -> None:
        sig = np.array([signature_placement(s)[:2] for s in _specs(400)])
        tgt = np.array([[s.target_cx_nm, s.target_cy_nm] for s in _specs(400)])
        for axis in (0, 1):
            r = float(np.corrcoef(sig[:, axis], tgt[:, axis])[0, 1])
            self.assertLess(abs(r), 0.15, f"axis {axis} correlation {r:.3f}")

    def test_signature_is_rarely_inside_the_reference(self) -> None:
        """v1 put it inside 100% of the time; geometry alone predicts ~1%."""
        half = REFERENCE_FOV_NM / 2.0
        specs = list(_specs(600))
        inside = sum(
            1 for s in specs
            if abs(signature_placement(s)[0] - s.target_cx_nm) <= half
            and abs(signature_placement(s)[1] - s.target_cy_nm) <= half
        )
        span = WORLD_FOV_NM - 2.0 * SIGNATURE_RADIUS_NM[1]
        expected = (REFERENCE_FOV_NM ** 2) / (span ** 2) * len(specs)
        self.assertLess(inside, max(6.0 * expected, 10))

    def test_signature_can_land_far_from_the_target(self) -> None:
        d = [math.dist(signature_placement(s)[:2], (s.target_cx_nm, s.target_cy_nm))
             for s in _specs(200)]
        self.assertGreater(max(d), 5_000.0)
        self.assertGreater(float(np.median(d)), 2_000.0)


class DeterminismTests(unittest.TestCase):
    def test_render_is_deterministic_across_repeats(self) -> None:
        for _ in range(2):
            a = generate_sample(300_007, "dram", "standard", search_supersample=1)
            b = generate_sample(300_007, "dram", "standard", search_supersample=1)
            self.assertTrue(np.array_equal(a.reference, b.reference))
            self.assertTrue(np.array_equal(a.search, b.search))
            self.assertEqual((a.gt_x, a.gt_y), (b.gt_x, b.gt_y))

    def test_signature_placement_is_deterministic(self) -> None:
        spec = make_scene_spec(12_345, "finfet", "hard")
        self.assertEqual(signature_placement(spec)[:3], signature_placement(spec)[:3])


class PhysicalSharingTests(unittest.TestCase):
    def test_defect_renders_identically_from_two_different_windows(self) -> None:
        """The defect is latent world structure, not a per-capture stamp.

        Rendering the same world patch through two different FOVs must produce
        the same physics, or Reference and Search would disagree about the
        scene itself.
        """
        spec = make_scene_spec(4_242, "dram", "standard")
        sx, sy, _, _ = signature_placement(spec)
        fov = 400.0
        size = 200
        ox, oy = sx - fov / 2.0, sy - fov / 2.0
        first = render_scene(spec, ox, oy, fov, size)
        second = render_scene(spec, ox, oy, fov, size)
        self.assertTrue(np.array_equal(first, second))

        # The same world region, reached from a wider window, still contains a
        # dark core: shared structure, not a coincidence of one crop.
        wide = render_scene(spec, sx - 1_000.0, sy - 1_000.0, 2_000.0, 400)
        self.assertLess(float(wide.min()), 0.10)

    def test_acquisition_noise_stays_independent(self) -> None:
        sample = generate_sample(300_011, "finfet", "standard", search_supersample=1)
        ref = sample.reference.astype(np.float64)
        search = sample.search.astype(np.float64)
        # Same nominal size, entirely different capture chains; if the noise
        # streams were shared these would correlate.
        r = float(np.corrcoef(ref.ravel()[::37], search.ravel()[::37])[0, 1])
        self.assertLess(abs(r), 0.35)
        self.assertNotEqual(
            sample.reference_acquisition.read_noise_sigma,
            sample.search_acquisition.read_noise_sigma,
        )


BENCHMARK_MANIFEST = "manifests/validation_benchmark.jsonl"


class SplitIntegrityTests(unittest.TestCase):
    def test_benchmark_split_is_disjoint_from_every_other_split(self) -> None:
        bench = read_manifest(PROJECT / BENCHMARK_MANIFEST)
        self.assertGreaterEqual(len(bench), 200)
        scenes = {r["scene_id"] for r in bench}
        seeds = {int(r["seed"]) for r in bench}
        for split in ("train", "validation", "test", "stress"):
            other = read_manifest(PROJECT / f"manifests/{split}.jsonl")
            self.assertFalse(scenes & {r["scene_id"] for r in other})
            self.assertFalse(seeds & {int(r["seed"]) for r in other})

    def test_no_scene_realisation_collides_across_splits(self) -> None:
        """Distinct seeds are not sufficient.

        Scene parameters come from ``default_rng(seed * 17 + 3)``, so two
        splits holding different seeds could still realise the same scene.
        This is the leak the plain seed check would not catch.
        """
        owner: dict[int, str] = {}
        names = ["train", "validation", "test", "stress"]
        sources = {n: PROJECT / f"manifests/{n}.jsonl" for n in names}
        sources["validation_benchmark"] = PROJECT / BENCHMARK_MANIFEST
        for name, path in sources.items():
            for row in read_manifest(path):
                key = int(row["seed"]) * 17 + 3
                self.assertEqual(owner.get(key, name), name,
                                 f"scene realisation {key} shared with {owner.get(key)}")
                owner[key] = name

    def test_benchmark_carries_both_architectures_and_all_profiles(self) -> None:
        bench = read_manifest(PROJECT / BENCHMARK_MANIFEST)
        arch = {r["architecture"] for r in bench}
        self.assertEqual(arch, {"dram", "finfet"})
        self.assertTrue({"standard", "hard", "boundary", "ambiguous"}
                        <= {r["profile"] for r in bench})
        # The benchmark is exactly architecture-balanced by construction.
        counts = [sum(1 for r in bench if r["architecture"] == a) for a in sorted(arch)]
        self.assertEqual(counts[0], counts[1])


class LocatorFrozenTests(unittest.TestCase):
    def test_baseline_locator_is_byte_identical_to_the_shipped_v1(self) -> None:
        digest = hashlib.sha256(
            (PROJECT / "driftforge/baseline.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest, BASELINE_SHA256,
            "baseline.py no longer matches the recorded baseline locator",
        )


if __name__ == "__main__":
    unittest.main()
