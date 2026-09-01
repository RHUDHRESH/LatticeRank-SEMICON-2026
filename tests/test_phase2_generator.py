"""Phase 2 acceptance tests (prompt §11).

These verify the generator contract: zoom by field of view, measured ground
truth, honest absent pairs, decoys, CD bias on the search only, fixed image
geometry, no metadata in filenames, byte-identical regeneration, the RGB
optical mode, split disjointness including derived realizations, and that
severity is not reachable from trivial image statistics.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.config import WORLD_FOV_NM
from driftforge.generator import generate_phase2_sample
from driftforge.phase2 import (
    GRAY_PAIR_PROB,
    HOLDOUT_FAMILY,
    build_citations,
    channel_spread,
    sample_rgb_chroma,
)
from driftforge.pose import rotation_oracle, scale_oracle
from driftforge.splits import SPLIT_SEED_BASE, assert_disjoint, make_records

MANIFEST_FIELDS = {
    "id", "split", "scene_seed", "ref_seed", "search_seed", "architecture",
    "preset_family", "severity", "modality", "present", "gt_x", "gt_y",
    "gt_theta", "gt_scale", "n_decoys", "decoy_sites", "occlusion_frac",
    "cd_bias_pct", "edge_case", "ref_image", "search_image",
}


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class Phase2GeneratorTests(unittest.TestCase):
    def test_zoom_is_produced_by_fov_not_resize(self) -> None:
        """The reference is rendered at its own FOV; nothing is resized."""
        import driftforge.generator as generator_module

        calls = []
        original = generator_module.render_scene_with_layers

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        generator_module.render_scene_with_layers = spy
        try:
            sample = generate_phase2_sample(
                1_100_001, split="p2_train", present=True, edge_case="s_high",
                search_supersample=1,
            )
        finally:
            generator_module.render_scene_with_layers = original
        nominal = sample.metadata["nominal_zoom"]
        ref_calls = [
            (args, kwargs)
            for args, kwargs in calls
            if len(args) >= 5 and abs(args[3] - WORLD_FOV_NM / nominal) < 1e-6 and args[4] == 1000
        ]
        self.assertTrue(ref_calls, "reference must be rendered at WORLD_FOV_NM / s, size 1000")
        # and never by resampling a wider render to fake the scale
        self.assertAlmostEqual(sample.metadata["ref_fov_nm"], WORLD_FOV_NM / nominal, places=6)
        self.assertNotAlmostEqual(sample.metadata["ref_fov_nm"], WORLD_FOV_NM / 10.0, places=2)

    def test_true_scale_matches_label_via_oracle(self) -> None:
        sample = generate_phase2_sample(1_100_020, split="p2_train", present=True, search_supersample=1)
        rec, _ = scale_oracle(
            sample.reference, sample.search, sample.gt_x, sample.gt_y,
            sample.gt_theta, shape_scale=sample.gt_scale,
        )
        self.assertLess(abs(rec - sample.gt_scale) / sample.gt_scale, 0.005)

    def test_rotation_label_matches_oracle_sign(self) -> None:
        """The sign trap (§2.3): the label must be the convention the oracle
        recovers, verified against the brute-force search over the full span."""
        sample = generate_phase2_sample(1_100_021, split="p2_train", present=True, search_supersample=1)
        rec, _ = rotation_oracle(
            sample.reference, sample.search, sample.gt_x, sample.gt_y, sample.gt_scale
        )
        self.assertLess(abs(rec - sample.gt_theta), 0.15)
        # the naive search-minus-reference hypothesis has the opposite sign
        predicted = sample.metadata["predicted_theta_deg"]
        self.assertLess(abs(predicted - sample.gt_theta), 1.5)
        self.assertAlmostEqual(
            predicted, -(sample.metadata["search_acquisition"]["search_rotation_deg"]
                         - sample.metadata["ref_acquisition"]["ref_rotation_deg"]),
            places=3,
        )

    def test_absent_pairs_match_present_marginals(self) -> None:
        rng = np.random.default_rng(0)
        seeds = rng.choice(np.arange(1_100_000, 1_300_000), size=48, replace=False)
        present_s, absent_s = [], []
        for seed in seeds:
            sample = generate_phase2_sample(int(seed), split="p2_train", search_supersample=1)
            (present_s if sample.present else absent_s).append(sample.gt_scale)
        self.assertGreater(len(absent_s), 4)
        # the zoom draw precedes the present/absent branch, so both marginals
        # are draws from the same uniform: a two-sample KS cannot separate them
        from driftforge.phase2 import _ks_p_value

        stat = _ks(present_s, absent_s)
        self.assertGreater(_ks_p_value(stat, len(present_s), len(absent_s)), 0.05)

    def test_absent_reference_is_not_from_search_scene(self) -> None:
        from driftforge.scene import make_scene_spec

        sample = generate_phase2_sample(1_100_030, split="p2_train", present=False, search_supersample=1)
        self.assertFalse(sample.present)
        meta = sample.metadata
        self.assertEqual(meta["absent_scene_seed"], meta["scene_seed"] ^ 0x5EED)
        self.assertIsNone(sample.gt_x)
        self.assertIsNone(sample.gt_theta)
        search_scene = make_scene_spec(meta["scene_seed"], sample.architecture, meta["profile"], preset=sample.preset_family)
        ref_scene = make_scene_spec(meta["absent_scene_seed"], sample.architecture, meta["profile"], preset=sample.preset_family)
        self.assertNotEqual(search_scene, ref_scene)
        # same family, same severity ladder
        self.assertEqual(search_scene.preset, ref_scene.preset)

    def test_decoys_present_when_requested(self) -> None:
        sample = generate_phase2_sample(1_100_040, split="p2_train", present=True, n_decoys=3, search_supersample=1)
        self.assertEqual(sample.n_decoys, 3)
        self.assertEqual(len(sample.metadata["decoy_sites"]), 3)
        template_half = WORLD_FOV_NM / sample.gt_scale / 2.0 / 10.0
        for x, y in sample.metadata["decoy_sites"]:
            self.assertTrue(0 <= x <= 1000 and 0 <= y <= 1000)
            self.assertGreater(np.hypot(x - sample.gt_x, y - sample.gt_y), template_half)
        # absent pairs carry decoys too (leakage rule §3.4: decoy presence
        # must not correlate with `present`); their near-duplicates mirror
        # the search scene's own motif, not the foreign reference
        absent = generate_phase2_sample(1_100_041, split="p2_train", present=False, n_decoys=2, search_supersample=1)
        self.assertEqual(absent.n_decoys, 2)
        self.assertFalse(absent.present)
        self.assertIsNone(absent.gt_x)

    def test_cd_bias_applied_to_search_only(self) -> None:
        import driftforge.generator as generator_module

        calls = []
        original = generator_module.render_scene_with_layers

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        generator_module.render_scene_with_layers = spy
        try:
            sample = generate_phase2_sample(1_100_050, split="p2_train", present=True, search_supersample=1)
        finally:
            generator_module.render_scene_with_layers = original
        cd = sample.metadata["cd_bias_pct"] / 100.0
        cd_args = [kwargs.get("cd_bias_frac", 0.0) for _, kwargs in calls]
        self.assertTrue(any(abs(v - cd) < 1e-3 for v in cd_args))
        search_call = next(i for i, v in enumerate(cd_args) if abs(v - cd) < 1e-3)
        ref_calls = [i for i, v in enumerate(cd_args) if i != search_call and abs(v) < 1e-12]
        self.assertTrue(ref_calls, "reference renders with cd_bias_frac=0.0")

    def test_all_images_are_1000x1000_for_every_scale(self) -> None:
        for edge in ("s_low", "s_high", None):
            sample = generate_phase2_sample(
                1_100_060, split="p2_train", present=True, edge_case=edge, search_supersample=1
            )
            self.assertEqual(sample.reference.shape, (1000, 1000))
            self.assertEqual(sample.search.shape, (1000, 1000))
            self.assertEqual(sample.reference.dtype, np.uint8)

    def test_no_metadata_in_filenames(self) -> None:
        from scripts.generate_dataset import _phase2_record

        sample = generate_phase2_sample(1_300_500, split="p2_val", present=True, search_supersample=1)
        record = _phase2_record(137, "p2_val", sample, export_debug=False)
        pattern = re.compile(r"^images/\d{5}_(ref|search)\.png$")
        self.assertRegex(record["ref_image"], pattern)
        self.assertRegex(record["search_image"], pattern)
        blob = json.dumps({"ref_image": record["ref_image"], "search_image": record["search_image"]})
        for token in (str(record["scene_seed"]), str(record["ref_seed"]), str(record["search_seed"])):
            self.assertNotIn(token, blob)
        # the strict pattern already excludes any other metadata encoding
        self.assertNotIn("severity", blob)
        self.assertNotIn("present", blob)

    def test_regeneration_is_byte_identical(self) -> None:
        a = generate_phase2_sample(1_100_070, split="p2_train", present=True, search_supersample=1)
        b = generate_phase2_sample(1_100_070, split="p2_train", present=True, search_supersample=1)
        self.assertEqual(_png_bytes(a.reference), _png_bytes(b.reference))
        self.assertEqual(_png_bytes(a.search), _png_bytes(b.search))
        self.assertEqual(a.metadata, b.metadata)

    def test_rgb_mode_produces_three_channels(self) -> None:
        sample = generate_phase2_sample(1_100_080, split="p2_train", modality="rgb", present=True, search_supersample=1)
        self.assertEqual(sample.reference.shape, (1000, 1000, 3))
        self.assertEqual(sample.search.shape, (1000, 1000, 3))
        self.assertEqual(sample.reference.dtype, np.uint8)

    def test_rgb_mode_includes_effectively_gray_pairs(self) -> None:
        self.assertGreaterEqual(GRAY_PAIR_PROB, 0.15)
        gray = None
        seed = 1_100_200
        for _ in range(40):  # ~18% of pairs: expect a hit within a few draws
            candidate = generate_phase2_sample(seed, split="p2_train", modality="rgb", present=True, search_supersample=1)
            if candidate.metadata["rgb_gray_pair"]:
                gray = candidate
                break
            seed += 1
        self.assertIsNotNone(gray, "no effectively-gray pair produced in 40 draws")
        self.assertLess(channel_spread(gray.reference), 3.0)
        chromatic = generate_phase2_sample(1_100_080, split="p2_train", modality="rgb", present=True, search_supersample=1)
        self.assertGreater(channel_spread(chromatic.reference), 6.0)

    def test_manifest_schema_complete(self) -> None:
        sample = generate_phase2_sample(1_300_501, split="p2_val", present=True, search_supersample=1)
        from scripts.generate_dataset import _phase2_record

        record = _phase2_record(7, "p2_val", sample, export_debug=True)
        missing = MANIFEST_FIELDS - set(record)
        self.assertEqual(missing, set())
        self.assertEqual(record["present"], 1)
        self.assertIsInstance(record["gt_scale"], float)
        self.assertIsInstance(record["decoy_sites"], list)
        self.assertIn("diagnostics", record)
        self.assertIn("row", build_citations()["parameters"]["zoom_via_field_of_view"])

    def test_splits_are_disjoint_including_derived_realizations(self) -> None:
        groups = {name: make_records(name, 50) for name in ("p2_train", "p2_val", "p2_test", "p2_stress")}
        assert_disjoint(groups)  # must not raise
        # a deliberate derived collision must be rejected
        groups["p2_val"][0]["seed"] = groups["p2_train"][0]["seed"] ^ 0x5EED
        with self.assertRaises(AssertionError):
            assert_disjoint(groups)
        self.assertEqual(SPLIT_SEED_BASE["p2_holdout_fam"], 1_700_000)

    def test_severity_is_not_reachable_from_image_statistics(self) -> None:
        from scripts.validate_phase2 import _cv_splits, _pair_histogram_features

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict

        rng = np.random.default_rng(5)
        seeds = rng.choice(np.arange(1_100_000, 1_200_000), size=24, replace=False)
        records = []
        for i, seed in enumerate(seeds):
            severity = i % 4
            sample = generate_phase2_sample(
                int(seed), split="p2_train", severity=severity, search_supersample=1
            )
            records.append(
                {
                    "ref_image": "",  # filled in memory below
                    "search_image": "",
                    "present": int(sample.present),
                    "severity": severity,
                }
            )
            globals()["_img_%d" % i] = (sample.reference, sample.search)
        # build features directly from in-memory images
        features = np.zeros((len(records), 128), dtype=np.float64)
        for i in range(len(records)):
            ref, search = globals()["_img_%d" % i]
            for slot, data in enumerate((ref, search)):
                if data.ndim == 3:
                    data = data.mean(axis=-1)
                hist, _ = np.histogram(np.asarray(data, dtype=np.float64), bins=64, range=(0.0, 255.0))
                features[i, slot * 64 : (slot + 1) * 64] = hist / max(hist.sum(), 1.0)
        y = np.array([r["severity"] for r in records])
        clf = LogisticRegression(max_iter=2000, random_state=0)
        proba = cross_val_predict(clf, features, y, cv=_cv_splits(y), method="predict_proba")
        auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        self.assertLessEqual(auc, 0.70)


def _ks(a: list, b: list) -> float:
    from driftforge.phase2 import _ks_statistic

    return _ks_statistic(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
