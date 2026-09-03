from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from driftforge.generator import (
    generate_sample,
    normalize_architecture,
    normalize_profile,
)
from scripts import generate_dataset


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_architecture_names_are_case_insensitive() -> None:
    assert normalize_architecture("DRAM") == "dram"
    assert normalize_architecture("FinFET") == "finfet"
    parser = generate_dataset._build_parser()
    args = parser.parse_args(
        ["--architecture", "BoTh", "--count", "4", "--output-dir", "unused"]
    )
    records = generate_dataset._records_from_args(args)
    assert [record["architecture"] for record in records] == [
        "dram",
        "finfet",
        "dram",
        "finfet",
    ]
    assert normalize_profile("HARD") == "hard"
    with pytest.raises(ValueError, match="profile must be one of"):
        normalize_profile("unsupported")


def test_generator_api_is_deterministic_for_same_seed() -> None:
    first = generate_sample(410_123, "DRAM", "standard", search_supersample=1)
    second = generate_sample(410_123, "dram", "standard", search_supersample=1)
    assert first.gt_x == second.gt_x
    assert first.gt_y == second.gt_y
    assert first.reference.tobytes() == second.reference.tobytes()
    assert first.search.tobytes() == second.search.tobytes()


def test_cli_records_ground_truth_seed_provenance_and_repeats(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dataset"
    arguments = [
        "--count",
        "1",
        "--architecture",
        "FinFET",
        "--seed-start",
        "420123",
        "--search-supersample",
        "1",
        "--output-dir",
        str(output),
    ]
    assert generate_dataset.main(arguments) == 0

    expected_files = [
        output / "reference" / "custom-000000.png",
        output / "search" / "custom-000000.png",
        output / "metadata" / "custom-000000.json",
        output / "labels.csv",
        output / "ground_truth.json",
        output / "DATASET_INFO.json",
    ]
    before = {path.relative_to(output): _digest(path) for path in expected_files}

    info = json.loads((output / "DATASET_INFO.json").read_text(encoding="utf-8"))
    truth = json.loads((output / "ground_truth.json").read_text(encoding="utf-8"))
    assert info["architectures"] == {"finfet": 1}
    assert info["seed_provenance"]["selected_seeds"] == [420123]
    assert truth["samples"][0]["seed"] == 420123
    assert truth["samples"][0]["architecture"] == "finfet"
    assert {"x", "y"} <= truth["samples"][0].keys()

    assert generate_dataset.main([*arguments, "--overwrite"]) == 0
    after = {path.relative_to(output): _digest(path) for path in expected_files}
    assert after == before


def test_hide_labels_omits_seed_and_regeneration_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "public"
    assert generate_dataset.main(
        [
            "--count",
            "1",
            "--architecture",
            "DRAM",
            "--seed-start",
            "987654",
            "--search-supersample",
            "1",
            "--hide-labels",
            "--output-dir",
            str(output),
        ]
    ) == 0
    info_text = (output / "DATASET_INFO.json").read_text(encoding="utf-8")
    info = json.loads(info_text)
    assert "seed" not in info_text.casefold()
    assert "provenance" not in info_text.casefold()
    assert "selected_ids" not in info
    assert (output / "reference" / "public-000000.png").is_file()
    assert (output / "search" / "public-000000.png").is_file()
    assert not (output / "labels.csv").exists()
    assert not (output / "ground_truth.json").exists()
    assert not (output / "metadata").exists()


def test_manifest_profile_validation_is_not_bypassed() -> None:
    record = {
        "id": "bad-profile",
        "split": "custom",
        "seed": 1,
        "scene_id": "scene-1",
        "architecture": "dram",
        "profile": "not-a-profile",
    }
    with pytest.raises(ValueError, match="profile must be one of"):
        generate_dataset._validate_records([record])
