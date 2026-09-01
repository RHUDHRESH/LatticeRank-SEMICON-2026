from __future__ import annotations

import tomllib
from pathlib import Path

import joblib
import pytest

from driftforge.model import (
    MODEL_FEATURES,
    MODEL_FORMAT_VERSION,
    MODEL_PATH,
    ModelCompatibilityError,
    forbidden_features,
    load_model_bundle,
    model_file_provenance,
    model_metadata,
    validate_model_bundle,
)


def test_shipped_model_loads_with_stable_feature_contract() -> None:
    bundle = load_model_bundle()
    assert bundle["features"] == list(MODEL_FEATURES)
    assert bundle["model"].n_features_in_ == len(MODEL_FEATURES)
    assert callable(bundle["model"].predict_proba)
    assert bundle["metadata"]["format_version"] == MODEL_FORMAT_VERSION


def test_model_metadata_is_public_and_complete() -> None:
    metadata = model_metadata()
    assert metadata["features"] == list(MODEL_FEATURES)
    assert metadata["feature_count"] == len(MODEL_FEATURES)
    assert metadata["centre_distance_feature"] is False
    assert not forbidden_features(MODEL_FEATURES)


def test_missing_and_malformed_models_fail_actionably(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model file not found"):
        load_model_bundle(tmp_path / "missing.joblib")

    malformed = tmp_path / "malformed.joblib"
    joblib.dump({"not_a_model": True}, malformed)
    with pytest.raises(ModelCompatibilityError, match="model.*features"):
        load_model_bundle(malformed)


def test_schema_mismatch_is_rejected_before_prediction() -> None:
    shipped = joblib.load(MODEL_PATH)
    with pytest.raises(ModelCompatibilityError, match="feature schema mismatch"):
        validate_model_bundle(
            {"model": shipped["model"], "features": list(MODEL_FEATURES[:-1])}
        )


def test_wheel_configuration_includes_the_shipped_ranker() -> None:
    project = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads(
        (project / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_data = configuration["tool"]["setuptools"]["package-data"]
    assert "models/*.joblib" in package_data["driftforge"]
    assert "models/*.json" in package_data["driftforge"]
    assert configuration["project"]["requires-python"] == ">=3.11"
    assert MODEL_PATH.is_file()


def test_model_provenance_is_portable_and_content_addressed() -> None:
    provenance = model_file_provenance()
    assert provenance["path"] == "driftforge/models/hgb_r2.joblib"
    assert "\\" not in provenance["path"]
    assert len(provenance["sha256"]) == 64
    int(provenance["sha256"], 16)
