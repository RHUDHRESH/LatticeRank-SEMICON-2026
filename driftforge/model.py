"""Stable feature and serialized-model contract for localization."""
from __future__ import annotations

import hashlib
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .structural_descriptor import STRUCT_FEATURES

MODEL_FORMAT_VERSION = 1
MODEL_FILENAME = "hgb_r2.joblib"
MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
MODEL_RUNTIME_VERSIONS = {
    "scikit-learn": "1.9.0",
    "joblib": "1.5.3",
}

CANDIDATE_DELTA = 0.10
MAX_CANDIDATES = 8_000
POSITIVE_TOLERANCE_PX = 5.0
EQUIVALENCE_MARGIN = 0.05
RESIDUAL_WEIGHT = 1.0

BASE_FEATURES = (
    "raw",
    "midband",
    "directionality",
    "votes",
    "chan_spread",
    "chan_min",
    "raw_pct",
    "mid_pct",
    "dir_pct",
    "raw_delta",
    "mid_delta",
    "dir_delta",
    "raw_z",
    "mid_z",
    "dir_z",
    "agreement_count",
    "channel_best_count",
    "lat_phase_res",
    "lat_conf",
    "lat_agree",
    "disp_periods_x",
    "disp_periods_y",
    "parity_even",
    "local_entropy",
    "noise_est",
    "n_cand_scene",
)
MODEL_FEATURES = BASE_FEATURES + tuple(STRUCT_FEATURES)

# Candidate position is allowed as an output and for the specified final
# tie-break, but no distance or displacement from Search centre may be learned.
FORBIDDEN_FEATURE_FRAGMENTS = (
    "center_distance",
    "centre_distance",
    "distance_to_center",
    "distance_to_centre",
    "dist_center",
    "dist_centre",
    "center_dx",
    "center_dy",
    "centre_dx",
    "centre_dy",
    "dx_center",
    "dy_center",
    "dx_centre",
    "dy_centre",
)


class ModelCompatibilityError(ValueError):
    """Raised when a serialized ranker does not satisfy the public contract."""


def package_versions() -> dict[str, str]:
    """Return deterministic runtime provenance for a trained model."""
    names = ("numpy", "scipy", "scikit-learn", "joblib", "Pillow")
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def model_file_provenance(path: str | Path = MODEL_PATH) -> dict[str, str]:
    """Return a host-independent model path and content checksum."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    resolved = model_path.resolve()
    package_root = Path(__file__).resolve().parents[1]
    try:
        portable_path = resolved.relative_to(package_root).as_posix()
    except ValueError:
        portable_path = model_path.name
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": portable_path, "sha256": digest.hexdigest()}


def forbidden_features(features: Sequence[str]) -> list[str]:
    """List feature names that violate the centre-distance prohibition."""
    bad: list[str] = []
    for feature in features:
        normalized = feature.lower().replace("-", "_").replace(" ", "_")
        if any(fragment in normalized for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            bad.append(feature)
    return bad


def validate_feature_names(features: Sequence[str]) -> tuple[str, ...]:
    """Validate order, uniqueness, and the fixed production feature schema."""
    names = tuple(str(feature) for feature in features)
    if len(names) != len(set(names)):
        raise ModelCompatibilityError("model feature list contains duplicate names")
    bad = forbidden_features(names)
    if bad:
        raise ModelCompatibilityError(
            "model uses prohibited Search-centre features: " + ", ".join(bad)
        )
    if names != MODEL_FEATURES:
        raise ModelCompatibilityError(
            f"model feature schema mismatch: expected {len(MODEL_FEATURES)} "
            f"ordered features, received {len(names)}"
        )
    return names


def model_metadata() -> dict[str, Any]:
    """Metadata that is stable even for the legacy two-key shipped bundle."""
    return {
        "format_version": MODEL_FORMAT_VERSION,
        "model_filename": MODEL_FILENAME,
        "algorithm": "HistGradientBoostingClassifier candidate ranker",
        "feature_count": len(MODEL_FEATURES),
        "features": list(MODEL_FEATURES),
        "candidate_delta": CANDIDATE_DELTA,
        "max_inference_candidates": MAX_CANDIDATES,
        "positive_tolerance_px": POSITIVE_TOLERANCE_PX,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "residual_weight": RESIDUAL_WEIGHT,
        "centre_distance_feature": False,
        "coordinate_convention": "x=column, y=row, origin=top-left, Search pixels",
    }


def validate_model_bundle(bundle: Any) -> dict[str, Any]:
    """Validate and normalize a joblib payload without changing predictions."""
    if not isinstance(bundle, Mapping):
        raise ModelCompatibilityError("model file must contain a mapping")
    if "model" not in bundle or "features" not in bundle:
        raise ModelCompatibilityError("model file must contain 'model' and 'features'")

    normalized = dict(bundle)
    features = validate_feature_names(normalized["features"])
    model = normalized["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelCompatibilityError("serialized estimator does not provide predict_proba")
    n_features = getattr(model, "n_features_in_", len(features))
    if int(n_features) != len(features):
        raise ModelCompatibilityError(
            f"estimator expects {n_features} features but bundle declares {len(features)}"
        )
    classes = tuple(getattr(model, "classes_", ()))
    if classes and (len(classes) != 2 or int(classes[0]) != 0 or int(classes[1]) != 1):
        raise ModelCompatibilityError(
            f"estimator classes must be ordered binary labels [0, 1], received {classes}"
        )

    metadata = normalized.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ModelCompatibilityError("model metadata must be a mapping")
        version = metadata.get("format_version", MODEL_FORMAT_VERSION)
        if int(version) != MODEL_FORMAT_VERSION:
            raise ModelCompatibilityError(
                f"unsupported model format version {version}; expected {MODEL_FORMAT_VERSION}"
            )
    else:
        metadata = {**model_metadata(), "legacy_bundle": True}

    normalized["features"] = list(features)
    normalized["metadata"] = dict(metadata)
    return normalized


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load the ranker with actionable missing/corrupt/incompatible errors."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    versions = package_versions()
    mismatches = [
        f"{name}=={expected} (installed {versions.get(name, 'not-installed')})"
        for name, expected in MODEL_RUNTIME_VERSIONS.items()
        if versions.get(name) != expected
    ]
    if mismatches:
        raise ModelCompatibilityError(
            "model runtime version mismatch; install " + ", ".join(mismatches)
        )
    try:
        import joblib

        payload = joblib.load(model_path)
    except Exception as exc:
        raise ModelCompatibilityError(
            f"could not load model file {model_path}: {exc}"
        ) from exc
    try:
        return validate_model_bundle(payload)
    except ModelCompatibilityError as exc:
        raise ModelCompatibilityError(f"incompatible model {model_path}: {exc}") from exc


__all__ = [
    "BASE_FEATURES",
    "CANDIDATE_DELTA",
    "EQUIVALENCE_MARGIN",
    "FORBIDDEN_FEATURE_FRAGMENTS",
    "MAX_CANDIDATES",
    "MODEL_FEATURES",
    "MODEL_FILENAME",
    "MODEL_FORMAT_VERSION",
    "MODEL_PATH",
    "MODEL_RUNTIME_VERSIONS",
    "ModelCompatibilityError",
    "POSITIVE_TOLERANCE_PX",
    "RESIDUAL_WEIGHT",
    "forbidden_features",
    "load_model_bundle",
    "model_file_provenance",
    "model_metadata",
    "package_versions",
    "validate_feature_names",
    "validate_model_bundle",
]
