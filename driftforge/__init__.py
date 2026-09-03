"""DriftForge synthetic SEM generation and localization."""

from .generator import (
    Sample,
    generate_sample,
    normalize_architecture,
    normalize_profile,
)
from .model import MODEL_FEATURES, MODEL_FORMAT_VERSION, model_metadata

__all__ = [
    "MODEL_FEATURES",
    "MODEL_FORMAT_VERSION",
    "Sample",
    "generate_sample",
    "model_metadata",
    "normalize_architecture",
    "normalize_profile",
]
__version__ = "1.0.0"

