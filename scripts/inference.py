#!/usr/bin/env python3
"""Command-line entry point for the packaged localization pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

# The public CLI is intentionally single-worker. Declaring that limit also
# prevents joblib's Windows physical-core probe from emitting an irrelevant
# warning on minimal or sandboxed hosts.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.config import OUTPUT_SIZE
from driftforge.model import (
    MODEL_PATH,
    load_model_bundle,
    model_file_provenance,
)
from driftforge.pipeline import locate_v2

COLOR_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}
NUMERIC_MODES = {"I", "F", "I;16", "I;16L", "I;16B", "I;16N"}


def _numeric_to_uint8(array: np.ndarray, path: Path) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"image must decode to one grayscale plane: {path}")
    if values.dtype == np.uint8:
        return values.copy()
    work = values.astype(np.float64)
    if not np.isfinite(work).all():
        raise ValueError(f"image contains non-finite pixels: {path}")
    if np.issubdtype(values.dtype, np.unsignedinteger):
        maximum = float(np.iinfo(values.dtype).max)
        work = work * (255.0 / maximum)
    else:
        lo = float(work.min())
        hi = float(work.max())
        if 0.0 <= lo and hi <= 1.0:
            work *= 255.0
        elif not (0.0 <= lo and hi <= 255.0):
            if hi <= lo:
                work.fill(0.0)
            else:
                work = (work - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(work), 0, 255).astype(np.uint8)


def load_input_image(path: Path, label: str) -> np.ndarray:
    """Decode a supported single-frame image into the production uint8 form."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} image not found: {path}")
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"{label} image must contain exactly one frame: {path}")
            if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                raise ValueError(
                    f"{label} image must be {OUTPUT_SIZE}x{OUTPUT_SIZE} pixels; "
                    f"received {image.size[0]}x{image.size[1]}: {path}"
                )
            image.load()
            if image.mode in COLOR_MODES:
                return np.asarray(image.convert("L"), dtype=np.uint8)
            if image.mode in NUMERIC_MODES:
                return _numeric_to_uint8(np.asarray(image), path)
            raise ValueError(
                f"unsupported {label} image mode {image.mode!r}: {path}"
            )
    except UnidentifiedImageError as exc:
        raise ValueError(f"{label} is not a supported image file: {path}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatticeRank navigation-error recovery.")
    parser.add_argument("reference", type=Path, help="1000x1000 high-mag Reference PNG")
    parser.add_argument("search", type=Path, help="1000x1000 low-mag Search PNG")
    parser.add_argument("--json", action="store_true", help="print full diagnostics as JSON")
    parser.add_argument("--no-residual", action="store_true",
                        help="disable the residual evidence stage (debug only)")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="path to a compatible packaged ranker",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    reference = load_input_image(args.reference, "Reference")
    search = load_input_image(args.search, "Search")
    bundle = load_model_bundle(args.model)
    result = locate_v2(
        reference,
        search,
        use_residual=not args.no_residual,
        model_path=args.model,
        model_bundle=bundle,
    )
    if not (math.isfinite(result.x) and math.isfinite(result.y)):
        raise RuntimeError("localizer returned a non-finite coordinate")
    if not (0.0 <= result.x < search.shape[1] and 0.0 <= result.y < search.shape[0]):
        raise RuntimeError(
            f"localizer returned an out-of-bounds coordinate: ({result.x}, {result.y})"
        )
    if args.json:
        metadata = bundle["metadata"]
        payload = {
            "x": result.x,
            "y": result.y,
            "coordinate_convention": (
                "x=column, y=row, origin=top-left, units=Search pixels"
            ),
            "probability": result.probability,
            "eq_set_size": result.eq_set_size,
            "n_candidates": result.n_candidates,
            "used_residual": result.used_residual,
            "model": {
                **model_file_provenance(args.model),
                "format_version": metadata.get("format_version"),
                "feature_count": len(bundle["features"]),
            },
            **result.diagnostics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"({result.x:.2f}, {result.y:.2f})")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
