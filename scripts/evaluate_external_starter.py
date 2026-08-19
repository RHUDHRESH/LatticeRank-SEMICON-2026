#!/usr/bin/env python3
"""Evaluate LatticeRank against an external Drift-Sense starter checkout.

The external generator is intentionally not vendored.  Pass a checkout that
contains ``src/pipeline.py`` and ``src/presets.py``; this script imports only
its data generator and evaluates the packaged LatticeRank inference path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from driftforge.model import MODEL_PATH, load_model_bundle
from driftforge.pipeline import (
    _add_structural_features,
    compute_candidate_rows,
    lattice_compatibility_diagnostic,
    locate_v2,
    normalize_evidence,
    rank_candidate_rows,
)
from driftforge.residual import ResidualMatcher


def _install_cv2_compatibility() -> None:
    """Install the small OpenCV subset used by the public starter generator.

    LatticeRank deliberately has no OpenCV runtime dependency.  The starter
    uses it only for drawing, Gaussian blur, 10x area reduction, and remapping;
    equivalent NumPy/SciPy implementations keep this external benchmark
    reproducible without changing the submitted environment.
    """
    try:
        __import__("cv2")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("cv2")
    module.INTER_AREA = 3
    module.INTER_LINEAR = 1
    module.BORDER_REPLICATE = 1
    module.MORPH_ELLIPSE = 2
    module.MORPH_OPEN = 2
    module.MORPH_CLOSE = 3

    def gaussian_blur(image, ksize, sigmaX, sigmaY=0):
        sigma_y = sigmaX if not sigmaY else sigmaY
        return ndimage.gaussian_filter(
            image, sigma=(sigma_y, sigmaX), mode="nearest"
        ).astype(image.dtype)

    def resize(image, size, interpolation=module.INTER_LINEAR):
        width, height = map(int, size)
        source_h, source_w = image.shape[:2]
        factor_y, factor_x = source_h // height, source_w // width
        if (
            interpolation == module.INTER_AREA
            and source_h == height * factor_y
            and source_w == width * factor_x
        ):
            reduced = image.reshape(height, factor_y, width, factor_x).mean((1, 3))
            return np.clip(np.rint(reduced), 0, 255).astype(image.dtype)
        zoom = (height / source_h, width / source_w)
        return ndimage.zoom(image, zoom, order=1, prefilter=False).astype(image.dtype)

    def remap(image, map_x, map_y, interpolation=module.INTER_LINEAR, borderMode=None):
        return ndimage.map_coordinates(
            image,
            [np.asarray(map_y), np.asarray(map_x)],
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(image.dtype)

    def circle(image, center, radius, color, thickness=-1):
        cx, cy = map(int, center)
        radius = int(radius)
        y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        view = image[y0:y1, x0:x1]
        view[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = color
        return image

    def rectangle(image, point0, point1, color, thickness=-1):
        x0, y0 = map(int, point0)
        x1, y1 = map(int, point1)
        xa, xb = max(0, min(x0, x1)), min(image.shape[1] - 1, max(x0, x1))
        ya, yb = max(0, min(y0, y1)), min(image.shape[0] - 1, max(y0, y1))
        if xa <= xb and ya <= yb:
            image[ya : yb + 1, xa : xb + 1] = color
        return image

    module.GaussianBlur = gaussian_blur
    module.resize = resize
    module.remap = remap
    module.circle = circle
    module.rectangle = rectangle
    sys.modules["cv2"] = module


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _z(values: np.ndarray) -> np.ndarray:
    normalized = normalize_evidence(values)
    return normalized if normalized is not None else np.asarray(values, dtype=float)


def _select(rows: list[dict], score: np.ndarray, shape: tuple[int, ...], margin: float) -> int:
    maximum = float(np.max(score))
    equivalent = np.flatnonzero(score >= maximum - margin)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    return int(
        min(
            equivalent,
            key=lambda index: (rows[int(index)]["x"] - cx) ** 2
            + (rows[int(index)]["y"] - cy) ** 2,
        )
    )


def _prediction(rows: list[dict], index: int, gt_x: float, gt_y: float) -> dict:
    x, y = float(rows[index]["x"]), float(rows[index]["y"])
    return {"x": x, "y": y, "error_px": math.hypot(x - gt_x, y - gt_y)}


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(args: argparse.Namespace) -> dict:
    starter = args.starter.resolve()
    pipeline_path = starter / "src" / "pipeline.py"
    presets_path = starter / "src" / "presets.py"
    if not pipeline_path.is_file() or not presets_path.is_file():
        raise FileNotFoundError(
            "starter checkout must contain src/pipeline.py and src/presets.py"
        )
    _install_cv2_compatibility()
    sys.path.insert(0, str(starter))
    from src.pipeline import GenerationParams, generate_sample
    from src.presets import PRESETS

    params = GenerationParams()
    if args.stress:
        params = GenerationParams(
            dose_search=55.0,
            shear_amplitude_px=4.5,
            drift_jitter_px=2.0,
            detector_noise_sigma_search=9.0,
            astigmatism_ratio=1.35,
            vignette_strength=0.32,
            gamma=1.18,
            barrel_distortion_k=0.025,
            charging_streak_prob=2.0,
            charging_streak_intensity=1.5,
            speckle_sigma=0.18,
            salt_pepper_prob=0.006,
        )

    rng = np.random.default_rng(args.seed)
    preset_names = tuple(PRESETS)
    bundle = None if args.residual_only else load_model_bundle(args.model)
    records: list[dict] = []
    wall_started = time.perf_counter()

    for index in range(args.count):
        preset = preset_names[int(rng.integers(0, len(preset_names)))]
        generated = generate_sample(
            preset,
            rng,
            params,
            preset_overrides=(PRESETS[preset] if args.exercise_presets else None),
        )
        reference = generated["reference_img"]
        search = generated["search_img"]
        gt_x, gt_y = float(generated["gt_x"]), float(generated["gt_y"])

        started = time.perf_counter()
        rows = compute_candidate_rows(reference, search, struct=False)
        lattice_compatibility = lattice_compatibility_diagnostic(reference, search)
        candidate_best = min(
            (math.hypot(float(row["x"]) - gt_x, float(row["y"]) - gt_y) for row in rows),
            default=float("inf"),
        )
        residual = ResidualMatcher(reference, search)
        residual_raw = np.asarray(
            [residual.score(row["x"], row["y"])["res_int_m50"] for row in rows],
            dtype=float,
        )
        residual_score = _z(residual_raw)
        raw_score = _z(np.asarray([row["raw"] for row in rows], dtype=float))
        mid_score = _z(np.asarray([row["midband"] for row in rows], dtype=float))
        direction_score = _z(
            np.asarray([row["directionality"] for row in rows], dtype=float)
        )
        methods = {
            "raw": _prediction(rows, _select(rows, raw_score, search.shape, 0.05), gt_x, gt_y),
            "midband": _prediction(rows, _select(rows, mid_score, search.shape, 0.05), gt_x, gt_y),
            "directionality": _prediction(rows, _select(rows, direction_score, search.shape, 0.05), gt_x, gt_y),
            "residual": _prediction(rows, _select(rows, residual_score, search.shape, 0.05), gt_x, gt_y),
            "residual_midband": _prediction(
                rows,
                _select(
                    rows,
                    residual_score + 0.05 * mid_score,
                    search.shape,
                    0.025,
                ),
                gt_x,
                gt_y,
            ),
            "residual_consensus": _prediction(
                rows,
                _select(
                    rows,
                    residual_score + 0.05 * raw_score + 0.05 * mid_score,
                    search.shape,
                    0.025,
                ),
                gt_x,
                gt_y,
            ),
        }
        if not args.residual_only:
            result = locate_v2(
                reference,
                search,
                model_bundle=bundle,
                model_path=args.model,
                candidate_rows=rows,
            )
            methods = {
                "packaged": {
                    "x": float(result.x),
                    "y": float(result.y),
                    "error_px": math.hypot(result.x - gt_x, result.y - gt_y),
                },
                **methods,
            }
            model_scored = result.diagnostics.get("selection_mode") == "ranked_evidence"
            if model_scored:
                _add_structural_features(reference, search, rows)
                probabilities, model_score = rank_candidate_rows(rows, bundle)
                methods["model"] = _prediction(
                    rows, _select(rows, model_score, search.shape, 0.05), gt_x, gt_y
                )
        else:
            model_scored = False
        candidate_signals = [
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "error_px": math.hypot(float(row["x"]) - gt_x, float(row["y"]) - gt_y),
                "raw_z": float(raw_score[i]),
                "midband_z": float(mid_score[i]),
                "directionality_z": float(direction_score[i]),
                "residual_z": float(residual_score[i]),
            }
            for i, row in enumerate(rows)
        ]
        if model_scored:
            for i, signal in enumerate(candidate_signals):
                signal["probability"] = float(probabilities[i])
                signal["model_z"] = float(model_score[i])
        runtime = time.perf_counter() - started
        records.append(
            {
                "index": index,
                "preset": preset,
                "architecture": str(generated["architecture"]),
                "gt_x": gt_x,
                "gt_y": gt_y,
                "candidate_count": len(rows),
                "candidate_best_error_px": candidate_best,
                "lattice_compatibility": lattice_compatibility,
                "runtime_seconds": runtime,
                "methods": methods,
                "candidate_signals": candidate_signals,
            }
        )
        progress_method = "residual_consensus" if args.residual_only else "packaged"
        hits = sum(
            row["methods"][progress_method]["error_px"] <= 5.0 for row in records
        )
        print(
            f"[{index + 1:02d}/{args.count}] {preset:14s} "
            f"error={methods[progress_method]['error_px']:7.2f}px "
            f"running={hits / len(records):.1%}",
            flush=True,
        )

    summary: dict[str, dict] = {}
    for method in records[0]["methods"]:
        errors = np.asarray([row["methods"][method]["error_px"] for row in records])
        summary[method] = {
            "accuracy_at_5px": float(np.mean(errors <= 5.0)),
            "catastrophic_rate_over_25px": float(np.mean(errors > 25.0)),
            "median_error_px": float(np.median(errors)),
            "p95_error_px": float(np.percentile(errors, 95)),
            "maximum_error_px": float(errors.max()),
        }

    return {
        "schema_version": 1,
        "benchmark": {
            "kind": "external_starter_checkout",
            "checkout": str(starter),
            "revision": _git_revision(starter),
            "pipeline_sha256": _sha256(pipeline_path),
            "presets_sha256": _sha256(presets_path),
            "seed": args.seed,
            "count": args.count,
            "stress": args.stress,
            "residual_only": args.residual_only,
            "exercise_presets": args.exercise_presets,
        },
        "model": {"path": str(args.model), "sha256": _sha256(args.model)},
        "summary": summary,
        "candidate_recall_at_5px": float(
            np.mean([row["candidate_best_error_px"] <= 5.0 for row in records])
        ),
        "wall_seconds": time.perf_counter() - wall_started,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--count", type=_positive_int, default=30)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--residual-only", action="store_true")
    parser.add_argument(
        "--exercise-presets",
        action="store_true",
        help="pass each named preset explicitly instead of using zoned defaults",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate(args)
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
