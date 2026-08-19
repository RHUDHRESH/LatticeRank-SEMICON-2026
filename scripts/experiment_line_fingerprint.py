"""Evaluate a first-principles line-identity channel on fixed validation pairs.

Dense array aliases share the same carrier pitch.  They do not share the same
sequence of line widths/offsets.  This experiment projects edge energy along
each lattice axis, removes the one-period carrier, and correlates the remaining
one-dimensional "serial number" in two orthogonal directions.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driftforge.baseline import _robust_contrast, _template_from_reference
from driftforge.generator import generate_sample
from driftforge.lattice import estimate_lattice
from driftforge.pipeline import compute_candidate_rows, normalize_evidence


def _cancel_periodic_1d(values: np.ndarray, period: float, axis: int) -> np.ndarray:
    shift = max(2, int(round(period)))
    # Explicit tuples keep this compatible with 1-D template profiles and 2-D
    # Search feature maps.
    delta = [0.0] * values.ndim
    delta[axis] = float(shift)
    a = ndimage.shift(values, delta, order=1, mode="nearest", prefilter=False)
    delta[axis] = -float(shift)
    b = ndimage.shift(values, delta, order=1, mode="nearest", prefilter=False)
    return (values - 0.5 * (a + b)).astype(np.float32)


def _ncc_rows(image: np.ndarray, template: np.ndarray, axis: int) -> np.ndarray:
    """NCC of a 1-D template along one axis of a 2-D feature field."""
    t = np.asarray(template, np.float32)
    t = t - float(t.mean())
    energy = float(np.sum(t * t))
    if energy < 1e-9:
        shape = list(image.shape)
        shape[axis] -= len(t) - 1
        return np.zeros(shape, np.float32)
    kernel_shape = [1, 1]
    kernel_shape[axis] = len(t)
    k = t.reshape(kernel_shape)
    ones = np.ones(kernel_shape, np.float32)
    n = float(len(t))
    sums = signal.fftconvolve(image, ones, mode="valid")
    sums2 = signal.fftconvolve(image * image, ones, mode="valid")
    local = np.maximum(sums2 - sums * sums / n, 1e-9)
    num = signal.fftconvolve(image, np.flip(k, axis=axis), mode="valid")
    return (num / np.sqrt(local * energy)).astype(np.float32)


def fingerprint_maps(reference: np.ndarray, search: np.ndarray, *, scale: float = 1.0,
                     rotation: float = 0.0) -> dict[str, np.ndarray]:
    sf = _robust_contrast(search)
    t = _template_from_reference(reference, scale, rotation)
    h, w = t.shape
    lat = estimate_lattice(search)
    px, py = float(lat.pitch_x), float(lat.pitch_y)

    # Gradient magnitude across vertical/horizontal device edges.  Averaging
    # over the orthogonal template span turns every candidate patch into two
    # stable line sequences without materializing thousands of crops.
    sx = np.abs(ndimage.sobel(ndimage.gaussian_filter(sf, 0.7), axis=1))
    sy = np.abs(ndimage.sobel(ndimage.gaussian_filter(sf, 0.7), axis=0))
    tx = np.abs(ndimage.sobel(ndimage.gaussian_filter(t, 0.7), axis=1)).mean(axis=0)
    ty = np.abs(ndimage.sobel(ndimage.gaussian_filter(t, 0.7), axis=0)).mean(axis=1)
    sx = ndimage.uniform_filter1d(sx, size=h, axis=0, mode="nearest")
    sy = ndimage.uniform_filter1d(sy, size=w, axis=1, mode="nearest")

    raw_x = _ncc_rows(sx, tx, axis=1)
    raw_y = _ncc_rows(sy, ty, axis=0)
    rx = _cancel_periodic_1d(sx, px, axis=1)
    ry = _cancel_periodic_1d(sy, py, axis=0)
    rtx = _cancel_periodic_1d(tx, px, axis=0)
    rty = _cancel_periodic_1d(ty, py, axis=0)
    res_x = _ncc_rows(rx, rtx, axis=1)
    res_y = _ncc_rows(ry, rty, axis=0)
    return {"raw_x": raw_x, "raw_y": raw_y, "res_x": res_x, "res_y": res_y,
            "half_w": (w - 1) / 2.0, "half_h": (h - 1) / 2.0}


def at(maps: dict[str, np.ndarray], key: str, x: float, y: float) -> float:
    c = int(round(x - maps["half_w"])) if key.endswith("x") else int(round(x))
    r = int(round(y)) if key.endswith("x") else int(round(y - maps["half_h"]))
    arr = maps[key]
    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
        return float(arr[r, c])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args()
    with (ROOT / "results" / "predictions.csv").open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))[:args.count]
    modes = ("raw_x", "raw_y", "raw_sum", "res_x", "res_y", "res_sum", "res_min", "hybrid")
    hits = {m: 0 for m in modes}
    recalls = 0
    for i, rec in enumerate(records, 1):
        sample = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        rows = compute_candidate_rows(sample.reference, sample.search, struct=False)
        errors = np.asarray([math.hypot(r["x"] - sample.gt_x, r["y"] - sample.gt_y) for r in rows])
        recalls += int(bool(len(errors)) and float(errors.min()) <= 5)
        oracle_scale = sample.search_acquisition.scale / sample.reference_acquisition.scale
        oracle_rotation = -(sample.search_acquisition.rotation_deg - sample.reference_acquisition.rotation_deg)
        maps = fingerprint_maps(sample.reference, sample.search, scale=oracle_scale,
                                rotation=oracle_rotation)
        vals = {k: normalize_evidence(np.asarray([at(maps, k, r["x"], r["y"]) for r in rows]))
                for k in ("raw_x", "raw_y", "res_x", "res_y")}
        scores = {
            "raw_x": vals["raw_x"],
            "raw_y": vals["raw_y"],
            "raw_sum": vals["raw_x"] + vals["raw_y"],
            "res_x": vals["res_x"],
            "res_y": vals["res_y"],
            "res_sum": vals["res_x"] + vals["res_y"],
            "res_min": np.minimum(vals["res_x"], vals["res_y"]),
            "hybrid": vals["res_x"] + vals["res_y"] + 0.15 * (vals["raw_x"] + vals["raw_y"]),
        }
        line = []
        for mode, score in scores.items():
            err = float(errors[int(np.nanargmax(score))])
            hits[mode] += int(err <= 5)
            line.append(f"{mode}={err:.1f}")
        print(f"[{i:02d}/{len(records)}] pool={errors.min():.1f} " + " ".join(line), flush=True)
    print({"count": len(records), "candidate_recall": recalls / len(records),
           **{m: n / len(records) for m, n in hits.items()}})


if __name__ == "__main__":
    main()
