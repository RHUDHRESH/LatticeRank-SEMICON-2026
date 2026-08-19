"""Test local descriptor correspondence and translation consensus.

This is deliberately dependency-light experimental code: Harris interest
points, normalized intensity/gradient descriptors, mutual-neighbour matching,
and a Hough vote for the Reference-centre translation.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driftforge.baseline import _robust_contrast, _template_from_reference
from driftforge.generator import generate_sample
from driftforge.pipeline import compute_candidate_rows


def keypoints(image: np.ndarray, limit: int, border: int = 8) -> tuple[np.ndarray, np.ndarray]:
    a = _robust_contrast(image)
    gx = ndimage.sobel(ndimage.gaussian_filter(a, 0.8), axis=1)
    gy = ndimage.sobel(ndimage.gaussian_filter(a, 0.8), axis=0)
    xx = ndimage.gaussian_filter(gx * gx, 1.2)
    yy = ndimage.gaussian_filter(gy * gy, 1.2)
    xy = ndimage.gaussian_filter(gx * gy, 1.2)
    response = xx * yy - xy * xy - 0.04 * (xx + yy) ** 2
    maxima = response == ndimage.maximum_filter(response, size=5, mode="nearest")
    maxima[:border] = False; maxima[-border:] = False
    maxima[:, :border] = False; maxima[:, -border:] = False
    ys, xs = np.nonzero(maxima)
    if not len(xs):
        return np.empty((0, 2), np.float32), np.empty((0, 147), np.float32)
    order = np.argsort(response[ys, xs])[::-1][:limit]
    xs, ys = xs[order], ys[order]
    desc = []
    for x, y in zip(xs, ys):
        channels = []
        for field in (a, gx, gy):
            patch = field[y - 7:y + 8, x - 7:x + 8]
            small = ndimage.zoom(patch, 7 / 15, order=1, prefilter=False)
            channels.append(small[:7, :7].ravel())
        d = np.concatenate(channels).astype(np.float32)
        d -= d.mean()
        d /= float(np.linalg.norm(d)) + 1e-8
        desc.append(d)
    return np.column_stack((xs, ys)).astype(np.float32), np.stack(desc)


def consensus(reference: np.ndarray, search: np.ndarray, rows: list[dict]) -> np.ndarray:
    t = _template_from_reference(reference, 1.0, 0.0)
    rp, rd = keypoints(t, 120)
    sp, sd = keypoints(search, 5000)
    if not len(rp) or not len(sp):
        return np.zeros(len(rows), np.float32)
    similarity = rd @ sd.T
    # Each reference feature keeps several aliases. Reciprocal rank suppresses
    # common lattice corners while retaining rare local structures.
    k = min(20, len(sp))
    cols = np.argpartition(similarity, -k, axis=1)[:, -k:]
    pair_r = np.repeat(np.arange(len(rp)), k)
    pair_s = cols.ravel()
    pair_sim = similarity[pair_r, pair_s]
    rare = np.maximum(0.0, pair_sim - 0.45) ** 2
    translations = sp[pair_s] - rp[pair_r]
    expected = np.column_stack((
        np.asarray([r["x"] for r in rows]) - (t.shape[1] - 1) / 2,
        np.asarray([r["y"] for r in rows]) - (t.shape[0] - 1) / 2,
    ))
    scores = np.zeros(len(rows), np.float32)
    # Small chunks avoid an candidates x correspondences allocation spike.
    for lo in range(0, len(rows), 256):
        delta = expected[lo:lo + 256, None, :] - translations[None, :, :]
        distance2 = np.sum(delta * delta, axis=2)
        scores[lo:lo + 256] = np.sum(rare[None, :] * np.exp(-distance2 / 18.0), axis=1)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args()
    with (ROOT / "results" / "predictions.csv").open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))[:args.count]
    hit = recall = 0
    for i, rec in enumerate(records, 1):
        s = generate_sample(int(rec["seed"]), rec["architecture"], rec["profile"], 2)
        rows = compute_candidate_rows(s.reference, s.search, struct=False)
        errors = np.asarray([math.hypot(r["x"] - s.gt_x, r["y"] - s.gt_y) for r in rows])
        recall += int(len(errors) > 0 and errors.min() <= 5)
        score = consensus(s.reference, s.search, rows)
        err = float(errors[int(np.argmax(score))]) if len(errors) else float("inf")
        hit += int(err <= 5)
        print(f"[{i:02d}/{len(records)}] pool={errors.min():.1f} keypoint={err:.1f} max={score.max():.2f}", flush=True)
    print({"count": len(records), "candidate_recall": recall / len(records), "keypoint": hit / len(records)})


if __name__ == "__main__":
    main()
