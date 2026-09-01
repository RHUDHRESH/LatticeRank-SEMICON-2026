#!/usr/bin/env python3
"""Phase 2 diagnostics: error CDFs, pose/CD error trends, score separation,
ROC/PR for the found flag, reliability, runtime, coverage, and failure exports.

Predictions CSV columns (header required):

    id,pred_x,pred_y,pred_theta,pred_scale,score,found,runtime_s

``found`` is the solver's 0/1 decision; ``score`` is its confidence in [0, 1].
Absent pairs must have found=0. Columns may be omitted; the corresponding
figures are skipped.

Usage:
    python scripts/phase2_report.py --split-dir data/phase2/p2_val \
        --predictions preds.csv --output-dir results/phase2_report \
        [--export-failures results/phase2_failures]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage, signal

from driftforge.baseline import _robust_contrast
from driftforge.pose import band_pass, build_template
from driftforge.splits import read_manifest

RUNTIME_LINES = (5.0, 20.0)


def _load_predictions(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _error_array(manifest: list[dict], predictions: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(errors, severities, ids) for present pairs with predictions."""
    errors, severities, ids = [], [], []
    for record in manifest:
        pred = predictions.get(record["id"])
        if pred is None or not record["present"] or record["gt_x"] is None:
            continue
        try:
            err = float(np.hypot(float(pred["pred_x"]) - record["gt_x"], float(pred["pred_y"]) - record["gt_y"]))
        except (KeyError, TypeError, ValueError):
            continue
        errors.append(err)
        severities.append(record["severity"])
        ids.append(record["id"])
    return np.asarray(errors), np.asarray(severities), ids


def _cdf(ax: plt.Axes, values: np.ndarray, label: str) -> None:
    if len(values) == 0:
        return
    ordered = np.sort(values)
    ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)


def figure_error_cdf(errors: np.ndarray, severities: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    _cdf(ax, errors, "overall")
    for sev in (0, 1, 2, 3):
        _cdf(ax, errors[severities == sev], f"severity {sev}")
    ax.set_xscale("log")
    ax.set_xlabel("localization error (px)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Error CDF by severity")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _trend_figure(values: np.ndarray, errors: np.ndarray, xlabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(values, errors + 1e-3, s=6, alpha=0.4, edgecolors="none")
    if len(values) > 8:
        bins = np.linspace(values.min(), values.max(), 9)
        idx = np.digitize(values, bins)
        centers, medians = [], []
        for b in range(1, len(bins)):
            sel = errors[idx == b]
            if len(sel) >= 2:
                centers.append(0.5 * (bins[b - 1] + bins[b]))
                medians.append(np.median(sel))
        ax.plot(centers, medians, "r-o", ms=4, label="binned median")
        ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error (px)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_score_histograms(manifest: list[dict], predictions: dict[str, dict], out: Path) -> None:
    correct, incorrect, absent = [], [], []
    for record in manifest:
        pred = predictions.get(record["id"])
        if pred is None or "score" not in pred:
            continue
        try:
            score = float(pred["score"])
        except ValueError:
            continue
        if not record["present"]:
            absent.append(score)
        elif int(float(pred.get("found", 0))) and "pred_x" in pred and record["gt_x"] is not None:
            err = float(np.hypot(float(pred["pred_x"]) - record["gt_x"], float(pred["pred_y"]) - record["gt_y"]))
            (correct if err <= 5.0 else incorrect).append(score)
        else:
            incorrect.append(score)
    fig, ax = plt.subplots(figsize=(6, 4))
    for values, label, color in (
        (correct, "correct", "tab:green"),
        (incorrect, "incorrect", "tab:red"),
        (absent, "absent", "tab:gray"),
    ):
        if values:
            ax.hist(values, bins=30, range=(0, 1), alpha=0.5, label=f"{label} (n={len(values)})", color=color)
    ax.set_xlabel("score")
    ax.set_ylabel("pairs")
    ax.set_title("Score distributions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_found_roc(manifest: list[dict], predictions: dict[str, dict], out: Path) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    y, scores = [], []
    for record in manifest:
        pred = predictions.get(record["id"])
        if pred is None or "score" not in pred:
            continue
        try:
            scores.append(float(pred["score"]))
            y.append(1 if record["present"] else 0)
        except ValueError:
            continue
    stats: dict = {}
    if len(set(y)) == 2:
        y_arr = np.asarray(y)
        s_arr = np.asarray(scores)
        fpr, tpr, _ = roc_curve(y_arr, s_arr)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"found ROC (AUC {np.trapezoid(tpr, fpr):.3f})")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        precision, recall, thresholds = precision_recall_curve(y_arr, s_arr)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(recall, precision)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title("found precision-recall")
        fig.tight_layout()
        fig.savefig(out.with_name(out.stem + "_pr.png"), dpi=150)
        plt.close(fig)
        stats["roc_auc"] = round(float(np.trapezoid(tpr, fpr)), 4)
    return stats


def figure_reliability(manifest: list[dict], predictions: dict[str, dict], out: Path) -> None:
    scores, truth = [], []
    for record in manifest:
        pred = predictions.get(record["id"])
        if pred is None or "score" not in pred:
            continue
        try:
            scores.append(float(pred["score"]))
            truth.append(1 if record["present"] else 0)
        except ValueError:
            continue
    if len(scores) < 20:
        return
    scores = np.asarray(scores)
    truth = np.asarray(truth)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(scores, bins) - 1
    centers, empirical = [], []
    for b in range(len(bins) - 1):
        sel = truth[idx == b]
        if len(sel) >= 3:
            centers.append(0.5 * (bins[b] + bins[b + 1]))
            empirical.append(sel.mean())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.plot(centers, empirical, "o-")
    ax.set_xlabel("predicted score")
    ax.set_ylabel("empirical P(present)")
    ax.set_title("Reliability")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_runtime(manifest: list[dict], predictions: dict[str, dict], out: Path) -> None:
    runtimes = []
    for record in manifest:
        pred = predictions.get(record["id"])
        if pred is None or "runtime_s" not in pred:
            continue
        try:
            runtimes.append(float(pred["runtime_s"]))
        except ValueError:
            continue
    if not runtimes:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(runtimes, bins=30)
    for line in RUNTIME_LINES:
        ax.axvline(line, color="r", ls="--", lw=1)
    ax.set_xlabel("runtime per pair (s)")
    ax.set_ylabel("pairs")
    ax.set_title("Runtime distribution (lines: 5 s / 20 s)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_coverage(split_dir: Path, out: Path) -> None:
    path = split_dir / "coverage_report.json"
    if not path.is_file():
        return
    coverage = json.loads(path.read_text())
    ks = coverage.get("ks_vs_claimed", {})
    labels = [f"{k}\nKS p={v.get('p', float('nan')):.3f}" for k, v in ks.items() if "p" in v]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    ax.text(0.01, 0.9, "Coverage summary", fontsize=11, weight="bold")
    rows = [
        f"present frac realized {coverage.get('present_frac_realized'):.3f} / claimed {coverage.get('present_frac_claimed'):.2f}",
        f"severities realized {coverage.get('severities_realized')}",
        f"near-edge frac {coverage.get('near_edge_frac_realized'):.3f} (claimed share {coverage.get('near_edge_share_claimed'):.2f})",
        f"decoys frac of present {coverage.get('decoys', {}).get('frac_of_present'):.3f}",
        *labels,
    ]
    for i, row in enumerate(rows):
        ax.text(0.01, 0.78 - i * 0.09, row, fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def export_failure(record: dict, split_dir: Path, predictions: dict[str, dict], out: Path, pitch: tuple[float, float] | None) -> None:
    ref = np.asarray(Image.open(split_dir / record["ref_image"]).convert("L"), dtype=np.float32) / 255.0
    search_img = Image.open(split_dir / record["search_image"])
    search = np.asarray(search_img.convert("L"), dtype=np.float32) / 255.0
    pred = predictions.get(record["id"], {})
    try:
        pred_x = float(pred.get("pred_x", "nan"))
        pred_y = float(pred.get("pred_y", "nan"))
        scale = float(pred.get("pred_scale", record["gt_scale"]))
        theta = float(pred.get("pred_theta", record["gt_theta"]))
    except (TypeError, ValueError):
        pred_x = pred_y = float("nan")
        scale = record["gt_scale"]
        theta = record["gt_theta"] or 0.0

    template = build_template(ref, scale, theta)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6))
    axes[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("reference")
    axes[0, 1].imshow(search, cmap="gray", vmin=0, vmax=1)
    if np.isfinite(pred_x):
        axes[0, 1].plot(pred_x, pred_y, "r+", ms=14, mew=2, label="pred")
    if record["gt_x"] is not None:
        axes[0, 1].plot(record["gt_x"], record["gt_y"], "c_", ms=16, mew=2, label="true")
        axes[0, 1].legend(fontsize=6)
    axes[0, 1].set_title("search: predicted (r) vs true (c)")
    # correlation surface of the template over the band-passed search
    try:
        search_f = band_pass(search)
        template_f = band_pass(template)
        template_f -= template_f.mean()
        energy = float(np.sum(template_f * template_f)) + 1e-9
        ones = np.ones(template_f.shape, dtype=np.float32)
        local = signal.fftconvolve(search_f, ones, mode="valid")
        local_sq = signal.fftconvolve(search_f * search_f, ones, mode="valid")
        local_energy = np.maximum(local_sq - local * local / template_f.size, 1e-9)
        num = signal.fftconvolve(search_f, template_f[::-1, ::-1], mode="valid")
        surface = num / np.sqrt(local_energy * energy)
        axes[1, 0].imshow(surface, cmap="viridis")
        if np.isfinite(pred_x) and np.isfinite(pred_y):
            axes[1, 0].plot(pred_x - template.shape[1] / 2, pred_y - template.shape[0] / 2, "r+", ms=12, mew=1.5)
        axes[1, 0].set_title("ZNCC surface (template at pred pose)")
    except Exception:
        axes[1, 0].text(0.4, 0.5, "surface n/a")
    # periodic residual: displacement to the nearest pitch translate of truth
    axes[1, 1].axis("off")
    if record["gt_x"] is not None and np.isfinite(pred_x) and pitch is not None:
        px, py = pitch
        res_x = (pred_x - record["gt_x"]) - round((pred_x - record["gt_x"]) / px) * px
        res_y = (pred_y - record["gt_y"]) - round((pred_y - record["gt_y"]) / py) * py
        err = float(np.hypot(pred_x - record["gt_x"], pred_y - record["gt_y"]))
        axes[1, 1].text(
            0.05, 0.6,
            f"error: {err:.2f} px\nperiodic residual: ({res_x:+.2f}, {res_y:+.2f}) px\npitch: ({px:.1f}, {py:.1f}) px\nseverity: {record['severity']}  s: {record['gt_scale']:.2f}",
            fontsize=9, family="monospace",
        )
        axes[1, 1].set_title("periodic residual")
    else:
        axes[1, 1].text(0.05, 0.6, f"absent pair or no pitch\nseverity: {record['severity']}", fontsize=9, family="monospace")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-failures", type=Path, default=None)
    parser.add_argument("--failure-limit", type=int, default=40)
    args = parser.parse_args(argv)

    manifest = read_manifest(args.split_dir / "manifest.jsonl")
    predictions = _load_predictions(args.predictions)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    errors, severities, ids = _error_array(manifest, predictions)
    summary: dict = {"pairs": len(manifest), "scored_present": len(errors)}
    if len(errors):
        summary["median_px"] = round(float(np.median(errors)), 3)
        summary["p95_px"] = round(float(np.percentile(errors, 95)), 3)
        figure_error_cdf(errors, severities, out_dir / "01_error_cdf.png")

        records_by_id = {r["id"]: r for r in manifest}
        s_vals = np.array([records_by_id[i]["gt_scale"] for i in ids])
        theta_vals = np.array([records_by_id[i]["gt_theta"] for i in ids])
        cd_vals = np.abs([records_by_id[i]["cd_bias_pct"] for i in ids])
        _trend_figure(s_vals, errors, "true zoom s", out_dir / "02_error_vs_zoom.png")
        _trend_figure(theta_vals, errors, "true theta (deg)", out_dir / "03_error_vs_theta.png")
        _trend_figure(np.asarray(cd_vals, dtype=np.float64), errors, "|CD bias| (%)", out_dir / "04_error_vs_cd.png")

    figure_score_histograms(manifest, predictions, out_dir / "05_score_histograms.png")
    summary.update(figure_found_roc(manifest, predictions, out_dir / "06_found_roc.png"))
    figure_reliability(manifest, predictions, out_dir / "07_reliability.png")
    figure_runtime(manifest, predictions, out_dir / "08_runtime.png")
    figure_coverage(args.split_dir, out_dir / "09_coverage.png")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.export_failures and len(errors):
        fail_dir = args.export_failures
        fail_dir.mkdir(parents=True, exist_ok=True)
        records_by_id = {r["id"]: r for r in manifest}
        ranked = sorted(
            (i for i in ids if records_by_id[i]["present"]),
            key=lambda i: errors[ids.index(i)],
            reverse=True,
        )[: args.failure_limit]
        for rank, pair_id in enumerate(ranked):
            record = records_by_id[pair_id]
            scene = record.get("diagnostics", {}).get("scene")
            pitch = None
            if scene:
                pitch = (scene["pitch_x_nm"] / 10.0, scene["pitch_y_nm"] / 10.0)
            export_failure(
                record, args.split_dir, predictions,
                fail_dir / f"rank{rank:03d}_{pair_id}_err{errors[ids.index(pair_id)]:.1f}px.png",
                pitch,
            )
        summary["failures_exported"] = len(ranked)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
