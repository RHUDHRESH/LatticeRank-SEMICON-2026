#!/usr/bin/env python3
"""Generate the reviewer figures and compact example pairs.

Every image panel is rendered from a deterministic DriftForge sample. Every
chart reads the curated, committed results under ``results/``. The script does
not invent illustrative data or call a network.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driftforge.baseline import _template_from_reference
from driftforge.generator import generate_sample

RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "docs" / "images"
DEFAULT_EXAMPLES = ROOT / "examples"

INK = "#17212b"
BLUE = "#1769aa"
GREEN = "#16835b"
ORANGE = "#d97818"
RED = "#c43c39"
GRAY = "#87929d"

EXAMPLES = {
    "dram": {
        "sample_id": "validation-000240",
        "seed": 900240,
        "architecture": "dram",
        "profile": "hard",
    },
    "finfet": {
        "sample_id": "validation-000276",
        "seed": 900276,
        "architecture": "finfet",
        "profile": "standard",
    },
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _predictions() -> list[dict[str, str]]:
    with (RESULTS / "validation_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def _prediction(sample_id: str) -> dict[str, str]:
    return next(row for row in _predictions() if row["sample_id"] == sample_id)


def _sample_from_prediction(sample_id: str):
    row = _prediction(sample_id)
    seed = int(sample_id.rsplit("-", 1)[1]) + 900000
    # Validation IDs 000200:000279 map directly to seeds 900200:900279.
    return generate_sample(
        seed=seed,
        architecture=row["architecture"],
        profile=row["profile"],
        search_supersample=2,
    )


def _save_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(
        path, format="PNG", optimize=True, compress_level=9
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save through Pillow so avoidable PNG text/time metadata is removed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": None, "Creation Time": None},
    )
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as source:
        clean = source.copy()
    clean.info.clear()
    clean.save(path, format="PNG", optimize=True, compress_level=9)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _mark_search(
    ax: plt.Axes,
    search: np.ndarray,
    gt: tuple[float, float],
    pred: tuple[float, float] | None = None,
) -> None:
    ax.imshow(search, cmap="gray", vmin=0, vmax=255)
    ax.add_patch(
        Rectangle(
            (gt[0] - 50, gt[1] - 50),
            100,
            100,
            fill=False,
            edgecolor=GREEN,
            linewidth=1.5,
        )
    )
    ax.plot(gt[0], gt[1], "+", color=GREEN, ms=11, mew=2, label="ground truth")
    if pred is not None:
        ax.plot(
            pred[0],
            pred[1],
            "x",
            color=RED,
            ms=9,
            mew=2,
            label="prediction",
        )
    ax.set_xlim(0, 999)
    ax.set_ylim(999, 0)
    ax.axis("off")


def _crop(image: np.ndarray, x: float, y: float, radius: int = 65) -> np.ndarray:
    pad = radius + 2
    padded = np.pad(image, pad, mode="reflect")
    cx, cy = int(round(x)) + pad, int(round(y)) + pad
    return padded[cy - radius : cy + radius, cx - radius : cx + radius]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).ravel()
    bv = b.astype(np.float64).ravel()
    av -= av.mean()
    bv -= bv.mean()
    den = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(av @ bv / den) if den > 1e-12 else 0.0


def _block_ncc(template: np.ndarray, patch: np.ndarray, grid: int = 5) -> np.ndarray:
    h, w = template.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    out = np.empty((grid, grid), dtype=float)
    for i in range(grid):
        for j in range(grid):
            out[i, j] = _ncc(
                template[ys[i] : ys[i + 1], xs[j] : xs[j + 1]],
                patch[ys[i] : ys[i + 1], xs[j] : xs[j + 1]],
            )
    return out


def _search_patch(image: np.ndarray, x: float, y: float, shape: tuple[int, int]):
    h, w = shape
    pad_y, pad_x = h // 2 + 2, w // 2 + 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    cx, cy = int(round(x)) + pad_x, int(round(y)) + pad_y
    y0, x0 = cy - h // 2, cx - w // 2
    return padded[y0 : y0 + h, x0 : x0 + w]


def write_examples(examples_dir: Path) -> None:
    for name, spec in EXAMPLES.items():
        sample = generate_sample(
            seed=spec["seed"],
            architecture=spec["architecture"],
            profile=spec["profile"],
            search_supersample=2,
        )
        folder = examples_dir / name
        _save_array(sample.reference, folder / "reference.png")
        _save_array(sample.search, folder / "search.png")
        ground_truth = {
            "schema_version": 1,
            "sample_id": spec["sample_id"],
            "seed": spec["seed"],
            "architecture": spec["architecture"],
            "profile": spec["profile"],
            "reference": "reference.png",
            "search": "search.png",
            "x": sample.gt_x,
            "y": sample.gt_y,
            "coordinate_convention": "x=column, y=row, origin=top-left, units=search pixels",
            "search_supersample": 2,
            "label_derivation": "Centroid of the reference-footprint mask after the exact Search geometric warp.",
            "generated_by": "python scripts/make_figures.py",
        }
        (folder / "ground_truth.json").write_text(
            json.dumps(ground_truth, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def figure_01(output: Path) -> None:
    sample = _sample_from_prediction("validation-000240")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    axes[0].imshow(sample.reference, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Reference\n1,000 nm field of view")
    axes[0].axis("off")
    _mark_search(axes[1], sample.search, (sample.gt_x, sample.gt_y))
    axes[1].set_title("Search\n10,000 nm field of view")
    axes[2].imshow(
        _crop(sample.search, sample.gt_x, sample.gt_y, 75),
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[2].set_title("Ground-truth neighbourhood\n100 × 100 px footprint")
    axes[2].axis("off")
    fig.suptitle(
        "Cross-scale localization: return the Reference centre in Search coordinates"
    )
    fig.tight_layout()
    _save_figure(fig, output / "01_localization_task.png")


def figure_02(output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0))
    for row, (name, spec) in enumerate(EXAMPLES.items()):
        sample = generate_sample(
            spec["seed"], spec["architecture"], spec["profile"], 2
        )
        axes[row, 0].imshow(sample.reference, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(
            f"{name.upper()} Reference · seed {spec['seed']}"
        )
        axes[row, 0].axis("off")
        _mark_search(axes[row, 1], sample.search, (sample.gt_x, sample.gt_y))
        axes[row, 1].set_title(
            f"{name.upper()} Search · GT ({sample.gt_x:.1f}, {sample.gt_y:.1f})"
        )
    fig.suptitle("Deterministic SEM-like pairs rendered from one world-coordinate scene")
    fig.tight_layout()
    _save_figure(fig, output / "02_generated_pairs.png")


def figure_03(output: Path) -> None:
    data = _load_json(RESULTS / "candidate_recall.json")
    sweep = data["sweep"]
    x = np.array([row["delta"] for row in sweep])
    recall = np.array([row["overall_recall_le_5px"] for row in sweep])
    count = np.array([row["median_candidates"] for row in sweep])
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(x, recall, marker="o", color=BLUE, lw=2.2, label="candidate recall ≤5 px")
    ax.scatter([0.10], [0.90], s=85, color=GREEN, zorder=3, label="shipped δ=0.10")
    ax.scatter([0.15], [0.925], s=85, color=ORANGE, zorder=3, label="diagnostic δ=0.15")
    for xi, yi, ci in zip(x, recall, count):
        ax.annotate(
            f"{yi:.1%}\nmedian {ci:g}",
            (xi, yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("adaptive score margin δ")
    ax.set_ylabel("candidate-pool recall within 5 px")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Wider pools improve coverage but do not measure localization")
    fig.tight_layout()
    _save_figure(fig, output / "03_candidate_recall.png")


def figure_04(output: Path) -> None:
    data = _load_json(RESULTS / "validation_metrics.json")
    comp = data["measured_comparisons"]
    names = [
        ("Peak features HGB", "peak_features_hgb", GRAY),
        ("Peak + structural HGB", "peak_plus_structural_hgb", BLUE),
        ("Final residual blend", "final_residual_blend", GREEN),
    ]
    keys = [
        ("accuracy_le_5px", "top-1"),
        ("top_5_recall_le_5px", "top-5"),
        ("top_10_recall_le_5px", "top-10"),
    ]
    positions = np.arange(len(keys))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (label, key, color) in enumerate(names):
        values = [comp[key][metric] for metric, _ in keys]
        bars = ax.bar(
            positions + (i - 1) * width,
            values,
            width,
            color=color,
            label=label,
        )
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=2, fontsize=7)
    ax.set_xticks(positions, [label for _, label in keys])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("true-site recall within 5 px")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Ranking improves, but remains far below candidate-pool coverage")
    fig.tight_layout()
    _save_figure(fig, output / "04_ranker_topk.png")


def figure_05(output: Path) -> None:
    errors = np.array([float(row["top1_error_px"]) for row in _predictions()])
    ordered = np.sort(errors)
    cdf = np.arange(1, len(ordered) + 1) / len(ordered)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(ordered, cdf, color=BLUE, lw=2.4)
    ax.axvline(5, color=GREEN, ls="--", lw=1.4, label="success threshold: 5 px")
    ax.axvline(25, color=RED, ls=":", lw=1.4, label="catastrophic threshold: 25 px")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlim(0, max(1000, ordered.max() * 1.05))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("final localization error (search pixels)")
    ax.set_ylabel("fraction of validation pairs")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Final error CDF: correct lattice site or a distant periodic alias")
    fig.tight_layout()
    _save_figure(fig, output / "05_error_cdf.png")


def figure_06(output: Path) -> None:
    data = _load_json(RESULTS / "validation_metrics.json")
    comp = data["measured_comparisons"]
    names = ["Peak\nfeatures", "Peak +\nstructural", "Final\nblend"]
    keys = [
        "peak_features_hgb",
        "peak_plus_structural_hgb",
        "final_residual_blend",
    ]
    good = [comp[key]["accuracy_le_5px"] for key in keys]
    bad = [comp[key]["catastrophic_rate_gt_25px"] for key in keys]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars_good = ax.bar(x - 0.18, good, 0.36, color=GREEN, label="≤5 px")
    bars_bad = ax.bar(x + 0.18, bad, 0.36, color=RED, label=">25 px")
    ax.bar_label(bars_good, labels=[f"{v:.1%}" for v in good], padding=2)
    ax.bar_label(bars_bad, labels=[f"{v:.1%}" for v in bad], padding=2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("fraction of 80 validation pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Measured ablation: residual evidence adds 1.25 points")
    fig.tight_layout()
    _save_figure(fig, output / "06_pipeline_ablation.png")


def _case_figure(sample_id: str, output: Path, title: str, filename: str) -> None:
    row = _prediction(sample_id)
    sample = _sample_from_prediction(sample_id)
    gt = (float(row["gt_x"]), float(row["gt_y"]))
    pred = (float(row["pred_x"]), float(row["pred_y"]))
    error = float(row["error_px"])
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.4))
    axes[0].imshow(sample.reference, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Reference")
    axes[0].axis("off")
    _mark_search(axes[1], sample.search, gt, pred)
    axes[1].set_title(f"Search · error {error:.2f} px")
    axes[2].imshow(_crop(sample.search, *gt), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Ground-truth site")
    axes[2].axis("off")
    axes[3].imshow(_crop(sample.search, *pred), cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Selected site")
    axes[3].axis("off")
    fig.suptitle(f"{title} · {sample_id} · {row['architecture'].upper()}")
    fig.tight_layout()
    _save_figure(fig, output / filename)


def figure_09(output: Path) -> None:
    sample_id = "validation-000256"
    row = _prediction(sample_id)
    sample = _sample_from_prediction(sample_id)
    template = _template_from_reference(sample.reference, 1.0, 0.0)
    gt = (float(row["gt_x"]), float(row["gt_y"]))
    pred = (float(row["pred_x"]), float(row["pred_y"]))
    true_patch = _search_patch(sample.search, *gt, template.shape)
    wrong_patch = _search_patch(sample.search, *pred, template.shape)
    true_blocks = _block_ncc(template, true_patch)
    wrong_blocks = _block_ncc(template, wrong_patch)
    panels = [
        (template, "Search-scale Reference"),
        (true_patch, f"True patch\nNCC {_ncc(template, true_patch):.3f}"),
        (
            wrong_patch,
            f"Selected alias\nNCC {_ncc(template, wrong_patch):.3f}",
        ),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.1))
    for ax, (image, label) in zip(axes[:3], panels):
        ax.imshow(image, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    for ax, matrix, label in [
        (axes[3], true_blocks, "Block NCC · true"),
        (axes[4], wrong_blocks, "Block NCC · alias"),
    ]:
        ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=1.0)
        ax.set_title(f"{label}\nmean {matrix.mean():.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6)
    fig.suptitle(
        "Periodic copies can look globally similar while local correspondence differs",
        y=1.08,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, output / "09_structural_comparison.png")


def figure_10(output: Path) -> None:
    metrics = _load_json(RESULTS / "validation_metrics.json")["metrics"]
    recall = _load_json(RESULTS / "candidate_recall.json")
    values = np.array(
        [
            [
                metrics["dram_accuracy_le_5px"],
                recall["shipped_pipeline"]["dram_recall_le_5px"],
                recall["diagnostic_wider_pool"]["dram_recall_le_5px"],
            ],
            [
                metrics["finfet_accuracy_le_5px"],
                recall["shipped_pipeline"]["finfet_recall_le_5px"],
                recall["diagnostic_wider_pool"]["finfet_recall_le_5px"],
            ],
        ]
    )
    labels = [
        "Final localization\nshipped δ=0.10",
        "Candidate recall\nshipped δ=0.10",
        "Candidate recall\ndiagnostic δ=0.15",
    ]
    colors = [BLUE, GREEN, ORANGE]
    x = np.arange(2)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i in range(3):
        bars = ax.bar(
            x + (i - 1) * width,
            values[:, i],
            width,
            color=colors[i],
            label=labels[i],
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.1%}" for v in values[:, i]],
            padding=2,
            fontsize=8,
        )
    ax.set_xticks(x, ["DRAM (n=39)", "FinFET (n=41)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction within 5 px")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Candidate coverage and final localization are separate metrics")
    fig.tight_layout()
    _save_figure(fig, output / "10_architecture_breakdown.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ten evidence-backed reviewer figures and example pairs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--skip-examples", action="store_true", help="only regenerate docs/images"
    )
    args = parser.parse_args()

    _style()
    if not args.skip_examples:
        write_examples(args.examples_dir)
    figure_01(args.output_dir)
    figure_02(args.output_dir)
    figure_03(args.output_dir)
    figure_04(args.output_dir)
    figure_05(args.output_dir)
    figure_06(args.output_dir)
    _case_figure(
        "validation-000240",
        args.output_dir,
        "Representative measured success",
        "07_success_example.png",
    )
    _case_figure(
        "validation-000256",
        args.output_dir,
        "Representative periodic-alias failure",
        "08_periodic_alias_failure.png",
    )
    figure_09(args.output_dir)
    figure_10(args.output_dir)
    pngs = sorted(args.output_dir.glob("*.png"))
    if len(pngs) != 10:
        raise RuntimeError(f"expected 10 figures, found {len(pngs)}")
    print(f"wrote {len(pngs)} figures to {args.output_dir}")
    if not args.skip_examples:
        print(f"wrote DRAM and FinFET examples to {args.examples_dir}")


if __name__ == "__main__":
    main()
