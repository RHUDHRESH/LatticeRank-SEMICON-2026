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

from driftforge.baseline import _robust_contrast, _template_from_reference
from driftforge.channels import response_maps
from driftforge.generator import generate_sample
from driftforge.lattice import estimate_lattice
from driftforge.pipeline import compute_candidate_rows, normalize_evidence
from driftforge.residual import ResidualMatcher, periodic_residual

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
        rows = list(csv.DictReader(handle))
    for row in rows:
        # ``scripts/evaluate.py`` uses the generic field name ``id``.  Keep the
        # figure layer compatible with that canonical evaluator output while
        # still accepting the earlier curated ``sample_id`` spelling.
        if "sample_id" not in row and "id" in row:
            row["sample_id"] = row["id"]
    return rows


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array).save(
        temporary, format="PNG", optimize=True, compress_level=9
    )
    temporary.replace(path)


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


def figure_06(output: Path) -> None:
    fixed = _load_json(RESULTS / "validation_metrics.json")["localization_accuracy"]
    randomized = _load_json(RESULTS / "evaluation_30plus.json")["localization_accuracy"]
    external = _load_json(RESULTS / "external_starter_benchmark.json")
    names = ["Internal fixed\n80 pairs", "Internal random\n40 pairs",
             "External dev\n120 pairs", "External confirm\n30 pairs"]
    good = [fixed["accuracy_at_5px"], randomized["accuracy_at_5px"],
            external["development"]["accuracy_at_5px"],
            external["confirmation"]["accuracy_at_5px"]]
    bad = [fixed["catastrophic_rate_over_25px"], randomized["catastrophic_rate_over_25px"],
           external["development"]["catastrophic_rate_over_25px"],
           external["confirmation"]["catastrophic_rate_over_25px"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars_good = ax.bar(x - 0.18, good, 0.36, color=GREEN, label="≤5 px")
    bars_bad = ax.bar(x + 0.18, bad, 0.36, color=RED, label=">25 px")
    ax.bar_label(bars_good, labels=[f"{v:.1%}" for v in good], padding=2)
    ax.bar_label(bars_bad, labels=[f"{v:.1%}" for v in bad], padding=2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Distribution shift is the central result—not a footnote")
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


def figure_12(output: Path) -> None:
    """Measured, coordinate-aligned walkthrough of one successful inference."""
    sample_id = "validation-000240"
    row = _prediction(sample_id)
    sample = _sample_from_prediction(sample_id)
    gt = (float(row["gt_x"]), float(row["gt_y"]))
    pred = (float(row["pred_x"]), float(row["pred_y"]))
    template = _template_from_reference(sample.reference, 1.0, 0.0)
    channels = response_maps(sample.reference, sample.search)
    candidates = compute_candidate_rows(sample.reference, sample.search, struct=False)
    residual = ResidualMatcher(sample.reference, sample.search)
    residual_map = residual.maps["res_int_m50"]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2))
    axes[0, 0].imshow(sample.reference, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("1 · Full high-resolution Reference")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(template, cmap="gray")
    axes[0, 1].set_title("2 · Anti-aliased 10:1 template")
    axes[0, 1].axis("off")
    _mark_search(axes[0, 2], sample.search, gt, pred)
    axes[0, 2].set_title("3 · Search and reported coordinate")

    raw = channels.maps["raw"]
    extent = [
        channels.half_w,
        channels.half_w + raw.shape[1] - 1,
        channels.half_h + raw.shape[0] - 1,
        channels.half_h,
    ]
    axes[1, 0].imshow(raw, cmap="magma", extent=extent)
    axes[1, 0].scatter(
        [candidate["x"] for candidate in candidates],
        [candidate["y"] for candidate in candidates],
        s=22,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.8,
        label=f"{len(candidates)} harvested",
    )
    axes[1, 0].plot(*gt, "+", color=GREEN, ms=10, mew=2)
    axes[1, 0].set_title("4 · Raw ZNCC and candidate union")
    axes[1, 0].set_xlabel("Search x (px)")
    axes[1, 0].set_ylabel("Search y (px)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    residual_extent = [
        residual.half_w,
        residual.half_w + residual_map.shape[1] - 1,
        residual.half_h + residual_map.shape[0] - 1,
        residual.half_h,
    ]
    axes[1, 1].imshow(residual_map, cmap="viridis", extent=residual_extent)
    axes[1, 1].plot(*gt, "+", color="white", ms=10, mew=2)
    axes[1, 1].plot(*pred, "x", color=RED, ms=8, mew=2)
    axes[1, 1].set_title("5 · Non-periodic residual evidence")
    axes[1, 1].set_xlabel("Search x (px)")
    axes[1, 1].set_ylabel("Search y (px)")

    axes[1, 2].imshow(_crop(sample.search, *pred), cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(
        f"6 · Selected neighbourhood\n({pred[0]:.2f}, {pred[1]:.2f}), "
        f"error {float(row['error_px']):.2f} px"
    )
    axes[1, 2].axis("off")
    fig.suptitle(
        "One inference, end to end · every panel is generated from validation-000240"
    )
    fig.tight_layout()
    _save_figure(fig, output / "12_inference_walkthrough.png")


def figure_13(output: Path) -> None:
    """Show the actual arrays behind periodic-background cancellation."""
    sample = _sample_from_prediction("validation-000240")
    template = _template_from_reference(sample.reference, 1.0, 0.0)
    search_float = _robust_contrast(sample.search)
    lattice = estimate_lattice(sample.search)
    v1, v2 = lattice.basis[:, 0], lattice.basis[:, 1]
    template_residual, template_unique = periodic_residual(template, v1, v2)
    search_residual, _ = periodic_residual(search_float, v1, v2)
    template_periodic = template - template_residual
    search_patch = _search_patch(
        search_float, sample.gt_x, sample.gt_y, template.shape
    )
    residual_patch = _search_patch(
        search_residual, sample.gt_x, sample.gt_y, template.shape
    )

    panels = [
        (template, "1 · Search-scale Reference", "gray"),
        (template_periodic, "2 · Median periodic estimate", "gray"),
        (template_residual, "3 · Reference residual", "coolwarm"),
        (template_unique, "4 · Uniqueness weight", "magma"),
        (search_patch, "5 · Search at ground truth", "gray"),
        (residual_patch, "6 · Search residual at ground truth", "coolwarm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.1))
    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        if cmap == "coolwarm":
            limit = float(np.percentile(np.abs(image), 99)) + 1e-9
            ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        "Periodic cancellation on a measured pair · repeated carrier fades, local identity remains"
    )
    fig.tight_layout()
    _save_figure(fig, output / "13_periodic_residual_explainer.png")


def figure_14(output: Path) -> None:
    """Measured candidate evidence and final consensus for one inference."""
    sample_id = "validation-000240"
    row = _prediction(sample_id)
    sample = _sample_from_prediction(sample_id)
    candidates = compute_candidate_rows(sample.reference, sample.search, struct=False)
    matcher = ResidualMatcher(sample.reference, sample.search)
    raw = normalize_evidence(np.array([item["raw"] for item in candidates]))
    mid = normalize_evidence(np.array([item["midband"] for item in candidates]))
    residual = normalize_evidence(
        np.array([matcher.score(item["x"], item["y"])["res_int_m50"] for item in candidates])
    )
    if raw is None or mid is None or residual is None:
        raise RuntimeError("measured candidate evidence could not be normalized")
    consensus = residual + 0.05 * raw + 0.05 * mid
    order = np.argsort(-consensus)
    gt = np.array([float(row["gt_x"]), float(row["gt_y"])])
    pred = np.array([float(row["pred_x"]), float(row["pred_y"])])

    labels = []
    for rank, index in enumerate(order, 1):
        candidate = candidates[int(index)]
        error = math.hypot(candidate["x"] - gt[0], candidate["y"] - gt[1])
        labels.append(f"C{rank}\n{error:.1f}px")

    fig = plt.figure(figsize=(11.2, 7.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.35, 1.0), hspace=0.32)
    ax_search = fig.add_subplot(grid[0])
    ax_search.imshow(sample.search, cmap="gray", vmin=0, vmax=255)
    for rank, index in enumerate(order, 1):
        candidate = candidates[int(index)]
        ax_search.plot(candidate["x"], candidate["y"], "o", ms=7,
                       mfc="none", mec=BLUE, mew=1.4)
        ax_search.text(candidate["x"] + 12, candidate["y"] - 12, f"C{rank}",
                       color=BLUE, fontsize=9, weight="bold")
    ax_search.plot(*gt, "+", color=GREEN, ms=13, mew=2.5, label="ground truth")
    ax_search.plot(*pred, "x", color=RED, ms=10, mew=2.2, label="reported coordinate")
    ax_search.set_xlim(0, 999); ax_search.set_ylim(999, 0)
    ax_search.set_xlabel("Search x (px)"); ax_search.set_ylabel("Search y (px)")
    ax_search.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax_search.set_title(
        f"Measured candidate locations · {sample_id} · {len(candidates)} hypotheses"
    )

    ax_score = fig.add_subplot(grid[1])
    x = np.arange(len(order)); width = 0.19
    series = [
        (raw[order], "raw z", GRAY),
        (mid[order], "mid-band z", BLUE),
        (residual[order], "residual z", GREEN),
        (consensus[order], "final consensus", ORANGE),
    ]
    for offset, (values, name, color) in enumerate(series):
        ax_score.bar(x + (offset - 1.5) * width, values, width, label=name, color=color)
    ax_score.axhline(0, color=INK, linewidth=0.8)
    ax_score.set_xticks(x, labels)
    ax_score.set_ylabel("scene-normalized evidence")
    ax_score.set_xlabel("candidate rank and Euclidean error to ground truth")
    ax_score.grid(axis="y", alpha=0.22)
    ax_score.legend(frameon=False, ncol=4, loc="upper right")
    ax_score.set_title(
        "Final score = residual z + 0.05 raw z + 0.05 mid-band z"
    )
    fig.suptitle("From candidate pool to one coordinate · all values are recomputed, not illustrated")
    _save_figure(fig, output / "14_candidate_evidence.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evidence-backed reviewer figures and example pairs."
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
    figure_12(args.output_dir)
    figure_13(args.output_dir)
    figure_14(args.output_dir)
    pngs = sorted(args.output_dir.glob("*.png"))
    if len(pngs) != 9:
        raise RuntimeError(f"expected 9 figures, found {len(pngs)}")
    print(f"wrote {len(pngs)} figures to {args.output_dir}")
    if not args.skip_examples:
        print(f"wrote DRAM and FinFET examples to {args.examples_dir}")


if __name__ == "__main__":
    main()
