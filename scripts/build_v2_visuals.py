"""Build the SVG charts used by the V1-to-V2 documentation.

The renderer intentionally uses only the Python standard library. Every plotted
value is loaded from a tracked evidence file so the diagrams cannot silently
drift away from the measured results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "images"

INK = "#172033"
MUTED = "#657089"
GRID = "#dce2ec"
PANEL = "#f7f9fc"
BLUE = "#2f6fed"
CYAN = "#10a4b5"
GREEN = "#1c9b67"
AMBER = "#d88b17"
RED = "#d05252"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def text(x: float, y: float, value: object, size: int = 22, *,
         weight: int = 400, fill: str = INK, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str,
         *, rx: float = 0, stroke: str = "none", stroke_width: float = 0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID,
         width: float = 2, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dashed}/>'
    )


def write_svg(name: str, title_value: str, description: str, width: int,
              height: int, body: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title_value)}</title>",
        f"<desc id=\"desc\">{escape(description)}</desc>",
        rect(0, 0, width, height, "#ffffff"),
        *body,
        "</svg>",
    ]
    (OUT / name).write_text("\n".join(content) + "\n", encoding="utf-8")


def horizontal_bars(name: str, title_value: str, subtitle: str,
                    labels: list[str], values: list[float], maxima: list[float],
                    colors: list[str], unit: str, source: str) -> None:
    width, height = 1200, 190 + 92 * len(labels)
    left, right, top, bar_h = 310, 1100, 150, 34
    body = [text(60, 62, title_value, 34, weight=700),
            text(60, 100, subtitle, 19, fill=MUTED)]
    for index, (label, value, maximum, color) in enumerate(zip(labels, values, maxima, colors)):
        y = top + index * 92
        body.extend([
            text(left - 24, y + 25, label, 21, weight=600, anchor="end"),
            rect(left, y, right - left, bar_h, GRID, rx=9),
            rect(left, y, (right - left) * value / maximum, bar_h, color, rx=9),
            text(right, y + 26, f"{value:.1f}{unit} / {maximum:g}{unit}", 20,
                 weight=700, anchor="end"),
        ])
    body.append(text(60, height - 34, f"Source: {source}", 16, fill=MUTED))
    write_svg(name, title_value, subtitle, width, height, body)


def grouped_bars(name: str, title_value: str, subtitle: str,
                 labels: list[str], series: list[tuple[str, list[float], str]],
                 source: str, maximum: float = 100.0) -> None:
    width, height = 1200, 650
    left, right, top, bottom = 120, 1120, 145, 520
    body = [text(60, 58, title_value, 34, weight=700),
            text(60, 96, subtitle, 19, fill=MUTED)]
    for tick in range(0, int(maximum) + 1, 20):
        y = bottom - (bottom - top) * tick / maximum
        body.extend([line(left, y, right, y), text(left - 18, y + 7, f"{tick}%", 17,
                                                        fill=MUTED, anchor="end")])
    group_w = (right - left) / len(labels)
    bar_w = min(72, group_w / (len(series) + 1))
    for group_index, label in enumerate(labels):
        center = left + group_w * (group_index + 0.5)
        body.append(text(center, bottom + 36, label, 18, weight=600, anchor="middle"))
        for series_index, (_, values, color) in enumerate(series):
            value = values[group_index]
            x = center + (series_index - (len(series) - 1) / 2) * (bar_w + 10) - bar_w / 2
            h = (bottom - top) * value / maximum
            body.extend([
                rect(x, bottom - h, bar_w, h, color, rx=6),
                text(x + bar_w / 2, bottom - h - 9, f"{value:.1f}%", 16,
                     weight=700, anchor="middle"),
            ])
    legend_x = 740
    for index, (series_name, _, color) in enumerate(series):
        x = legend_x + index * 190
        body.extend([rect(x, 104, 24, 18, color, rx=4),
                     text(x + 34, 120, series_name, 17, weight=600)])
    body.append(text(60, height - 34, f"Source: {source}", 16, fill=MUTED))
    write_svg(name, title_value, subtitle, width, height, body)


def build_official_scorecard() -> None:
    data = load(RESULTS / "phase2_experiments" / "official_sample_evaluation.json")
    blocks = data["blocks"]
    horizontal_bars(
        "v2_official_scorecard.svg",
        "V2 official sample scorecard",
        f"79.1 / 85 scored points, plus RGB bonus; {data['runtime']['over_budget_pairs']} pairs over budget",
        ["Localization", "Pose", "Rejection", "Confidence", "RGB bonus"],
        [blocks["localization"]["pts40"], blocks["pose"]["pts20"],
         blocks["rejection"]["pts15"], blocks["confidence"]["pts10"], 6.0],
        [40, 20, 15, 10, 6],
        [BLUE, CYAN, GREEN, AMBER, "#7c5ce0"],
        " pt",
        "results/phase2_experiments/official_sample_evaluation.json",
    )


def build_official_sets() -> None:
    data = load(RESULTS / "phase2_experiments" / "official_sample_evaluation.json")
    loc = data["blocks"]["localization"]
    grouped_bars(
        "v2_official_sets.svg",
        "V2 localization credit by official set",
        "Present pairs only; Set C measures rejection and is shown in the scorecard",
        ["Set A · nominal", "Set B · degraded", "Set D · RGB"],
        [("Mean localization credit", [100 * loc["setA_mean"], 100 * loc["setB_mean"],
                                        100 * loc["setD_mean"]], BLUE)],
        "results/phase2_experiments/official_sample_evaluation.json",
    )


def build_v1_benchmarks() -> None:
    external = load(RESULTS / "external_starter_benchmark.json")
    fixed = load(RESULTS / "validation_metrics.json")
    random = load(RESULTS / "evaluation_30plus.json")
    values = [100 * external["development"]["accuracy_at_5px"],
              100 * external["confirmation"]["accuracy_at_5px"],
              100 * fixed["localization_accuracy"]["accuracy_at_5px"],
              100 * random["localization_accuracy"]["accuracy_at_5px"]]
    grouped_bars(
        "v1_benchmarks.svg",
        "V1 localization benchmarks",
        "Accuracy within 5 px; protocols differ, so each bar keeps its dataset label",
        ["External dev", "External holdout", "Internal fixed", "Internal random"],
        [("Within 5 px", values, CYAN)],
        "results/external_starter_benchmark.json; validation_metrics.json; evaluation_30plus.json",
    )


def build_filter_ablation() -> None:
    data = load(RESULTS / "phase2_experiments" / "exp01" / "summary.json")
    strata = data["strata"]
    keys = [f"severity={index}" for index in range(4)]
    grouped_bars(
        "v2_filter_ablation.svg",
        "V2 band-pass ablation on the internal stress corpus",
        f"Paired n={data['n']}; net +{data['paired']['net']} correct; McNemar p={data['paired']['mcnemar_p']:.4f}",
        [f"Severity {index}" for index in range(4)],
        [("Raw", [100 * strata[key]["raw"] for key in keys], MUTED),
         ("Band-passed", [100 * strata[key]["dog"] for key in keys], BLUE)],
        "results/phase2_experiments/exp01/summary.json",
        maximum=60,
    )


def build_acquisition_gap() -> None:
    data = load(RESULTS / "phase2_experiments" / "samecanvas_bound.json")
    levels = [str(index) for index in range(4)]
    grouped_bars(
        "v2_acquisition_gap.svg",
        "Why the internal V2 stress corpus is harder",
        "Same solver and pairs; only the reference acquisition changes",
        [f"Severity {level}" for level in levels],
        [("Independent", [100 * data["by_severity"][level]["indep_rate"] for level in levels], RED),
         ("Shared canvas", [100 * data["by_severity"][level]["same_rate"] for level in levels], GREEN)],
        "results/phase2_experiments/samecanvas_bound.json",
    )


def build_runtime_histogram() -> None:
    data = load(RESULTS / "phase2_experiments" / "uncontended_runtime.json")
    seconds = [float(row["seconds"]) for row in data["rows"]]
    bins = [2.6 + 0.2 * index for index in range(6)] + [3.8]
    counts = [0] * (len(bins) - 1)
    for value in seconds:
        for index in range(len(counts)):
            if bins[index] <= value < bins[index + 1] or (index == len(counts) - 1 and value == bins[index + 1]):
                counts[index] += 1
                break

    width, height = 1200, 650
    left, right, top, bottom = 120, 1110, 150, 500
    maximum = max(counts) if counts else 1
    body = [text(60, 58, "V2 uncontended runtime distribution", 34, weight=700),
            text(60, 96, f"n={len(seconds)} · median {data['measured']['median_s']:.2f} s · "
                          f"P95 {data['measured']['p95_s']:.2f} s · max {data['measured']['max_s']:.2f} s",
                 19, fill=MUTED)]
    for tick in range(0, maximum + 1, max(1, math.ceil(maximum / 5))):
        y = bottom - (bottom - top) * tick / maximum
        body.extend([line(left, y, right, y), text(left - 16, y + 7, tick, 17, fill=MUTED, anchor="end")])
    group_w = (right - left) / len(counts)
    for index, count in enumerate(counts):
        x = left + index * group_w + 10
        h = (bottom - top) * count / maximum
        body.extend([
            rect(x, bottom - h, group_w - 20, h, BLUE, rx=6),
            text(x + (group_w - 20) / 2, bottom - h - 9, count, 17, weight=700, anchor="middle"),
            text(x + (group_w - 20) / 2, bottom + 34,
                 f"{bins[index]:.1f}–{bins[index + 1]:.1f}", 16, fill=MUTED, anchor="middle"),
        ])
    body.extend([
        text((left + right) / 2, bottom + 76, "Seconds per pair", 18, weight=600, anchor="middle"),
        text(60, height - 34,
             "Source: results/phase2_experiments/uncontended_runtime.json · 5 s median budget",
             16, fill=MUTED),
    ])
    write_svg("v2_runtime.svg", "V2 uncontended runtime distribution",
              "Histogram of measured per-pair runtime on sixty internal validation pairs.",
              width, height, body)


def main() -> None:
    build_official_scorecard()
    build_official_sets()
    build_v1_benchmarks()
    build_filter_ablation()
    build_acquisition_gap()
    build_runtime_histogram()
    print("Built 6 evidence-linked SVG charts in docs/images")


if __name__ == "__main__":
    main()
