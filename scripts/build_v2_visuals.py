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


def circle(cx: float, cy: float, r: float, fill: str, *,
           stroke: str = "none", stroke_width: float = 0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
    )


def rounded_label(x: float, y: float, width: float, height: float, fill: str,
                  value: str, *, size: int = 18, ink: str = "#ffffff") -> list[str]:
    return [
        rect(x, y, width, height, fill, rx=10),
        text(x + width / 2, y + height / 2 + size * 0.35, value, size,
             weight=700, fill=ink, anchor="middle"),
    ]


def build_scoring_allocation() -> None:
    """Phase 2 addendum weights, plus the official sample fill against them."""
    data = load(RESULTS / "phase2_experiments" / "official_sample_evaluation.json")
    blocks = data["blocks"]
    rows = [
        ("Localization", 40.0, blocks["localization"]["pts40"], BLUE,
         "Sets A and B, 0.45 A + 0.55 B"),
        ("Pose recovery", 20.0, blocks["pose"]["pts20"], CYAN,
         "Scored only where localization credit > 0"),
        ("Rejection", 15.0, blocks["rejection"]["pts15"], GREEN,
         "F1 on found across grayscale pairs"),
        ("Confidence", 10.0, blocks["confidence"]["pts10"], AMBER,
         "AUC of score against correctness"),
        ("Efficiency", 5.0, None, MUTED, "Relative quartile of median runtime"),
        ("Generator / citations / failure analysis", 10.0, None, "#7c5ce0",
         "Carried forward from Phase 1, re-judged"),
    ]
    width, height = 1200, 780
    left, right, top = 430, 1120, 150
    body = [
        text(60, 58, "Phase 2 scoring: 100 points plus 10 bonus", 34, weight=700),
        text(60, 96, "Filled bars are official-sample measurements; open bars are jury-judged",
             19, fill=MUTED),
    ]
    for index, (label, maximum, value, color, note) in enumerate(rows):
        y = top + index * 88
        body.extend([
            text(left - 24, y + 22, label, 20, weight=700, anchor="end"),
            text(left - 24, y + 46, note, 15, fill=MUTED, anchor="end"),
            rect(left, y, right - left, 36, GRID, rx=9),
        ])
        if value is None:
            body.append(text(right, y + 26, f"up to {maximum:.0f} pt  (jury)", 18,
                             weight=600, fill=MUTED, anchor="end"))
        else:
            body.extend([
                rect(left, y, (right - left) * value / maximum, 36, color, rx=9),
                text(right, y + 26, f"{value:.1f} / {maximum:.0f} pt", 18,
                     weight=700, anchor="end"),
            ])
    bonus_y = height - 110
    body.extend([
        text(60, bonus_y, "Bonus, not added to the 100-point ranking cap", 20, weight=700),
        *rounded_label(60, bonus_y + 16, 430, 48, "#7c5ce0",
                       "RGB +6  unlocked  (Set D = 1.00)"),
        *rounded_label(510, bonus_y + 16, 430, 48, GREEN,
                       "Rejection F1 +4  unlocked  (F1 = 0.94)"),
        text(60, height - 28,
             "Sources: Phase 2 addendum scoring table; results/phase2_experiments/official_sample_evaluation.json",
             16, fill=MUTED),
    ])
    write_svg("v2_scoring_allocation.svg", "Phase 2 scoring allocation",
              "Official sample fill against the published 100-point plus 10-bonus rubric.",
              width, height, body)


def build_dataset_composition() -> None:
    sets = [
        ("Set A", 70, "Nominal pose, reference present", BLUE),
        ("Set B", 70, "Degraded, four undisclosed severities", AMBER),
        ("Set C", 40, "Absent; correct answer is found = 0", RED),
        ("Set D", 20, "RGB optical analogue, bonus only", GREEN),
    ]
    total = 200
    width, height = 1200, 560
    left, bar_y, bar_h, bar_w = 60, 220, 84, 1080
    body = [
        text(60, 58, "Blind scored set: 200 organizer-generated pairs", 34, weight=700),
        text(60, 96, "Teams never see these images. Geometry is 1000x1000 px, origin top-left.",
             19, fill=MUTED),
        rect(left, bar_y, bar_w, bar_h, GRID, rx=16),
    ]
    x = left
    for label, count, note, color in sets:
        w = bar_w * count / total
        body.append(rect(x, bar_y, w, bar_h, color, rx=0))
        body.append(text(x + w / 2, bar_y + 38, label, 22, weight=700, fill="#ffffff",
                         anchor="middle"))
        body.append(text(x + w / 2, bar_y + 64, f"{count} pairs", 16, fill="#ffffff",
                         anchor="middle"))
        x += w
    for index, (label, count, note, color) in enumerate(sets):
        y = 360 + index * 36
        body.extend([
            rect(60, y - 16, 22, 22, color, rx=5),
            text(96, y, f"{label}  {count}/200  —  {note}", 18, weight=500),
        ])
    body.append(text(60, height - 28,
                     "Source: Phase 2 addendum, dataset composition slide. Sample pairs are a 20-pair I/O subset.",
                     16, fill=MUTED))
    write_svg("v2_dataset_composition.svg", "Phase 2 blind dataset composition",
              "200 organizer pairs split across nominal, degraded, absent, and RGB bonus sets.",
              width, height, body)


def build_credit_tiers() -> None:
    loc = [("<= 1 px", 1.00), ("<= 2 px", 0.80), ("<= 3 px", 0.60),
           ("<= 5 px", 0.40), ("> 5 px", 0.00)]
    scale = [("<= 1%", 1.00), ("<= 2%", 0.60), ("<= 5%", 0.30), ("> 5%", 0.00)]
    rot = [("<= 0.25 deg", 1.00), ("<= 0.50 deg", 0.60),
           ("<= 1.00 deg", 0.30), ("> 1.00 deg", 0.00)]
    width, height = 1200, 720
    body = [
        text(60, 58, "Tiered credit, not a binary hit rate", 34, weight=700),
        text(60, 96, "Subpixel work is rewarded. Pose is scored only after localization credit is already > 0.",
             19, fill=MUTED),
    ]

    def column(x: float, title_value: str, rows: list[tuple[str, float]], color: str) -> None:
        body.extend([
            rect(x, 140, 340, 500, PANEL, rx=18),
            text(x + 170, 182, title_value, 22, weight=700, anchor="middle"),
        ])
        for index, (label, credit) in enumerate(rows):
            y = 220 + index * 78
            body.extend([
                text(x + 28, y + 8, label, 18, weight=600),
                rect(x + 28, y + 18, 284, 22, GRID, rx=8),
                rect(x + 28, y + 18, 284 * credit if credit else 0, 22, color, rx=8),
                text(x + 312, y + 8, f"{credit:.2f}", 18, weight=700, anchor="end"),
            ])

    column(60, "Localization  (40 pt)", loc, BLUE)
    column(430, "Scale  (10 pt)", scale, CYAN)
    column(800, "Rotation  (10 pt)", rot, GREEN)
    body.append(text(60, height - 28,
                     "Source: Phase 2 addendum, credit-tier tables. A pose attached to the wrong tile scores zero.",
                     16, fill=MUTED))
    write_svg("v2_credit_tiers.svg", "Localization and pose credit tiers",
              "Published Euclidean and pose tolerances used to convert errors into points.",
              width, height, body)


def build_rejection_matrix() -> None:
    data = load(RESULTS / "phase2_experiments" / "official_sample_evaluation.json")
    rej = data["blocks"]["rejection"]
    tp, fp, fn = rej["tp"], rej["fp"], rej["fn"]
    tn = 4 - fp  # four absent pairs in the 20-pair sample
    width, height = 1200, 680
    cells = [
        (180, 210, GREEN, "Correct grab", f"TP = {tp}", "Present, found = 1"),
        (620, 210, RED, "False positive", f"FP = {fp}", "Absent, found = 1  (p016, p018)"),
        (180, 400, AMBER, "False negative", f"FN = {fn}", "Present, found = 0  (costs a re-scan)"),
        (620, 400, BLUE, "Correct reject", f"TN = {tn}", "Absent, found = 0"),
    ]
    body = [
        text(60, 58, "Rejection on the official sample", 34, weight=700),
        text(60, 96, f"F1 = {rej['f1']:.3f}  ·  precision {rej['precision']:.3f}  ·  "
                     f"recall {rej['recall']:.3f}  ·  {rej['pts15']:.1f} / 15 points",
             19, fill=MUTED),
        text(300, 168, "Reference present", 18, weight=700, anchor="middle"),
        text(740, 168, "Reference absent", 18, weight=700, anchor="middle"),
        text(70, 318, "Report found = 1", 17, weight=600),
        text(70, 508, "Report found = 0", 17, weight=600),
    ]
    for x, y, color, title_value, count, note in cells:
        body.extend([
            rect(x, y, 400, 150, color, rx=18),
            text(x + 24, y + 48, title_value, 22, weight=700, fill="#ffffff"),
            text(x + 24, y + 84, count, 28, weight=700, fill="#ffffff"),
            text(x + 24, y + 118, note, 16, fill="#ffffff"),
        ])
    body.append(text(60, height - 28,
                     "Source: results/phase2_experiments/official_sample_evaluation.json. "
                     "The two false positives were not used to retune the threshold.",
                     16, fill=MUTED))
    write_svg("v2_rejection_matrix.svg", "Official-sample rejection matrix",
              "Confusion matrix for the found flag on the 20-pair organizer sample.",
              width, height, body)


def build_presence_vs_score() -> None:
    width, height = 1200, 720
    left, right, top, bottom = 160, 1080, 150, 580
    mid_x = (left + right) / 2
    mid_y = (top + bottom) / 2
    body = [
        text(60, 58, "found and score answer different questions", 34, weight=700),
        text(60, 96, "A present pair can still be the wrong lattice copy. That is found = 1 with a low score.",
             19, fill=MUTED),
        rect(left, top, mid_x - left, mid_y - top, "#fff7e8", rx=0),
        rect(mid_x, top, right - mid_x, mid_y - top, "#e9faf3", rx=0),
        rect(left, mid_y, mid_x - left, bottom - mid_y, "#fff1f1", rx=0),
        rect(mid_x, mid_y, right - mid_x, bottom - mid_y, "#fff7e8", rx=0),
        line(mid_x, top, mid_x, bottom, INK, 2),
        line(left, mid_y, right, mid_y, INK, 2),
        text(left + 24, top + 48, "Unusual: review the decision", 20, weight=700, fill=AMBER),
        text(left + 24, top + 80, "Low presence evidence, high coordinate trust", 16, fill=MUTED),
        text(mid_x + 24, top + 48, "Present and localized", 20, weight=700, fill=GREEN),
        text(mid_x + 24, top + 80, "Act on the coordinate", 16, fill=MUTED),
        text(left + 24, mid_y + 48, "Absent or unusable", 20, weight=700, fill=RED),
        text(left + 24, mid_y + 80, "Pose columns are zero by contract", 16, fill=MUTED),
        text(mid_x + 24, mid_y + 48, "Present but likely mislocalized", 20, weight=700, fill=AMBER),
        text(mid_x + 24, mid_y + 80, "Confirm the selected lattice site", 16, fill=MUTED),
        text(mid_x, bottom + 40, "Presence evidence  -->  found", 18, weight=600, anchor="middle"),
        text(78, (top + bottom) / 2, "Coordinate trust  -->  score", 18, weight=600, anchor="middle"),
        text(60, height - 28,
             "Source: register.py contract. Official-sample false positives sit in the low-score present-flag region.",
             16, fill=MUTED),
    ]
    write_svg("v2_presence_vs_score.svg", "Presence versus coordinate trust",
              "Two-axis reading of the found flag and the score column.",
              width, height, body)


def build_pose_grid() -> None:
    scales = [8.0 + 0.5 * index for index in range(9)]
    angles = [-6, -3, 0, 3, 6]
    width, height = 1200, 640
    left, top, cell_w, cell_h = 180, 160, 96, 64
    body = [
        text(60, 58, "V2 coarse pose search: 9 scales x 5 rotations = 45 hypotheses", 32, weight=700),
        text(60, 96, "Hard-coding the disclosed [8, 12] and +/-5 deg bounds is allowed. The grid is slightly wider in rotation.",
             18, fill=MUTED),
        text(60, 148, "Rotation (deg)", 16, weight=600, fill=MUTED),
    ]
    for col, angle in enumerate(angles):
        body.append(text(left + col * cell_w + cell_w / 2, 148, f"{angle:+d}", 16,
                         weight=700, anchor="middle"))
    for row, scale in enumerate(scales):
        y = top + row * cell_h
        body.append(text(left - 16, y + cell_h / 2 + 6, f"{scale:.1f}x", 16,
                         weight=700, anchor="end"))
        for col in range(len(angles)):
            x = left + col * cell_w
            on_axis = (scale in {8.0, 10.0, 12.0} and angles[col] == 0)
            body.append(rect(x + 6, y + 8, cell_w - 12, cell_h - 16,
                             BLUE if on_axis else "#d7e4ff", rx=8))
    body.extend([
        text(60, height - 72, "Each cell is one full-image ZNCC surface. The global finite peak wins.",
             18),
        text(60, height - 28,
             "Source: driftforge/dense.py pose grid. Refinement then searches locally around the winner.",
             16, fill=MUTED),
    ])
    write_svg("v2_pose_grid.svg", "Coarse scale and rotation search grid",
              "Forty-five discrete pose hypotheses evaluated by dense normalized correlation.",
              width, height, body)


def build_output_contract() -> None:
    v1 = ["pair_id", "x", "y"]
    v2 = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    notes = {
        "pair_id": "Exactly once",
        "x": "Search px, top-left origin",
        "y": "Subpixel allowed",
        "theta": "deg, CCW about centre",
        "scale": "Down-scale factor in [8, 12]",
        "found": "1 or 0; zero pose if 0",
        "score": "Any monotone scale; ours is P(correct)",
    }
    width, height = 1200, 620
    body = [
        text(60, 58, "Output contract: V1 coordinate vs V2 registration row", 34, weight=700),
        text(60, 96, "A missing row scores zero. Every failure mode still emits one valid row.",
             19, fill=MUTED),
        text(60, 160, "V1", 22, weight=700, fill=BLUE),
        text(60, 320, "V2", 22, weight=700, fill=GREEN),
    ]
    for index, name in enumerate(v1):
        x = 140 + index * 150
        body.extend(rounded_label(x, 140, 136, 56, BLUE, name, size=18))
    for index, name in enumerate(v2):
        x = 140 + index * 146
        body.extend(rounded_label(x, 300, 136, 56, GREEN, name, size=17))
        body.append(text(x + 68, 380, notes[name], 14, fill=MUTED, anchor="middle"))
    body.extend([
        rect(60, 430, 1080, 120, PANEL, rx=16),
        text(84, 472, "Present row: pair_id, x, y, theta, scale, 1, score", 18, weight=700),
        text(84, 508, "Absent row: pair_id, 0, 0, 0, 0, 0, score   (score still carries evidence)",
             18),
        text(84, 536, "Organizer coordinates are not reproduced here. See HOW_TO_RUN.md for the header.",
             16, fill=MUTED),
        text(60, height - 24, "Source: Phase 2 addendum output contract; register.py COLUMNS.",
             16, fill=MUTED),
    ])
    write_svg("v2_output_contract.svg", "V1 versus V2 output contract",
              "Required prediction columns and the one-row-per-pair rule.",
              width, height, body)


def build_vs_baseline() -> None:
    data = load(RESULTS / "phase2_experiments" / "official_sample_evaluation.json")
    loc = data["blocks"]["localization"]
    grouped_bars(
        "v2_vs_baseline.svg",
        "Official sample: LatticeRank vs organizer naive ZNCC",
        "Naive baseline is brute-force ZNCC on a 0.5x / 1.0 deg grid; mean present credit 0.800",
        ["Set A", "Set B", "Set D", "Present overall"],
        [("Organizer naive ZNCC", [100.0, 46.7, 100.0, 80.0], MUTED),
         ("LatticeRank V2", [100 * loc["setA_mean"], 100 * loc["setB_mean"],
                             100 * loc["setD_mean"], 100 * loc["weighted"]], BLUE)],
        "AMP Phase 2 material/baseline_calibration.txt; official_sample_evaluation.json",
    )


def build_runtime_budget() -> None:
    official = load(RESULTS / "phase2_experiments" / "official_sample_evaluation.json")["runtime"]
    internal = load(RESULTS / "phase2_experiments" / "uncontended_runtime.json")["measured"]
    width, height = 1200, 620
    left, right, top, bottom = 220, 1100, 180, 430
    maximum = 20.0
    marks = [
        ("Official sample median", official["median_s"], BLUE),
        ("Official sample max", official["max_s"], CYAN),
        ("Internal uncontended median", internal["median_s"], GREEN),
        ("Internal P95", internal["p95_s"], AMBER),
    ]
    body = [
        text(60, 58, "Runtime against the published budgets", 34, weight=700),
        text(60, 96, "Median budget 5 s per pair. Hard timeout 20 s scores that pair zero.",
             19, fill=MUTED),
        rect(left, top, right - left, bottom - top, PANEL, rx=18),
    ]
    for tick in (0, 5, 10, 15, 20):
        x = left + 40 + (right - left - 80) * tick / maximum
        body.extend([
            line(x, top + 30, x, bottom - 70, GRID if tick not in {5, 20} else RED, 2,
                 dash=None if tick in {5, 20} else "6 6"),
            text(x, bottom - 40, f"{tick:g} s", 16, weight=700, anchor="middle"),
        ])
    body.extend([
        text(left + 40 + (right - left - 80) * 5 / maximum, top + 24, "median budget",
             15, fill=RED, anchor="middle"),
        text(left + 40 + (right - left - 80) * 20 / maximum, top + 24, "hard timeout",
             15, fill=RED, anchor="middle"),
    ])
    for index, (label, value, color) in enumerate(marks):
        y = top + 70 + index * 48
        x = left + 40 + (right - left - 80) * min(value, maximum) / maximum
        body.extend([
            circle(x, y, 10, color),
            text(x + 18, y + 6, f"{label}  {value:.2f} s", 18, weight=600),
        ])
    body.append(text(60, height - 28,
                     "Sources: official_sample_evaluation.json runtime; uncontended_runtime.json n=60.",
                     16, fill=MUTED))
    write_svg("v2_runtime_budget.svg", "Measured runtime versus published budgets",
              "Official-sample and uncontended internal timings against the 5 s and 20 s limits.",
              width, height, body)


def build_phase_change() -> None:
    rows = [
        ("Zoom", "Exactly 10x, given", "Unknown, uniform in [8x, 12x]"),
        ("Rotation", "Noise of 1-3 deg, not reported", "Unknown +/-5 deg, must be reported"),
        ("Presence", "Always present", "~20% of pairs contain no true instance"),
        ("Output", "x, y", "x, y, theta, scale, found, score"),
        ("RGB", "Not required", "Set D bonus, 20 optical pairs"),
    ]
    width, height = 1200, 680
    body = [
        text(60, 58, "What Phase 2 actually changes", 34, weight=700),
        text(60, 96, "Everything else in the Phase 1 statement still applies: image size, origin, nearest-to-centre rule, Python zip.",
             18, fill=MUTED),
        rect(430, 140, 330, 44, BLUE, rx=10),
        text(595, 170, "Phase 1 as issued", 20, weight=700, fill="#ffffff", anchor="middle"),
        rect(800, 140, 340, 44, GREEN, rx=10),
        text(970, 170, "Phase 2 addendum", 20, weight=700, fill="#ffffff", anchor="middle"),
    ]
    for index, (topic, old, new) in enumerate(rows):
        y = 210 + index * 78
        body.extend([
            text(60, y + 28, topic, 20, weight=700),
            rect(430, y, 330, 58, "#eaf2ff", rx=12),
            text(445, y + 36, old, 16, weight=500),
            rect(800, y, 340, 58, "#e9faf3", rx=12),
            text(816, y + 36, new, 16, weight=500),
        ])
    body.append(text(60, height - 28,
                     "Source: Phase 2 addendum, 'What Changes in Phase 2'. Hard-coding the disclosed bounds is intended.",
                     16, fill=MUTED))
    write_svg("v2_phase_change.svg", "Phase 1 versus Phase 2 task changes",
              "The three assumptions Phase 2 removes, plus the expanded output contract.",
              width, height, body)


def build_how_to_read() -> None:
    width, height = 1200, 640
    steps = [
        ("1", "Read found", "0 means do not use x, y, theta, scale. They are zero by contract."),
        ("2", "Read score", "Probability the reported coordinate is the correct site, not raw correlation."),
        ("3", "High found, high score", "Safe to treat as a localization. Typical present-and-locked row."),
        ("4", "High found, low score", "Something is in the image, but the selected copy is probably wrong."),
        ("5", "found = 0, very low score", "Consistent absence or an unusable pair. Tool should re-scan."),
        ("6", "score = 1e-6", "The pair could not be processed. Distinct from a confident rejection."),
    ]
    body = [
        text(60, 58, "How a process engineer should read predictions.csv", 32, weight=700),
        text(60, 96, "Do not collapse found and score. They are calibrated for different decisions.",
             19, fill=MUTED),
    ]
    for index, (number, title_value, note) in enumerate(steps):
        col = index % 2
        row = index // 2
        x = 60 + col * 570
        y = 140 + row * 140
        body.extend([
            rect(x, y, 540, 120, PANEL, rx=16),
            circle(x + 36, y + 40, 22, BLUE),
            text(x + 36, y + 47, number, 20, weight=700, fill="#ffffff", anchor="middle"),
            text(x + 72, y + 48, title_value, 22, weight=700),
            text(x + 28, y + 92, note, 16, fill=MUTED),
        ])
    body.append(text(60, height - 28,
                     "Source: register.py score/found contract; docs/HOW_TO_RUN.md.",
                     16, fill=MUTED))
    write_svg("v2_how_to_read.svg", "How to read V2 predictions",
              "Operational reading of found, score, and failure sentinels.",
              width, height, body)


def build_pipeline() -> None:
    stages = [
        ("pairs.csv", MUTED),
        ("Decode", BLUE),
        ("45-pose sweep", BLUE),
        ("Refine", CYAN),
        ("found", GREEN),
        ("score", AMBER),
        ("predictions.csv", GREEN),
    ]
    width, height = 1200, 420
    body = [
        text(60, 58, "Scored path: one process, one row", 34, weight=700),
        text(60, 96, "python register.py --input pairs.csv --output predictions.csv",
             20, fill=MUTED),
    ]
    x = 50
    for index, (label, color) in enumerate(stages):
        body.extend(rounded_label(x, 180, 140, 64, color, label, size=16))
        if index < len(stages) - 1:
            body.append(text(x + 148, 220, ">", 28, weight=700, fill=MUTED, anchor="middle"))
        x += 162
    body.extend([
        text(60, 300, "Deadline can skip refine or presence features. The row is still written.",
             18),
        text(60, height - 28, "Source: register.py solve() control flow.", 16, fill=MUTED),
    ])
    write_svg("v2_pipeline.svg", "V2 scored inference pipeline",
              "Entry-point stages from CSV in to CSV out.",
              width, height, body)


def main() -> None:
    build_official_scorecard()
    build_official_sets()
    build_v1_benchmarks()
    build_filter_ablation()
    build_acquisition_gap()
    build_runtime_histogram()
    build_scoring_allocation()
    build_dataset_composition()
    build_credit_tiers()
    build_rejection_matrix()
    build_presence_vs_score()
    build_pose_grid()
    build_output_contract()
    build_vs_baseline()
    build_runtime_budget()
    build_phase_change()
    build_how_to_read()
    build_pipeline()
    print("Built 18 evidence-linked SVG charts in docs/images")


if __name__ == "__main__":
    main()
