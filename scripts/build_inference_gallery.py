#!/usr/bin/env python3
"""Run the scored solver on our own corpus and draw what it actually did.

Every other figure in this repository is a summary: a bar, a rate, a scorecard.
None of them show a single registration. This one runs ``register.process`` --
the same function the scored entry point calls, with the same packaged weights
and the same per-pair budget -- over a stratified slice of ``data/phase2`` and
plots the result against ground truth, pair by pair.

The corpus here is **ours**, not the organizer's. Reference and search are drawn
as independent acquisitions, which is materially harder than cutting the
reference out of the search canvas, so the hit rate on these figures is far
below the official sample's. That gap is the subject of
``docs/failure_analysis.md``; these plots are where it becomes visible.

    python scripts/build_inference_gallery.py
    python scripts/build_inference_gallery.py --count 96 --split p2_val

Writes ``docs/images/v2_inference_*.{png,svg}`` and the per-pair record to
``results/phase2_experiments/inference_gallery.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import register  # noqa: E402

DATA = ROOT / "data" / "phase2"
IMAGES = ROOT / "docs" / "images"
EVIDENCE = ROOT / "results" / "phase2_experiments" / "inference_gallery.json"

INK = "#17212b"
BLUE = "#1769aa"
GREEN = "#16835b"
ORANGE = "#d97818"
RED = "#c43c39"
GRAY = "#87929d"

#: Localization credit boundaries in pixels, from the addendum's tier table.
TIERS = (1.0, 2.0, 3.0, 5.0)
HIT_PX = 5.0


@dataclass
class Outcome:
    record: dict
    row: dict
    seconds: float
    error_px: float | None = None
    scale_pct: float | None = None
    theta_deg: float | None = None

    @property
    def pair_id(self) -> str:
        return str(self.record["id"])

    @property
    def present(self) -> bool:
        return bool(self.record.get("present", 0))

    @property
    def hit(self) -> bool:
        return self.error_px is not None and self.error_px <= HIT_PX


def load_manifest(split: str) -> list[dict]:
    path = DATA / split / "manifest.jsonl"
    if not path.is_file():
        raise SystemExit(f"no manifest at {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratify(records: list[dict], count: int) -> list[dict]:
    """Deterministic slice that keeps the corpus's severity and presence mix."""
    buckets: dict[tuple, list[dict]] = {}
    for record in sorted(records, key=lambda r: str(r["id"])):
        key = (int(record.get("severity", 0)), int(record.get("present", 0)))
        buckets.setdefault(key, []).append(record)

    total = sum(len(v) for v in buckets.values())
    picked: list[dict] = []
    for key in sorted(buckets):
        pool = buckets[key]
        share = max(1, round(count * len(pool) / total))
        step = max(1, len(pool) // share)
        picked.extend(pool[::step][:share])
    return sorted(picked, key=lambda r: str(r["id"]))[:count]


def run(records: list[dict], split: str) -> list[Outcome]:
    root = DATA / split
    register.SEARCH_ROOTS.clear()
    register.SEARCH_ROOTS.append(root.resolve())

    try:
        presence = register.PresenceModel.load()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"presence model unavailable: {exc}")
    try:
        correctness = register.CorrectnessModel.load()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"correctness model unavailable: {exc}")

    outcomes: list[Outcome] = []
    for i, record in enumerate(records, start=1):
        reference = root / record["ref_image"]
        search = root / record["search_image"]
        if not reference.is_file() or not search.is_file():
            print(f"  skip {record['id']}: image missing")
            continue
        started = time.perf_counter()
        row, reason = register.process(str(record["id"]), str(reference), str(search),
                                       presence, correctness)
        seconds = time.perf_counter() - started
        if reason:
            print(f"  {record['id']}: FAILED {reason}")
        outcome = Outcome(record=record, row=row, seconds=seconds)
        if outcome.present and row["found"]:
            outcome.error_px = float(np.hypot(row["x"] - record["gt_x"],
                                              row["y"] - record["gt_y"]))
            truth_scale = float(record["gt_scale"])
            outcome.scale_pct = abs(row["scale"] - truth_scale) / truth_scale * 100.0
            outcome.theta_deg = abs(row["theta"] - float(record["gt_theta"]))
        outcomes.append(outcome)
        flag = "abs" if not outcome.present else (
            f"{outcome.error_px:7.2f}px" if outcome.error_px is not None else "  missed")
        print(f"  [{i:>3}/{len(records)}] {record['id']}  sev{record.get('severity', 0)}  "
              f"found={row['found']}  score={row['score']:.3f}  {flag}  {seconds:5.2f}s")
    return outcomes


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def save(fig: plt.Figure, path: Path, *, dpi: int = 170) -> None:
    """SVG for charts, JPEG for anything showing SEM pixels.

    These panels are mostly sensor noise, which PNG cannot compress: the older
    walkthrough figure in this repository is 1.8 MB of it. JPEG at quality 86 is
    a fifth of the size and the difference is invisible at figure scale.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".svg":
        fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
        return
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        if path.suffix in {".jpg", ".jpeg"}:
            image.convert("RGB").save(path, format="JPEG", quality=86, optimize=True,
                                      progressive=True)
        else:
            image.convert("RGB").save(path, format="PNG", optimize=True,
                                      compress_level=9)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")


def show(ax, image: np.ndarray, factor: int = 2) -> None:
    """Draw decimated pixels but keep the axes in full-resolution coordinates.

    A 1000 x 1000 plate rendered into a 450 px panel costs megabytes and shows
    nothing extra. ``extent`` keeps every patch below in image coordinates.
    """
    height, width = image.shape[:2]
    ax.imshow(image[::factor, ::factor], cmap="gray", interpolation="nearest",
              extent=(0, width, height, 0))


def footprint(scale: float) -> float:
    """Side of the reference's footprint in search pixels."""
    return 1000.0 / max(scale, 1e-6)


#: Truth is drawn as an outer frame rather than at the true footprint. When the
#: solver is subpixel-correct the two squares coincide exactly and the green one
#: disappears under the blue one, which reads as "truth was never plotted".
TRUTH_FRAME = 1.4


def mark(ax, x: float, y: float, side: float, colour: str, *, style: str = "-",
         width: float = 1.6, cross: bool = True) -> None:
    ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side, fill=False,
                           edgecolor=colour, linewidth=width, linestyle=style))
    if cross:
        ax.plot([x], [y], marker="+", color=colour, markersize=9, markeredgewidth=1.6)


def walkthrough(outcome: Outcome, split: str, path: Path) -> None:
    """One registration, end to end, with the impostor sites it had to reject."""
    root = DATA / split
    record, row = outcome.record, outcome.row
    reference = load_gray(root / record["ref_image"])
    search = load_gray(root / record["search_image"])

    fig = plt.figure(figsize=(13.6, 4.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.45, 1.0, 1.15], wspace=0.16)

    ax = fig.add_subplot(grid[0, 0])
    show(ax, reference)
    ax.set_title("Reference\n1000 x 1000, 1 nm/px", fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 1])
    show(ax, search)
    truth = footprint(float(record["gt_scale"]))
    for dx, dy in record.get("decoy_sites") or []:
        mark(ax, float(dx), float(dy), truth, ORANGE, style=(0, (3, 2)), width=1.1,
             cross=False)
    mark(ax, float(record["gt_x"]), float(record["gt_y"]), truth * TRUTH_FRAME, GREEN,
         width=1.8, cross=False)
    mark(ax, float(row["x"]), float(row["y"]), footprint(float(row["scale"])), BLUE,
         style=(0, (5, 2)), width=1.8)
    ax.set_title(f"Search, {record.get('n_decoys', 0)} impostor sites marked\n"
                 f"green truth · blue reported · orange impostors",
                 fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 2])
    half = int(truth * 1.6)
    cx, cy = int(round(row["x"])), int(round(row["y"]))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    ax.imshow(search[y0:cy + half, x0:cx + half], cmap="gray",
              interpolation="nearest", extent=(x0, cx + half, cy + half, y0))
    mark(ax, float(record["gt_x"]), float(record["gt_y"]), truth * TRUTH_FRAME, GREEN,
         width=1.8, cross=False)
    mark(ax, float(row["x"]), float(row["y"]), footprint(float(row["scale"])), BLUE,
         style=(0, (5, 2)), width=1.8)
    ax.set_title(f"Reported site, {outcome.error_px:.2f} px from truth", fontsize=9,
                 color=INK)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(grid[0, 3])
    ax.axis("off")
    lines = [
        ("the row register.py wrote", None),
        (f"x       {row['x']:9.2f}   truth {record['gt_x']:.2f}", INK),
        (f"y       {row['y']:9.2f}   truth {record['gt_y']:.2f}", INK),
        (f"theta   {row['theta']:9.2f}   truth {record['gt_theta']:.2f}", INK),
        (f"scale   {row['scale']:9.3f}   truth {record['gt_scale']:.3f}", INK),
        (f"found   {row['found']:9d}   truth {int(record['present'])}", INK),
        (f"score   {row['score']:9.3f}", INK),
        ("", None),
        ("error against ground truth", None),
        (f"centre  {outcome.error_px:8.2f} px", GREEN if outcome.hit else RED),
        (f"scale   {outcome.scale_pct:8.2f} %", INK),
        (f"rotation{outcome.theta_deg:8.2f} deg", INK),
        (f"wall    {outcome.seconds:8.2f} s", GRAY),
        ("", None),
        (f"{record['architecture']} · {record.get('preset_family', '')}", GRAY),
        (f"severity {record.get('severity', 0)} · occlusion "
         f"{float(record.get('occlusion_frac') or 0) * 100:.1f}%", GRAY),
        (f"CD bias {float(record.get('cd_bias_pct') or 0):+.1f}%", GRAY),
    ]
    y = 0.97
    for text, colour in lines:
        if colour is None:
            ax.text(0.0, y, text, fontsize=9.5, color=INK, weight="bold",
                    family="DejaVu Sans", transform=ax.transAxes, va="top")
        else:
            ax.text(0.0, y, text, fontsize=9, color=colour, family="DejaVu Sans Mono",
                    transform=ax.transAxes, va="top")
        y -= 0.062
    save(fig, path)


def gallery(outcomes: list[Outcome], split: str, path: Path, columns: int = 4) -> None:
    """A dozen real runs: what was asked, what came back, how far off."""
    rows = (len(outcomes) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.15 * columns, 4.0 * rows))
    for ax, outcome in zip(np.ravel(axes), outcomes):
        record, row = outcome.record, outcome.row
        search = load_gray(DATA / split / record["search_image"])
        show(ax, search, factor=3)
        if outcome.present:
            truth = footprint(float(record["gt_scale"])) * TRUTH_FRAME
            mark(ax, float(record["gt_x"]), float(record["gt_y"]), truth, GREEN,
                 width=1.5, cross=False)
        if row["found"]:
            mark(ax, float(row["x"]), float(row["y"]), footprint(float(row["scale"])),
                 BLUE if outcome.hit else RED, style=(0, (4, 2)), width=1.5)
        if outcome.present:
            verdict = (f"{outcome.error_px:.2f} px" if outcome.error_px is not None
                       else "rejected a present pair")
            colour = GREEN if outcome.hit else RED
        else:
            verdict = "absent, rejected" if not row["found"] else "absent, accepted"
            colour = GREEN if not row["found"] else RED
        ax.set_title(f"{record['id'].split('-')[-1]} · {record['architecture']} · "
                     f"sev {record.get('severity', 0)}", fontsize=8.5, color=INK)
        ax.set_xlabel(f"{verdict}\nfound {row['found']} · score {row['score']:.3f}",
                      fontsize=8.5, color=colour, labelpad=6)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.ravel(axes)[len(outcomes):]:
        ax.axis("off")
    fig.suptitle("Real runs on our own corpus: green is truth, blue is a hit within "
                 "5 px, red is a miss or a wrong accept", fontsize=10.5, color=INK)
    # Two-line captions under each tile need explicit room, or they land on the
    # next row's title.
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.6)
    save(fig, path, dpi=140)


def score_vs_error(outcomes: list[Outcome], path: Path) -> None:
    """Does the score column know when the coordinate is wrong?"""
    present = [o for o in outcomes if o.present and o.error_px is not None]
    absent = [o for o in outcomes if not o.present]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                      gridspec_kw={"width_ratios": [1.55, 1]})

    errors = np.array([max(o.error_px, 0.01) for o in present])
    scores = np.array([o.row["score"] for o in present])
    hits = np.array([o.hit for o in present])
    for tier in TIERS:
        left.axvline(tier, color=GRAY, linewidth=0.8, linestyle=":")
        # x in data coordinates, y in axes coordinates: keeps the label inside the
        # frame instead of floating in the figure margin.
        left.text(tier, 0.02, f"{tier:g} px", fontsize=7.5, color=GRAY, ha="center",
                  va="bottom", transform=left.get_xaxis_transform())
    left.scatter(errors[hits], scores[hits], s=34, color=GREEN, alpha=0.85,
                 label=f"within 5 px ({hits.sum()})")
    left.scatter(errors[~hits], scores[~hits], s=34, color=RED, alpha=0.75,
                 label=f"beyond 5 px ({(~hits).sum()})")
    left.set_xscale("log")
    left.set_xlabel("centre error against ground truth, px (log)", color=INK)
    left.set_ylabel("reported score", color=INK)
    left.set_title("The score column ranks correct answers above incorrect ones",
                   fontsize=10.5, color=INK)
    left.legend(fontsize=8.5, loc="upper right", frameon=False)
    left.grid(alpha=0.18)

    groups = [
        ("correct\n(<= 5 px)", [o.row["score"] for o in present if o.hit], GREEN),
        ("wrong site\n(> 5 px)", [o.row["score"] for o in present if not o.hit], RED),
        ("absent", [o.row["score"] for o in absent], GRAY),
    ]
    data = [g[1] for g in groups if g[1]]
    labels = [g[0] for g in groups if g[1]]
    colours = [g[2] for g in groups if g[1]]
    parts = right.boxplot(data, patch_artist=True, widths=0.55,
                          medianprops={"color": INK})
    right.set_xticks(range(1, len(labels) + 1), labels)
    for patch, colour in zip(parts["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
    for i, (values, colour) in enumerate(zip(data, colours), start=1):
        jitter = np.random.default_rng(7).normal(0, 0.045, len(values))
        right.scatter(i + jitter, values, s=18, color=colour, alpha=0.8, zorder=3)
        right.text(i, 0.975, f"mean {np.mean(values):.3f}", fontsize=8, color=INK,
                   ha="center", va="top", transform=right.get_xaxis_transform())
    right.set_ylabel("reported score", color=INK)
    right.set_title("Score by outcome", fontsize=10.5, color=INK)
    right.grid(alpha=0.18, axis="y")
    top = max(max(values) for values in data)
    right.set_ylim(-0.02, top * 1.14)  # headroom for the mean labels

    fig.tight_layout()
    save(fig, path)


def presence_panel(outcomes: list[Outcome], path: Path) -> None:
    """found is a separate model, so it gets a separate plot."""
    tp = sum(1 for o in outcomes if o.present and o.row["found"])
    fn = sum(1 for o in outcomes if o.present and not o.row["found"])
    fp = sum(1 for o in outcomes if not o.present and o.row["found"])
    tn = sum(1 for o in outcomes if not o.present and not o.row["found"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.25]})
    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    left.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() or 1)
    for (i, j), value in np.ndenumerate(matrix):
        left.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=15,
                  color=INK if value < matrix.max() * 0.6 else "white", weight="bold")
    left.set_xticks([0, 1], ["found = 1", "found = 0"], color=INK)
    left.set_yticks([0, 1], ["present", "absent"], color=INK)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    left.set_title(f"Presence: F1 {f1:.3f}  ·  precision {precision:.3f}  ·  "
                   f"recall {recall:.3f}", fontsize=10, color=INK)

    by_severity: dict[int, list[Outcome]] = {}
    for outcome in outcomes:
        if outcome.present:
            by_severity.setdefault(int(outcome.record.get("severity", 0)), []).append(outcome)
    keys = sorted(by_severity)
    rate = [100.0 * sum(o.hit for o in by_severity[k]) / len(by_severity[k]) for k in keys]
    counts = [len(by_severity[k]) for k in keys]
    bars = right.bar([f"severity {k}\nn={c}" for k, c in zip(keys, counts)], rate,
                     color=[GREEN if r >= 50 else ORANGE if r >= 25 else RED for r in rate],
                     width=0.62)
    for bar, value in zip(bars, rate):
        right.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%",
                   ha="center", fontsize=9.5, color=INK)
    right.set_ylim(0, max(100, max(rate) * 1.25 if rate else 100))
    right.set_ylabel("localized within 5 px, %", color=INK)
    right.set_title("Localization by acquisition severity, our corpus",
                    fontsize=10, color=INK)
    right.grid(alpha=0.18, axis="y")

    fig.tight_layout()
    save(fig, path)


def auc(positive: list[float], negative: list[float]) -> float:
    """Rank AUC: probability a positive outranks a negative, ties at half."""
    if not positive or not negative:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def write_evidence(outcomes: list[Outcome], split: str, path: Path) -> dict:
    present = [o for o in outcomes if o.present]
    located = [o for o in present if o.error_px is not None]
    hits = [o for o in located if o.hit]
    seconds = sorted(o.seconds for o in outcomes)
    payload = {
        "experiment": "inference_gallery",
        "data": f"our generator, split {split}; independent reference and search "
                f"acquisitions. NOT organizer data and not comparable to the "
                f"official sample.",
        "solver": "register.process, packaged weights, per-pair budget as shipped",
        "pairs": len(outcomes),
        "localization": {
            "present_pairs": len(present),
            "within_5px": len(hits),
            "rate": len(hits) / len(present) if present else 0.0,
            "median_error_px": float(np.median([o.error_px for o in located]))
            if located else None,
            "by_severity": {
                str(sev): {
                    "n": sum(1 for o in present
                             if int(o.record.get("severity", 0)) == sev),
                    "within_5px": sum(1 for o in present
                                      if int(o.record.get("severity", 0)) == sev and o.hit),
                }
                for sev in sorted({int(o.record.get("severity", 0)) for o in present})
            },
        },
        "presence": {
            "tp": sum(1 for o in outcomes if o.present and o.row["found"]),
            "fn": sum(1 for o in outcomes if o.present and not o.row["found"]),
            "fp": sum(1 for o in outcomes if not o.present and o.row["found"]),
            "tn": sum(1 for o in outcomes if not o.present and not o.row["found"]),
        },
        "score_auc_vs_correctness": auc([o.row["score"] for o in located if o.hit],
                                        [o.row["score"] for o in located if not o.hit]),
        "runtime_s": {
            "median": float(np.median(seconds)),
            "max": float(max(seconds)),
        },
        "pair_rows": [
            {
                "pair_id": o.pair_id,
                "severity": int(o.record.get("severity", 0)),
                "architecture": o.record.get("architecture"),
                "present": int(o.present),
                "found": int(o.row["found"]),
                "x": round(float(o.row["x"]), 4),
                "y": round(float(o.row["y"]), 4),
                "theta": round(float(o.row["theta"]), 4),
                "scale": round(float(o.row["scale"]), 4),
                "score": round(float(o.row["score"]), 6),
                "error_px": None if o.error_px is None else round(o.error_px, 4),
                "scale_pct": None if o.scale_pct is None else round(o.scale_pct, 4),
                "theta_deg": None if o.theta_deg is None else round(o.theta_deg, 4),
                "seconds": round(o.seconds, 3),
            }
            for o in outcomes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return payload


def load_outcomes(split: str, path: Path) -> list[Outcome]:
    """Rebuild the last run from its record, so figures can be redrawn cheaply."""
    if not path.is_file():
        raise SystemExit(f"no recorded run at {path}; drop --reuse to run the solver")
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = {str(r["id"]): r for r in load_manifest(split)}
    outcomes = []
    for saved in payload["pair_rows"]:
        record = index.get(saved["pair_id"])
        if record is None:
            continue
        row = {"pair_id": saved["pair_id"], "x": saved["x"], "y": saved["y"],
               "theta": saved["theta"], "scale": saved["scale"],
               "found": saved["found"], "score": saved["score"]}
        outcomes.append(Outcome(record=record, row=row, seconds=saved["seconds"],
                                error_px=saved["error_px"],
                                scale_pct=saved["scale_pct"],
                                theta_deg=saved["theta_deg"]))
    print(f"reusing {len(outcomes)} recorded pairs from {path.relative_to(ROOT)}")
    return outcomes


def curate(outcomes: list[Outcome], want: int = 12) -> list[Outcome]:
    """A gallery that shows the wins, the characteristic miss, and the rejections."""
    hits = sorted((o for o in outcomes if o.hit),
                  key=lambda o: (-int(o.record.get("severity", 0)), o.error_px))
    misses = sorted((o for o in outcomes if o.present and not o.hit),
                    key=lambda o: -(o.error_px or 0))
    absent_ok = [o for o in outcomes if not o.present and not o.row["found"]]
    absent_bad = [o for o in outcomes if not o.present and o.row["found"]]

    picked = hits[:5] + misses[:4] + absent_ok[:2] + absent_bad[:1]
    for pool in (hits[5:], misses[4:], absent_ok[2:]):
        while len(picked) < want and pool:
            picked.append(pool.pop(0))
    return picked[:want]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="p2_val")
    parser.add_argument("--count", type=int, default=48,
                        help="pairs to run; they take a few seconds each")
    parser.add_argument("--gallery", type=int, default=12)
    parser.add_argument("--reuse", action="store_true",
                        help="redraw from the recorded run instead of solving again")
    args = parser.parse_args()

    if args.reuse:
        outcomes = load_outcomes(args.split, EVIDENCE)
        payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    else:
        records = stratify(load_manifest(args.split), args.count)
        print(f"running {len(records)} pairs from {args.split}")
        started = time.perf_counter()
        outcomes = run(records, args.split)
        print(f"solver time {time.perf_counter() - started:.1f}s\n")
        payload = write_evidence(outcomes, args.split, EVIDENCE)

    hero = max((o for o in outcomes if o.hit),
               key=lambda o: (int(o.record.get("severity", 0)), -(o.error_px or 0)),
               default=None)
    if hero is None:
        print("no pair landed within 5 px; skipping the walkthrough figure")
    else:
        print(f"walkthrough pair: {hero.pair_id} "
              f"(severity {hero.record.get('severity')}, {hero.error_px:.2f} px)")
        walkthrough(hero, args.split, IMAGES / "v2_inference_walkthrough.jpg")

    gallery(curate(outcomes, args.gallery), args.split,
            IMAGES / "v2_inference_gallery.jpg")
    score_vs_error(outcomes, IMAGES / "v2_score_vs_error.svg")
    presence_panel(outcomes, IMAGES / "v2_presence_evidence.svg")

    loc = payload["localization"]
    print(f"\nlocalized {loc['within_5px']}/{loc['present_pairs']} present pairs "
          f"within 5 px, median error "
          f"{loc['median_error_px'] and round(loc['median_error_px'], 2)} px")
    print(f"score AUC vs correctness {payload['score_auc_vs_correctness']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
