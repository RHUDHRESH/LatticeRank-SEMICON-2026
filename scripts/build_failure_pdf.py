"""Build docs/failure_analysis.pdf (max 2 pages), the judged deliverable.

Docs-build tool, **not** a runtime dependency: it needs ``pip install reportlab``,
which is deliberately absent from ``requirements.txt`` because that file must
describe the environment the scored run needs and nothing more.

The prose is inlined here rather than parsed out of ``failure_analysis.md``, so
the two can drift. Any edit to the shared claims must be made in both, and the
page count re-checked -- the addendum caps this document at two pages.
"""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

from pathlib import Path

# Script-relative: never take an output path from the working directory.
OUT = str(Path(__file__).resolve().parents[1] / "docs" / "failure_analysis.pdf")

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=13.5,
                             leading=16, spaceAfter=3),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.6,
                          leading=11.5, spaceBefore=6, spaceAfter=2.5,
                          textColor=colors.HexColor("#1a1a1a")),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=7.4,
                            leading=8.9, spaceAfter=2.5, alignment=TA_LEFT),
    "bodyb": ParagraphStyle("bodyb", fontName="Helvetica", fontSize=7.4,
                             leading=8.9, spaceAfter=2.5, alignment=TA_LEFT,
                             leftIndent=8, bulletIndent=0),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.0,
                            leading=8.4),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7.0,
                             leading=8.4),
    "cellhdr": ParagraphStyle("cellhdr", fontName="Helvetica-Bold", fontSize=7.0,
                               leading=8.4, textColor=colors.white),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
                               fontSize=6.8, leading=8.2, spaceBefore=1.5,
                               spaceAfter=5, textColor=colors.HexColor("#444444")),
}

def P(text, style="body"):
    return Paragraph(text, styles[style])

def hdr_row(cells):
    return [Paragraph(c, styles["cellhdr"]) for c in cells]

def data_row(cells, bold_cols=()):
    out = []
    for i, c in enumerate(cells):
        st = "cellb" if i in bold_cols else "cell"
        out.append(Paragraph(c, styles[st]))
    return out

def make_table(header, rows, col_widths, bold_cols=(), header_bg="#2b3a55"):
    data = [hdr_row(header)] + [data_row(r, bold_cols) for r in rows]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9a9a9a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f7")]),
    ]
    t.setStyle(TableStyle(style))
    return t

doc = SimpleDocTemplate(
    OUT, pagesize=LETTER,
    leftMargin=0.42 * inch, rightMargin=0.42 * inch,
    topMargin=0.36 * inch, bottomMargin=0.36 * inch,
    title="LatticeRank Phase 2 -- Failure Analysis",
)

story = []

story.append(P("LatticeRank Phase 2 &mdash; Failure Analysis", "title"))
story.append(P(
    "Two kinds of number appear below and they must not be confused. <b>&sect;1 is the only "
    "evaluation on organizer-generated data</b>: the 20-pair Applied Materials sample set, run "
    "once, blind, after the solver was frozen, with nothing tuned on it. Everything from &sect;3 "
    "onward is measured on <b>our own generator</b>, which we built deliberately harder than the "
    "spec &mdash; it renders the reference and the search as independent acquisitions, where the "
    "organizers cut the reference from the search canvas. Those internal figures are a "
    "robustness stress-test, not a forecast, and they read roughly 30% where the official "
    "sample reads 98%.", "body"))

story.append(P("1. Official sample: 38.8 / 40 localization, 79.1 / 85 on the scored blocks", "h2"))
story.append(P(
    "<b>results/phase2_experiments/official_sample_evaluation.json.</b> 20 pairs: 8 Set A "
    "nominal, 6 Set B degraded (severity 1&ndash;4), 4 Set C absent, 2 Set D RGB; 9 architecture "
    "presets; zoom spans 8.00&ndash;12.00 at both endpoints; rotation &minus;4.9 to +4.9&deg;. "
    "Scored with the organizers&rsquo; own rubric, including their rule that localization credit "
    "is zeroed when <i>found</i>=0. <b>Zero failures, zero missing rows, relative paths in "
    "pairs.csv resolved correctly from an unrelated working directory.</b>", "body"))
t0 = make_table(
    ["Block", "Result", "Score", "Detail"],
    [
        ["Localization", "A 0.975 &middot; B 0.967 &middot; D 1.000", "<b>38.8 / 40</b>",
         "every present pair localized; misses are 1&ndash;2 px tier slips, never a wrong site"],
        ["Pose (gated)", "mean credit 0.866", "<b>17.3 / 20</b>",
         "third refinement pass added 1.7 pts; residual misses are near-ties on the tiers"],
        ["Rejection", "TP 16, FP 2, FN 0 &rarr; F1 0.9412", "<b>14.1 / 15</b>",
         "clears the F1 &ge; 0.90 gate: <b>+4 bonus</b>"],
        ["Confidence", "AUC 0.8889 vs correctness", "8.9 / 10", "score column, not the found flag"],
        ["RGB bonus", "Set D 0.900, A&ndash;C 0.982", "<b>+6 unlocked</b>",
         "requires D &ge; 0.40 and A&ndash;C &ge; 0.50"],
        ["Runtime", "median 2.98 s, max 3.24 s", "<b>inside budget</b>",
         "budget 5 s median; hard cap 20 s never approached"],
    ],
    [0.95 * inch, 2.15 * inch, 0.95 * inch, 3.45 * inch], bold_cols=(2,))
story.append(t0)
story.append(Spacer(1, 3))
story.append(P(
    "For calibration, the organizers&rsquo; README reports their own naive ZNCC baseline at "
    "<b>0.800</b> mean credit on the same present pairs, with Set A &ldquo;too easy&rdquo; and "
    "severity 3&ndash;4 defeating it outright (p011, p012, p014 &rarr; 0). LatticeRank scores "
    "1.00 on each of those three.", "body"))

story.append(P("2. Where the remaining points are", "h2"))
story.append(P(
    "<b>Pose (4.4 pts).</b> Every shortfall is a near-miss: p005 scale 2.33% / 0.57&deg;, "
    "p007 1.07% / 0.68&deg;, p003 0.32% / 0.82&deg;, p019 1.65% / 0.60&deg;. Full credit needs "
    "&le;1% and &le;0.25&deg;; the coarse grid is 0.5 in scale and 3&deg; in rotation, and their "
    "README states plainly that &ldquo;a finer search or peak interpolation is required to earn "
    "top marks &mdash; which is the intended incentive.&rdquo; A local fine search at the "
    "already-chosen site cannot move the coordinate, so it cannot cost localization. "
    "<b>Rejection (1.3 pts).</b> The presence model reports <i>found</i>=1 on three absent pairs "
    "(p015, p016, p018). The <i>score</i> column separates present from absent perfectly on this "
    "set &mdash; every absent score (max 0.0769) is below every present score (min 0.1064). "
    "This is recorded, not acted on: choosing a threshold from it would be tuning on organizer "
    "data, which the addendum lists as a no-appeal disqualification. <b>Runtime.</b> "
    "dense_pose_search is 75.6% of the 5.66 s mean (45 full-resolution correlations); presence "
    "features 12.4%, refinement 9.4%. The Deadline guard sheds stages before the 20 s cap.",
    "body"))

story.append(P("3. Why our own corpus reads 30%, and why that is the right stress test", "h2"))
story.append(P(
    "Our generator (data/phase2/p2_val + p2_stress) scores <b>29.6%</b> &le;1 px at n=280, and "
    "we measured <i>why</i> rather than assuming. Decoys and the severity ladder &mdash; the two "
    "things it was built to stress &mdash; have <b>no effect</b> (30.0 / 31.8 / 30.0% across "
    "severity 0&ndash;1 without decoys, with decoys, and the full mix; set_a_calibration.json). "
    "What does have an effect is that we render the reference and the search as "
    "<b>independent acquisitions</b>, with separate noise realisations. Cropping the reference "
    "from the search pixels instead, on identical pairs with the identical pipeline, lifts "
    "&le;1 px from 30.0% to <b>65.0%</b> (paired McNemar 26 v 5, p = 1.9e&minus;4; "
    "samecanvas_bound.json). The organizers&rsquo; README confirms that their reference is cut "
    "from the search canvas &mdash; which is why the official sample reads 98% and ours reads "
    "30%. The corpus is not mis-built; it is a harder problem than the one being scored, and it "
    "is the reason we know the shipped preprocessing is at a local optimum rather than a lucky "
    "default.", "body"))
story.append(P(
    "That optimum was established by exhaustion, all paired on identical pairs, all end-to-end "
    "top-1 accuracy: an oracle handed the ground-truth pose still selects the true site only "
    "31.2% of the time (localization_ceiling.json), so search is not the bottleneck; cross-pose "
    "sum / votes / peakiness score 2.5 / 1.2 / 18.8% against 25.0%; dense periodic residual is "
    "indistinguishable (p = 1.00); five shortlist ranking rules are all below peak; trimmed "
    "ZNCC degrades <i>monotonically</i> as pixels are dropped, the signature of uniform "
    "independent noise rather than occlusion; a 12-configuration DoG band sweep leaves the "
    "shipped (2, 8) best and collapses to 0% above the lattice pitch, because the lattice is "
    "what lets the correlation lock a position at all; normalised-gradient-field re-ranking "
    "degrades monotonically with its weight (13.3% pure); and a nearest-to-centre tie-break "
    "loses even on the slice selected so that the convention holds (51.4% &rarr; 45.7%, "
    "centre_tiebreak_natural.json). Ten independent negatives converge on one mechanism.",
    "body"))

story.append(P("4. What worked: DoG band-pass, n=280", "h2"))
story.append(P(
    "results/phase2_experiments/exp01/summary.json, identical pose sweep and selection rule, "
    "only the ZNCC input differing: raw 18.6% &rarr; DoG <b>29.6%</b>, credit 0.189 &rarr; 0.334; "
    "paired fixed 49 / broke 18, net +31, McNemar <b>p = 0.0002</b>. The gain concentrates at "
    "severity 3 (+22.0 pp) &mdash; low-frequency charging-drift suppression, not generic "
    "contrast &mdash; which is exactly where Set B&rsquo;s 0.55 weight sits.", "body"))

story.append(P("5. Final shipped architecture", "h2"))
story.append(P(
    "Phase 1&rsquo;s periodic-aware ZNCC pipeline, <b>extended, not replaced</b>: dense "
    "full-resolution pose sweep (45 poses) over the disclosed ranges, DoG-band-passed ZNCC as "
    "the matching signal, refinement to subpixel / sub-degree credit, reported pose clamped to "
    "the disclosed [8, 12] and &plusmn;5&deg; (explicitly permitted), and <b>two separate fitted "
    "models</b>: a 17-feature HGB presence model for <i>found</i> and a 20-feature logistic "
    "correctness model for <i>score</i>. They answer different questions &mdash; <i>is the "
    "reference here</i> versus <i>is the coordinate I am reporting correct</i> &mdash; and a "
    "present-but-mislocalized pair must score low on the second while being 1 on the first.",
    "body"))

story.append(P("6. Honest limitations", "h2"))
lim_items = [
    "<b>n = 20 on the official sample.</b> Set B is six pairs; one miss moves the block by "
    "&plusmn;3.7 points. The 38.8 is a strong point estimate, not a guarantee on 200 pairs.",
    "<b>Runtime is inside budget</b>: median 2.98 s, max 3.24 s on the official set and 2.92 s on "
    "60 of our own, 0 pairs over 5 s. Three memoizations got it there, each verified byte-identical: "
    "the anti-alias depends on scale not rotation; scene_features was re-refining what the entry "
    "point had already refined; and the ZNCC denominator's windowed-variance term depends on "
    "template <i>shape</i> only, so the sweep needs 63 FFT convolutions rather than 135.",
    "<b>Two false positives on absent pairs</b> cost 0.9 rejection points. The fix is visible in "
    "our own score column and was deliberately not taken, per the tuning rule.",
    "<b>Confidence is the weakest block under severity.</b> On 40 further pairs drawn off-grid "
    "from the organizers' own generator, with all 12 presets and severity weighted to 3&ndash;4, "
    "localization holds at 35.2/40 and rejection improves to F1 0.9697, but confidence AUC falls "
    "from 0.889 to 0.775. That is where this submission is softest.",
    "<b>Pose is gated on localization</b>, so its 20 points are a multiplier on site selection, "
    "not an independent block.",
    "All internal evidence is synthetic and self-generated, and is harder than the scored task; "
    "read it as a bound on robustness, never as a prediction.",
]
for it in lim_items:
    story.append(P("&bull; " + it, "bodyb"))

story.append(P("7. Reproducibility", "h2"))
story.append(P(
    "Python 3.11, CPU-only, no network at run time (scripts/verify_offline.py, audit-hook "
    "based). Weights ship inside the package (presence_hgb.pkl, correctness_lr.pkl, "
    "hgb_r2.joblib, SHA256-pinned in their metadata files) and load script-relative. "
    "scripts/build_submission.py packages from an explicit allow-list, fails if any required "
    "file is absent or if this document cites an evidence file the zip does not contain, then "
    "extracts the archive to a scratch directory and runs the entry point inside it. A 21-clause "
    "output-contract audit passes against the built zip under 3.11. Determinism: byte-identical "
    "predictions across repeated runs; 12 generator gates in scripts/validate_phase2.py.",
    "body"))

doc.build(story)
