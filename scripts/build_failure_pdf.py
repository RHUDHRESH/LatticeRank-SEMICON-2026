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
    "All numbers below are measured on our own generator "
    "(<b>data/phase2/p2_val + p2_stress</b>), which is harder than the disclosed "
    "blind-set spec. We measured <i>why</i> rather than assuming: the 40% decoy rate "
    "and the severity ladder &mdash; the two things it was built to stress &mdash; have "
    "<b>no measurable effect</b> on localization (&sect;1b). What does is that we render "
    "the reference and search as independent acquisitions, worth <b>35 points of "
    "&le;1&nbsp;px rate</b>. Diagnostic numbers, not predictions for Set A.",
    "body"))

story.append(P("1. Diagnosis: localization error is binary; selection is the bottleneck", "h2"))
story.append(P(
    "At n=180 the pairs landing within 1, 2, 3 and 5&nbsp;px are the <b>same 33 pairs</b> "
    "&mdash; none fell between 1 and 5&nbsp;px. A correct site lands sub-pixel; a wrong one "
    "misses by hundreds. <b>Mean tiered credit therefore equals site-selection accuracy</b>; "
    "subpixel refinement cannot move this score block.", "body"))

t1 = make_table(
    ["Stage", "Rate", "Note"],
    [
        ["Detection (true site is a local max)", "~99&ndash;100%", "not the problem"],
        ["Pool (true site survives harvest, GT pose)", "90&ndash;95%", "not the problem"],
        ["<b>Selection</b> (ranker picks it)", "<b>15&ndash;43%</b>", "<b>the bottleneck</b>"],
    ],
    [2.55 * inch, 0.85 * inch, 1.95 * inch],
)
story.append(t1)
story.append(P(
    "Evidence: results/gt_pose_ceiling.json (n=316); PHASE2_FINDINGS.md &sect;3, &sect;5.",
    "caption"))

story.append(P(
    "Pool-high, selection-poor is an <i>identity-evidence</i> gap, not recall/cap/NMS. "
    "This sharpens Phase 1's own flagged weakness &mdash; internal selection at 48.75% "
    "against a 90% pool (README.md) &mdash; Phase 2 shows the same shape persisted under "
    "unknown pose: pool recall held at 90&ndash;95% while selection fell to 15&ndash;43%. "
    "Root cause, measured directly on 67,370 held-out candidates: the current "
    "lattice-relative features (disp_periods_x/y, lat_phase_res, parity_even) score "
    "<b>AUC 0.38&ndash;0.62</b> against the true-site label &mdash; near chance "
    "(results/phys_eval_full.json). Seven hand-designed selectors and two retrained "
    "rankers all land in the same 0.18&ndash;0.28 band for the same reason: re-weighting "
    "existing evidence cannot fix a feature set that does not carry the signal.", "body"))

story.append(P("1b. Why our corpus is hard &mdash; measured, not assumed", "h2"))
story.append(P(
    "Set A is specified as &ldquo;noise comparable to the Phase 1 sample prompt&rdquo; and "
    "decoys are never mentioned, so the worry was that we had made the task artificially "
    "hard. Three arms of the shipped pipeline, no code changes: severity 0&ndash;1 with "
    "<b>no decoys</b> (n=180) gives <b>30.0%</b> &le;1&nbsp;px [23.8, 37.1]; severity "
    "0&ndash;1 <b>with decoys</b> (n=110) gives <b>31.8%</b> [23.9, 41.0]; the full "
    "severity mix (n=110) gives <b>30.0%</b> [22.2, 39.1]. Indistinguishable &mdash; "
    "<b>neither decoys nor severity move localization</b>, so Set A conditions should not "
    "be expected to rescue the block.", "body"))
story.append(P(
    "What does move it is a generator design choice. We render the reference and the "
    "search as <b>independent acquisitions</b> of one latent scene &mdash; separate RNG "
    "streams, so the noise realizations differ. Generators that instead crop the reference "
    "from the rendered search pixels leave the noise pattern as a unique fingerprint of "
    "the true site. Paired, n=60, identical pairs, only the reference construction "
    "differing:", "body"))
story.append(make_table(
    ["arm", "&le;1 px", "95% CI", "pts/40", "median err"],
    [["independent acquisition (ours)", "30.0%", "[19.9, 42.5]", "12.9", "203.62 px"],
     ["reference cropped from search pixels", "<b>65.0%</b>", "[52.4, 75.8]",
      "<b>33.5</b>", "<b>0.87 px</b>"]],
    [2.55 * inch, 0.70 * inch, 1.00 * inch, 0.60 * inch, 0.90 * inch]))
story.append(Spacer(1, 3))
story.append(P(
    "Exact McNemar 26 vs 5, <b>p = 1.9e-4</b>, &Delta; <b>+35.0 pp</b>. By severity: "
    "+11.8 / +33.3 / +18.2 / <b>+90.9</b> pp at levels 0&ndash;3 &mdash; at severity 3 the "
    "pipeline goes 1/11 &rarr; 11/11. That is the noise-fingerprint signature: severe "
    "noise is catastrophic when independent and a strong unique cue when shared. This is "
    "an <b>upper bound, not a prediction</b> &mdash; a real same-canvas generator still "
    "applies its own reference-side PSF and noise, so an unknown generator lands inside "
    "[12.9, 33.5]/40. The point stands regardless: the 20-point spread on the largest "
    "scored block is set by a generator choice we cannot observe, not by anything "
    "reachable in the solver. The pipeline is sub-pixel (0.87 px median) the moment a "
    "shared-noise cue exists.", "body"))

story.append(P("1c. The ceiling, measured with an oracle", "h2"))
story.append(P(
    "The decisive test is not &ldquo;can we rank better&rdquo; but &ldquo;is the true site "
    "separable at all&rdquo;. Hand the matcher the <b>ground-truth pose</b> and remove search "
    "from the problem entirely (localization_ceiling.json n=80; dense_residual_ceiling.json "
    "n=60):", "body"))
story.append(make_table(
    ["arm", "&le;1 px", "95% CI"],
    [["shipped (global argmax over 45 poses)", "25.0%", "[16.8, 35.5]"],
     ["<b>oracle: ground-truth pose, band-passed ZNCC</b>", "<b>31.2%</b>", "[22.2, 42.1]"],
     ["<b>oracle: ground-truth pose, dense periodic residual</b>", "<b>36.7%</b>", "[25.6, 49.3]"]],
    [3.30 * inch, 0.75 * inch, 1.00 * inch]))
story.append(Spacer(1, 3))
story.append(P(
    "Perfect pose knowledge is worth <b>~6 pp, not 70</b>. At the true pose the true site is "
    "the global maximum in only <b>14&ndash;16 of 60&ndash;80</b> pairs, with a median of "
    "10&ndash;22 locations scoring higher. Dense residual matching &mdash; periodic "
    "cancellation, the mechanism at the heart of the Phase 1 method &mdash; is statistically "
    "indistinguishable from band-passed ZNCC (paired McNemar 5 v 4, <b>p = 1.00</b>). Two "
    "consequences follow. <b>(1) Pose search is not the bottleneck</b>, so grid refinement "
    "cannot help &mdash; consistent with the earlier null at 45&rarr;99&rarr;187 poses. "
    "<b>(2) Any rule computed over these surfaces is bounded above by ~35%</b>, because "
    "ranking cannot promote a site the similarity places 20th; candidate ranking, feature "
    "re-weighting and threshold tuning are all such rules, which is why ten families of them "
    "returned the same 0.18&ndash;0.28 band. The cross-pose information the sweep discards "
    "was tested and is <i>not</i> usable: summing across poses scores 2.5%, votes 1.2%, "
    "peakiness 18.8%, against 25.0% shipped. Within the extension space the addendum allows, "
    "<b>the localization block is capped near 13/40, and the cap is a property of the "
    "evidence rather than of the estimator</b>.", "body"))

story.append(P(
    "<b>Seven targeted attacks.</b> The loss splits into disjoint buckets at the estimated "
    "pose: 22.5% where the truth is not among the candidates at all, 45.0% where it is a "
    "candidate but is not picked. Each was attacked directly and all seven failed: oracle pose "
    "(+6 pp), cross-pose sum/votes/peakiness (2.5/1.2/18.8% against 25.0%), dense periodic "
    "residual (p=1.00), five ranking rules on identical shortlists (all below peak at 35.0%), "
    "NMS widening (structurally cannot help &mdash; the global argmax is NMS-independent), and "
    "four trimmed/robust similarities. The last is the informative negative: trimming degrades "
    "accuracy <i>monotonically</i> (35.0&rarr;33.3&rarr;31.7&rarr;30.0% at 0/10/25/40% "
    "dropped). If damage were localised, discarding the worst-agreeing pixels would rescue the "
    "true site; that it never does means the disagreement is spread uniformly across the "
    "template &mdash; the signature of independent acquisition noise, not occlusion, reached "
    "from the opposite direction to &sect;1b. Behind the 22.5% bucket the truth&rsquo;s median "
    "percentile on the surface is <b>80.2%</b>: a fifth of the surface outscores it, so the "
    "defect is the true site&rsquo;s <i>score</i>, not its rank &mdash; which is why every "
    "re-ranking attack was bound to fail. Two further attacks close the remaining leads. The "
    "<b>DoG band</b> was never tuned for localization &mdash; sigma 2&ndash;8 sits directly on "
    "the 3.55&ndash;11.6&nbsp;px lattice pitch, apparently the worst possible choice since the "
    "lattice is identical at every candidate. Suppressing it should have exposed the aperiodic "
    "content; instead accuracy <b>collapses to 0%</b> above the pitch (12.5/7.5/0.0/0.0/0.0% at "
    "sigma_lo 4/6/8/12/16). The lattice is what lets the correlation lock onto a position at "
    "all: registration and disambiguation want opposite filters, and the shipped (2.0, 8.0) is "
    "the best of twelve configurations. Finally, dense.py notes that selection among near-ties "
    "is <i>the periodic-residual ranking stage's job</i> &mdash; a stage the Phase 2 path never "
    "calls. Restoring it as a shortlist re-ranker reproduces the shipped result exactly "
    "(+1/&minus;1, p=1.00), so it is not a missed opportunity either.", "body"))

story.append(P("2. What worked: DoG band-pass, confirmed at n=280", "h2"))
story.append(P(
    "results/phase2_experiments/exp01/summary.json, identical pose sweep and selection "
    "rule, only the ZNCC input differing:", "body"))

t2 = make_table(
    ["Arm", "&le;1px", "Rate", "95% CI", "Mean credit", "Pts/40", "Median s/pair"],
    [
        ["Raw", "52/280", "18.6%", "[14.5,23.5]", "0.189", "7.5", "3.49"],
        ["<b>DoG</b>", "<b>83/280</b>", "<b>29.6%</b>", "[24.6,35.2]", "<b>0.334</b>", "<b>13.3</b>", "3.54"],
    ],
    [0.5 * inch, 0.6 * inch, 0.55 * inch, 0.85 * inch, 0.75 * inch, 0.55 * inch, 0.85 * inch],
)
story.append(t2)
story.append(P(
    "Paired: fixed 49, broke 18, net <b>+31</b>, McNemar <b>p=0.0002</b>. Gain concentrates "
    "where Set B's 0.55 weight sits &mdash; a low-frequency charging-drift suppression effect, "
    "not generic contrast (robust_contrast is affine and provably cannot move ZNCC, and did not).",
    "caption"))

t3 = make_table(
    ["Severity", "Raw", "DoG", "&Delta; (pp)"],
    [
        ["0", "26.4%", "35.8%", "+9.4"],
        ["1", "27.6%", "28.9%", "+1.3"],
        ["2", "15.9%", "26.1%", "+10.1"],
        ["<b>3</b>", "<b>7.3%</b>", "<b>29.3%</b>", "<b>+22.0</b>"],
    ],
    [0.9 * inch, 0.9 * inch, 0.9 * inch, 0.95 * inch],
)
story.append(t3)
story.append(P(
    "Severity 3 quadruples but remains the floor in absolute terms (29.3% vs 35.8% at "
    "severity 0) &mdash; the hardest regime is improved, not solved.", "caption"))

story.append(P(
    "<b>Real example</b> (results/phase2_failures/rank003_p2_val-000123_err466.7px.png): true "
    "site (238.29, 496.96), FinFET, severity 3; raw-input error 465.7&nbsp;px, DoG-input "
    "497.7&nbsp;px &mdash; one of the 18 pairs DoG <i>breaks</i>, so the net +31 is not "
    "uniform. Half of measured misses land near an integer lattice translate of the truth with "
    "scale and rotation still correct: the matcher finds <i>a</i> period-consistent site, just "
    "not the right one.", "body"))

t4 = make_table(
    ["Block", "Metric", "Value", "Evidence"],
    [
        ["Localization (n=280)", "mean credit, DoG vs raw", "0.334 / 0.189", "exp01/summary.json"],
        ["Pose | localized (n=25)", "scale / rotation credit", "1.000 / 0.956&ndash;1.000", "PHASE2_FINDINGS.md &sect;1"],
        ["Rejection (lockbox n=51)", "F1, AUC, CI90", "0.9024, 0.8707, [0.838,0.953]", "presence_hgb.metadata.json"],
        ["Confidence (lockbox n=74)", "AUC vs raw ZNCC", "0.892 vs 0.760", "exp18_correctness.json"],
        ["Runtime (n=60, uncontended)", "median/P95/max", "4.33/4.51/4.62 s", "uncontended_runtime.json"],
    ],
    [1.55 * inch, 1.55 * inch, 1.75 * inch, 1.85 * inch],
)
story.append(P("Compact scoreboard", "h2"))
story.append(t4)

story.append(P("3. What we killed, and why (do not re-run)", "h2"))
t5 = make_table(
    ["Hypothesis", "Verdict", "Evidence"],
    [
        ["Finer pose grid (45&rarr;99&rarr;187 poses)", "no effect", "identical credit to 3 decimals"],
        ["5 anti-alias filters", "no effect", "byte-identical site on 24/25 pairs"],
        ["Candidate pool caps (400, 32)", "harmful", "each silently deleted the true site"],
        ["Lattice-normalized DoG bandwidth", "harmful", "&minus;16 pp; measured pitches (3.55&ndash;11.6px) 1.8&ndash;22.5&times; too narrow"],
        ["Nearest-centre rule on full surface", "harmful", "0/60 fixed at any margin; likely a scope bug (Phase 1 applies it inside a ranked pool, not ~800k raw positions)"],
        ["Ranker retraining on Phase 2 corpus", "no gain", "&minus;5.8 pp selection-given-pool (p=0.51), 8.5&times; cost"],
        ["Lattice-sibling hard negatives", "no gain", "&minus;3.8 pp vs incumbent"],
        ["Bias/gain photometric compensation", "harmful", "AUC inverts to 0.429 at severity 3 (this session)"],
    ],
    [1.9 * inch, 0.65 * inch, 4.15 * inch],
)
story.append(t5)

story.append(P("4. Final shipped architecture", "h2"))
story.append(P(
    "Phase 1's periodic-aware ZNCC pipeline, <b>extended, not replaced</b>: dense "
    "full-resolution pose sweep (45 poses) over the disclosed scale/rotation ranges, "
    "DoG-band-passed ZNCC as the matching signal, refinement to subpixel/sub-degree "
    "credit, and <b>two separate fitted models</b>: a 17-feature HGB presence model for "
    "found (F1 0.90) and a 20-feature logistic correctness model for score (AUC 0.892 vs "
    "0.760 for raw ZNCC). They answer different questions &mdash; <i>is the reference "
    "here</i> versus <i>is the coordinate I am reporting correct</i> &mdash; and a "
    "present-but-mislocalized pair must score low on the second while being 1 on the "
    "first, so one probability cannot serve both. "
    "See README.md &ldquo;Phase 1 to Phase 2: what changed&rdquo;.", "body"))

story.append(P("5. Honest limitations", "h2"))
lim_items = [
    "<b>Selection is the only bottleneck that matters</b>: pool 90&ndash;95%, selection "
    "15&ndash;43%. Harvest engineering will not move the score; new discriminating "
    "evidence is required and does not yet exist.",
    "<b>FinFET is not weaker than DRAM</b> under Phase 2 conditions &mdash; the Phase 1 "
    "README claim does not reproduce (25.0% vs 34.6% after band-pass, overlapping "
    "intervals on smaller per-architecture samples). This document supersedes that claim.",
    "<b>Rejection lockbox is small</b> (51 and 55 scenes, two slices). F1=0.9024 point "
    "estimate, but the 90% CI lower tail is <b>0.838</b>, so the F1&ge;0.90 bonus gate is "
    "at the boundary, not cleared.",
    "<b>Confidence AUC 0.892 (n=74) beats raw ZNCC 0.760</b> on the same lockbox, but n=74 "
    "is a small sample; the run is checkpointed to exp18_correctness.json.",
    "<b>Runtime is fine for the shipped path</b> &mdash; median <b>4.33s</b>, P95 4.51s, max "
    "4.62s over 60 pairs measured uncontended with threads pinned to 4, so 4.3&times; headroom "
    "on the 20s cap and 0/60 over 5s. The figure rose from 3.54s when the correctness model "
    "was added. Forcing the budget down confirms the Deadline path sheds refinement and "
    "presence rather than overrunning, and the output contract holds at every budget tested "
    "to 0.5s. Separately, the heavier harvester used to measure 90&ndash;95% pool "
    "recall is not what ships &mdash; at that config 60/316 pairs (19%) exceeded the 20s cap "
    "(gt_pose_ceiling.json). Pool recall has not been re-measured on the exact shipped config.",
    "Pose credit (n=25) is a small sample; treat 1.000/0.956 as encouraging, not certain.",
    "All scored evidence is synthetic and self-generated; no organizer sample pairs were "
    "available at analysis time.",
]
for it in lim_items:
    story.append(P("&bull; " + it, "bodyb"))

story.append(P("6. Next steps, ranked by evidence", "h2"))
next_items = [
    "<b>Do not spend further effort on candidate ranking.</b> &sect;1c bounds every such rule "
    "at ~35% even with perfect pose, and new features over the same surfaces inherit that "
    "bound. The only escape is a different similarity measure, and a materially different "
    "method is a no-appeal disqualification &mdash; closed by rule, not just by budget.",
    "Grow the rejection/confidence lockbox past 51&ndash;74 scenes before trusting the "
    "F1&ge;0.90 gate or the 0.892 AUC estimate.",
]
for it in next_items:
    story.append(P("&bull; " + it, "bodyb"))

story.append(P("7. Reproducibility", "h2"))
story.append(P(
    "Python 3.11, CPU-only, no network at run time (scripts/verify_offline.py, "
    "audit-hook based). Weights ship inside the package (presence_hgb.pkl, correctness_lr.pkl, "
    "hgb_r2.joblib, SHA256-pinned in their metadata files) and are loaded script-relative. "
    "scripts/build_submission.py builds the zip from an explicit allow-list, fails if any "
    "required file is absent, then extracts the finished archive to a scratch directory and "
    "runs the documented entry point inside it. Validation is deterministic: 12 gates in "
    "scripts/validate_phase2.py, byte-identical regeneration, and "
    "results/phase2_experiments/exp01/per_pair.csv reproduces the DoG-vs-raw comparison "
    "in Section 2 row-for-row.", "body"))

doc.build(story)
print("built", OUT)
