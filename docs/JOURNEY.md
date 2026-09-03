# The journey: from a Phase 1 locator to a Phase 2 registration system

This is the engineering record: what we measured, what we shipped, what we threw
away, and the three claims we had to retract about our own system. It is written
for a reader deciding whether to trust the numbers in
[README.md](../README.md), so the dead ends are here too.

Ground rule for everything below: **two number families, never mixed.**
The *official sample* is 20 organizer-generated pairs, scored once after the
solver was frozen — that is the 79.14 / 85. Everything else is our own
generator, which renders reference and search as independent acquisitions and is
materially harder. Section 5 is where that stops being an excuse and becomes a
measurement.

| Jump | Section |
|---|---|
| [1](#1-where-we-started) | Where we started: the Phase 1 locator |
| [2](#2-what-phase-2-broke) | What Phase 2 broke, on the first honest run |
| [3](#3-the-one-change-that-survived) | The one change that survived |
| [4](#4-ten-families-that-did-not) | Ten experiment families that did not |
| [5](#5-why-it-is-hard-the-root-cause) | Why it is hard: the root cause |
| [6](#6-the-25-points-nobody-had-built) | The 25 points nobody had built |
| [7](#7-what-it-does-today) | What it does today, run live |
| [8](#8-three-things-we-had-to-retract) | Three things we had to retract |
| [9](#9-what-we-would-do-next) | What we would do next |

---

## 1. Where we started

Phase 1 gave us the scale (exactly 10×), promised the reference was present, and
asked for `x, y`. Under those rules the locator was strong:

| Protocol | Pairs | Within 5 px | Median error |
|---|---:|---:|---:|
| External development | 120 | **93.33%** | 1.44 px |
| External holdout | 30 | **100.00%** | 1.46 px |
| Internal fixed stress | 80 | 48.75% | 62.57 px |
| Internal randomized | 40 | 55.00% | 4.36 px |

![V1 localization across named protocols](images/v1_benchmarks.svg)

The split between the external and internal columns is the whole story of this
project in advance. Where the reference and the search share an acquisition, the
answer is subpixel. Where they do not, half the pairs land a lattice period away.
We did not understand that this was the governing variable until section 5.

## 2. What Phase 2 broke

The addendum removed three assumptions: the zoom became unknown in `[8, 12]`,
rotation became a reported quantity within ±5°, and about one pair in five
contains no true instance at all.

![Phase 2 removes three Phase 1 assumptions](images/v2_phase_change.svg)

We extended the matcher over a pose grid and measured it on 180 present pairs
from our own `p2_val` and `p2_stress`. The result was bad, and useful:

| Metric | Value |
|---|---|
| Site selection within 5 px | **33 / 180 = 18.3%**, 95% CI [13.4%, 24.6%] |
| Mean tiered credit | 0.183 → **7.3 / 40** |
| Runtime | median 6.75 s, max 17.4 s |

The important part was not the rate. It was the shape of the error:

> At n = 180, the count within 1 px, 2 px, 3 px and 5 px is the **same 33
> pairs**. Not one pair in 180 landed between 1 and 5 px.

**Localization error is binary.** When the pipeline picks the right lattice
site, it is already subpixel; when it picks a wrong one, it is off by hundreds of
pixels. Mean localization credit therefore *equals* site-selection accuracy, and
no amount of subpixel refinement can move the 40-point block. That single
observation redirected the rest of the work: stop polishing the answer, start
choosing the right site — and where that fails, say so honestly in `score`.

Two blocks were already healthy on pairs that localize. Pose recovery converts
essentially fully: scale within 1% on 25/25 pairs, median error 0.12%, rotation
median error 0.032°. And peak correlation against hit/miss gave AUC 0.704 —
before pose search existed, every presence feature had measured within one
standard error of chance.

## 3. The one change that survived

A Difference-of-Gaussians band-pass on the correlation input (σ = 2 minus σ = 8),
with the pose sweep and selection rule held identical. n = 280 present pairs:

| Arm | Within 1 px | Rate | 95% CI | Credit | Points / 40 | Runtime |
|---|---:|---:|---|---:|---:|---:|
| raw | 52 | 18.6% | [14.5, 23.5] | 0.189 | 7.5 | 3.49 s |
| **DoG** | **83** | **29.6%** | [24.6, 35.2] | **0.334** | **13.3** | 3.54 s |

Paired: 49 fixed, 18 broken, net **+31**, McNemar **p = 0.0002**, with
non-overlapping intervals. Roughly +5.8 points for a one-line change and 50 ms
per pair.

The reason to believe it is a mechanism rather than a fluke is *where* the gain
lands:

| Severity | raw | DoG | Δ |
|---|---:|---:|---:|
| 0 | 26.4% | 35.8% | +9.4 |
| 1 | 27.6% | 28.9% | +1.3 |
| 2 | 15.9% | 26.1% | +10.1 |
| **3** | **7.3%** | **29.3%** | **+22.0** |

![Band-pass ablation by severity](images/v2_filter_ablation.svg)

Severity 3 quadruples. That is the signature of suppressing low-frequency
charging drift, not of generic contrast enhancement — and Set B, which carries
0.55 of the localization weight, is exactly the degraded regime. It also
explains a Phase 1 dead end: `robust_contrast` is an affine intensity map and
ZNCC is invariant to those, so it provably could not have helped. A band-pass is
not affine.

## 4. Ten families that did not

A structured campaign ran ten experiment families, each screening its variants
against a cheap within-scene gate before earning an end-to-end run. **Not one
produced a localization improvement that survived full-pool measurement.**

| Family | Verdict | Decisive number |
|---|---|---|
| F19 found-threshold robustness | **STRONG GO** | identical metrics across [0.40, 0.52]; 0.48 retained |
| F5 sibling-relative evidence | **KILL**, reversed from STRONG GO | screen +11.6 pp → full pool **−0.42 pp**, McNemar p = 1.00 |
| F10 canonical unit-cell defect | KILL | best AUC 0.677 against raw ZNCC 0.775 |
| F14 SEM edge profile | KILL | best 0.663 against directionality 0.740 |
| F4 candidate peak topology | KILL | best 0.557, increment **−16.9 pp** |
| F6 bias/gain compensation | KILL | features clear 0.70 alone, increment negative for all ten |
| F11 DRAM contact fingerprint | INVESTIGATE | 7/10 clear 0.70, increment unproven |

F5 is the one worth reading. It screened at **+11.6 pp** on a sibling-weighted
hard-negative subset with an untuned rank-sum, and was called a strong go with
three caveats: a subset rather than the full pool, an untuned rule, and a
confidence interval sitting on the gate boundary. Resolving all three reversed
it — on the full pool with a fitted model over 42 features, end-to-end credit
moved −0.42 pp (90% CI [−6.67, +5.42]), and an ablation put the 11 sibling
features' own contribution at −4.17 pp.

The tell was the severity pattern. F5's screen showed severity 2 as its largest
*gain*, at +27.7 pp; on the full pool severity 2 was its largest *loss*, at
−11.67 pp. A sign reversal in the stratum that carried the original effect is
the signature of subset-selection noise, not of a real effect shrinking.
Shipping on the screen would have cost +0.88 s per pair for −0.42 pp.

Two general lessons came out of the campaign and are worth more than any of the
features would have been:

**Pairwise AUC and top-1 selection are different problems.** The DoG score
reaches within-scene AUC **0.937** against the true-site label while selection
sits near 50%. At 0.937, roughly 6% of candidates outrank the truth — about 30
of them in a 500-candidate pool. Beating a random wrong candidate and beating
*every* wrong candidate are not the same task, and features get screened on the
first while the rubric pays for the second.

**Increment must be measured against the baseline inference actually has.** The
harness control computed raw ZNCC at *ground-truth* pose and scored 0.965–0.972.
Against a near-perfect oracle nothing shows increment. The pipeline's real
estimated-pose figure is ~0.775, and F11 survived as INVESTIGATE rather than
KILL entirely on that distinction.

## 5. Why it is hard: the root cause

Ten families had asked *how do we improve localization*. This campaign asked
**why is it hard**, and whether our corpus is a fair proxy for the blind set. No
code changed.

First, the two things we had deliberately built to make the corpus hard turned
out not to matter. Our generator plants near-duplicate decoys in 41.5% of
present pairs and runs a severity ladder past the disclosed families:

| Arm | n | Within 1 px | 95% CI |
|---|---:|---:|---|
| severity 0–1, **no** decoys ("Set A-like") | 180 | 30.0% | [23.8, 37.1] |
| severity 0–1, **with** decoys | 110 | 31.8% | [23.9, 41.0] |
| full severity mix | 110 | 30.0% | [22.2, 39.1] |

Three indistinguishable results, with the decoy arm nominally *above* the clean
one. Neither planted decoys nor the severity ladder measurably moves
localization. Evidence:
[set_a_calibration.json](../results/phase2_experiments/set_a_calibration.json).

What does move it is one design decision, and it is not in the solver. Our
generator renders the reference and the search as **independent acquisitions** of
one latent scene — separate RNG streams, so the noise realizations differ. Many
generators instead crop the reference out of the already-rendered search pixels,
where the noise pattern becomes a unique fingerprint of the true site. Paired
test, n = 60, identical pairs, identical pipeline, differing only in how the
reference was produced:

| Arm | Within 1 px | 95% CI | Points / 40 | Median error |
|---|---:|---|---:|---:|
| independent acquisition (our corpus) | 30.0% | [19.9, 42.5] | 12.9 | 203.62 px |
| reference cropped from the search pixels | **65.0%** | [52.4, 75.8] | **33.5** | **0.87 px** |

Exact paired McNemar: 26 wins to 5, **p = 1.9 × 10⁻⁴**, **Δ +35.0 pp**. By
severity the gap is +11.8 / +33.3 / +18.2 / **+90.9** pp — at severity 3 the
pipeline goes from 1/11 to 11/11. Severe noise is catastrophic when independent
and a *strong unique signature* when shared, which is precisely the
noise-fingerprint mechanism.

![The acquisition-independence gap](images/v2_acquisition_gap.svg)

The cropped arm is an **upper bound, not a prediction**: a real same-canvas
generator still applies its own reference-side PSF and noise, so a generator of
unknown design lands somewhere inside [12.9, 33.5] of 40. What this establishes
is that the entire ~20-point spread on the largest scored block is controlled by
a generator design choice we cannot observe — not by anything reachable in the
solver. Evidence:
[samecanvas_bound.json](../results/phase2_experiments/samecanvas_bound.json).

Three independent angles then pin the bottleneck exactly:

```text
detection   ~99-100%
pool         ~90-95%
selection     15-43%   <-- here
```

Pool-high and selection-poor is the *identity-evidence* condition. It is not a
threshold, non-maximum-suppression, cap or dedup problem, so further harvest
engineering is not where the remaining points are.

## 6. The 25 points nobody had built

While localization was absorbing every experiment, 25 of the 85 scored points
sat untouched: 15 for rejection, 10 for confidence calibration. They needed no
new evidence — only that we stop conflating three different questions.

`found` answers *is the reference in this image at all*. `score` answers *is the
answer I am about to report correct*. A present pair localized 40 px away is
**incorrect**, so a presence probability is the wrong quantity for the score
column. They are separate models, and `score` is computed identically whether
`found` is 0 or 1.

![found and score answer different questions](images/v2_presence_vs_score.svg)

The third question is *did I crash*. An internal error is never written as a
confident rejection: it emits `found = 0` with a sentinel score and a line on
stderr. Conflating "I looked and it is not there" with "I crashed" would corrupt
the rejection F1 and the confidence AUC at the same time, and make the run
undebuggable afterwards.

Around that sit the parts that are worth points precisely because they are
boring: exactly one row per `pair_id` whatever happens, zeroed pose columns when
`found = 0`, atomic CSV writes, a per-pair deadline that sheds refinement and
presence rather than overrunning, and a packager that refuses to build if a
weight file is missing. A missing row scores zero regardless of algorithm
quality.

Runtime came along the same way — by subtraction, not addition. Three
memoizations, each verified to leave predictions byte-identical, took the median
from 6.75 s at first measurement to **2.92 s** (n = 60, uncontended, threads
pinned to 4, max 3.55 s): the anti-alias template is built once per scale rather
than once per pose, `scene_features` reuses the refinement the entry point
already computed, and the ZNCC denominator's windowed-variance term — which
depends on template *shape* only — is computed once per scale, 63 FFT
convolutions across the sweep where 135 were run.

## 7. What it does today

On the official 20-pair organizer sample, run once after freezing:
**79.14 / 85**, plus the RGB and rejection-F1 bonuses.

![Official-sample fill against the published rubric](images/v2_official_scorecard.svg)

That is 20 pairs, so it is a measurement with wide intervals, not a promise. The
figures below are the opposite trade: our own corpus, harder by construction, at
a sample size where the rate means something. They come from
`scripts/build_inference_gallery.py`, which calls the same `register.process`
the scored entry point calls, with the same packaged weights and the same
per-pair budget.

![A single registration, end to end](images/v2_inference_walkthrough.jpg)

Green is ground truth, blue is what the solver reported, orange dashed squares
are the near-duplicate impostor sites the generator planted to catch it. The
panel on the right is the CSV row against the truth it was scored on.

![Twelve real runs](images/v2_inference_gallery.jpg)

This gallery is deliberately not a highlight reel: it holds the wins, the
characteristic miss, and the rejections. The misses are what section 2 predicted
— a wrong lattice copy hundreds of pixels away, never a blurry near-answer.

That run also re-confirms section 2's binary-error finding on fresh data, five
weeks and one shipped solver later. Of the 139 present pairs that received a
pose:

| Within | 1 px | 2 px | 3 px | 5 px | 10 px | 50 px |
|---|---:|---:|---:|---:|---:|---:|
| Pairs | 53 | 55 | 56 | 56 | **56** | 59 |

The same 56 pairs, from 1 px out to 10 px. The finding that redirected the whole
project reproduces without being looked for.

![Score against localization error](images/v2_score_vs_error.svg)

This is the plot that matters operationally. When the solver picks the wrong
site, the score column largely knows: mean **0.379** on correct answers against
**0.192** on wrong sites and **0.056** on absent pairs, AUC **0.756** over the
same 199 pairs. The distributions overlap, so it is a ranking signal for triage
rather than a clean threshold — which is exactly why it is reported as a
separate column instead of being folded into `found`, which scores F1 **0.906**
here (TP 139, FP 11, FN 18, TN 31). Per-pair records, misses included:
[inference_gallery.json](../results/phase2_experiments/inference_gallery.json).

![Presence and localization by severity](images/v2_presence_evidence.svg)

One more thing this run says, and it is not flattering to our own corpus design:
localization by severity comes out at 39 / 31 / 45 / 25% for levels 0–3. It does
not fall off cleanly with severity. That is the section 5 finding showing up
again — the severity ladder is not the variable that governs the rate, and the
one that does is not in the solver.

## 8. Three things we had to retract

Kept because a record that only contains confirmations is not a record.

**"Candidate recall is the bottleneck."** An earlier figure of 74.3% pool recall
was measured through the funnel path at *estimated* pose and repeatedly quoted as
"a quarter of pairs never have the true site available". Measuring the pool
question directly, with pose fixed at ground truth to isolate it, gives
**90–95%**. The two numbers measure different things and should never have been
compared. The 90 → 95% improvement also came from adding DoG as a fourth harvest
view, not from the lattice-cell quota we had credited: a no-quota ablation ties
all three quota variants exactly. The quota's real effect is 20–38% pool
compression at equal recall.

**"FinFET is the weak architecture."** It is not, under Phase 2 conditions:
17.2% against DRAM's 19.4%, with thoroughly overlapping intervals, and FinFET
*gains more* from DoG (+12.5 against +9.6 pp). The claim had reached our own
documentation before measurement retired it.

**"Nearest-to-centre selection is harmful."** Our test applied the rule to the
entire correlation surface at margin 0.02 — roughly 800,000 positions, many
near-max on a periodic field — which pulls the answer toward the image centre
regardless of evidence. Phase 1 applies it *within a ranked candidate pool*. The
measured harm (p = 0.017) kills our implementation, not necessarily the rule.

One further caveat we never closed: a G3 gate record puts a published sponsor
baseline at 0.50 on `p2_val` against our 0.28–0.33. The control run was started
twice and stalled both times. A simple published baseline outscoring us on our
own corpus is the most actionable single fact available, and it remains
unresolved — which is why it is here and not omitted.

## 9. What we would do next

In order of expected points, given what sections 4 and 5 rule out:

1. **Settle the sponsor-baseline gap.** One clean paired run on `p2_val`. If a
   brute-force ZNCC grid really does beat us there, everything above is
   mis-prioritised, and that is worth knowing before anything else is built.
2. **New identity evidence, not re-weighted evidence.** On 67,370 held-out
   candidates the lattice-relative features score AUC 0.38–0.62 against the
   true-site label. Re-weighting what we already compute is closed by
   measurement; the discriminating information is not in the feature set.
3. **Retest nearest-to-centre correctly**, inside the ranked pool, per section 8.
4. **Screen on top-1, never on pairwise AUC.** Every reversal in section 4
   traces back to a gate that measured the wrong quantity.

Anything that adds runtime without addressing site *identity* is, on this
evidence, a way to spend 0.9 seconds per pair for nothing.

---

Jury run sheet: [HOW_TO_RUN.md](HOW_TO_RUN.md) ·
Architecture: [V1_VS_V2.md](V1_VS_V2.md) ·
Limits: [failure_analysis.md](failure_analysis.md) ·
Dated notebook this was distilled from: [PHASE2_FINDINGS.md](PHASE2_FINDINGS.md)
