# Phase 2 findings and game plan

> **Dated internal notebook (30 August 2026), not the scored result.**
> Localization figures in this file are from an early internal stress corpus.
> The jury-facing numbers are in [README.md](../README.md) and
> [V1_VS_V2.md](V1_VS_V2.md): **79.14 / 85** on the official 20-pair sample.
> Do not quote this notebook as the current submission score.

Everything measured on 30 August 2026, with the evidence behind each claim and
what it implied for the remaining days at that date. Submission closed
**3 September 2026, 23:59**.

Several early estimates in this notebook came from 25-pair samples with a 95%
interval of roughly +/-18 percentage points and were, in hindsight, inside
their own noise.

---

## 1. Current measured state

### Localization: 18.3%, or 7.3 of 40 points

`dense_pose_search`, 180 present pairs across `p2_val` and `p2_stress`:

| metric | value |
|---|---|
| Site selection (<= 5 px) | **33/180 = 18.3%**, 95% CI [13.4%, 24.6%] |
| Mean tiered credit | **0.1833** -> **7.3 / 40** |
| Runtime | median 6.75 s, p95 14.7 s, max 17.4 s |

**Localization error is binary.** At n=180, the count within 1 px, 2 px, 3 px
and 5 px is the *same 33 pairs*. Not one pair in 180 landed between 1 and 5 px.
When the pipeline finds the right site it lands inside a pixel; when it misses
it misses by hundreds.

The consequence governs all strategy: **mean localization credit equals
site-selection accuracy exactly.** Subpixel work, refinement quality and pose
precision cannot move the 40-point block at all. Only the fraction of pairs
landing on the correct site can.

Breakdowns:

| split | hit rate | | severity | hit rate | | architecture | hit rate |
|---|---|---|---|---|---|---|---|
| `p2_val` | 22.5% | | 0 | 17.6% | | DRAM | 19.4% |
| `p2_stress` | 10.0% | | 1 | **33.3%** | | FinFET | 17.2% |
| | | | 2 | 17.4% | | | |
| | | | 3 | **5.8%** | | | |

Two things worth carrying forward. **Severity dominates** -- severity 3 collapses
to 5.8% against severity 1's 33.3%, and Set B is weighted 0.55, so any
improvement must be reported per severity or it is not informative. And
**FinFET is not weaker than DRAM** under Phase 2 conditions (17.2% vs 19.4%,
thoroughly overlapping intervals). The README's standing claim that FinFET is
the weak architecture does not survive measurement and should be corrected
before it reaches `failure_analysis.pdf`.

### Pose: effectively solved, 20 of 20

On pairs that localize, after `refine.refine_candidate`:

| quantity | credit | detail |
|---|---|---|
| Scale | **1.000** | 25/25 within 1%, median error 0.12% |
| Rotation | **0.956-1.000** | median error 0.032 deg |

Pose is gated on localization credit, so this converts fully whenever
localization succeeds. **It is not a place to spend further effort.**

### Confidence: a real signal exists

Peak ZNCC score against hit/miss gives **AUC 0.704** (n=180), stable at 0.722
on an independent 25-pair sample. Before pose search existed, every presence
feature measured within one standard error of 0.5 -- indistinguishable from
chance. This is now a genuine foundation for the 15 rejection + 10 calibration
points.

---

## 2. Infrastructure: verified and banked

| item | status | evidence |
|---|---|---|
| Python 3.11 compatibility | **Fixed** | `numpy==2.5.1` and `scipy==1.18.0` have **no cp311 wheels**; `pip install` aborted before running any code. Repacked to 2.4.6 / 1.17.1, verified by real resolver + clean 3.11 venv |
| Test suite on 3.11 | **110 passing** | up from 64; identical results to 3.12 |
| Per-pair watchdog | **Shipped** | `budget.py`; worst case 26.3 s -> 11.8 s, **zero credit cost** |
| Offline / foreign-cwd gate | **Shipped** | `scripts/verify_offline.py`, audit-hook based, self-testing |
| Phase 1 contract | **Frozen** | render hashes byte-identical to committed HEAD |
| Dataset | **28,896 pairs verified** | 70/30 stratified to within 0.1 pp on presence, modality, architecture and severity |
| Repo hygiene | **Done** | superseded artifacts removed, 15 MB debug tree ignored, `generate_dataset.py` documented |

Three of the five disqualification rules are now mechanically checked rather
than asserted. The remaining two need `register.py` to exist.

---

## 3. Ruled out by measurement

Each of these was a plausible hypothesis, tested directly, and refuted. They are
recorded so nobody spends time re-deriving them.

| hypothesis | verdict | evidence |
|---|---|---|
| Pose grid too coarse | **No** | 45 vs 99 vs 187 poses give *identical* credit to three decimals; not one pair flips |
| Wrong anti-alias filter | **No** | Five builders (gaussian 0.35s, box-matched, exact area-average, cubic, Lanczos) pick the **byte-identical site on 24/25 pairs**. Area-averaging is *worse* (0.200 vs 0.280) |
| Candidate pooling / screening | **Harmful** | Caps at 400 and at 32 each silently deleted the true site; dense sweep 0.280 vs funnel 0.112 |
| Planted decoys defeating us | **No** | **0 of 18** misses within 5, 15 *or* 30 px of a decoy site; closest approach 123.7 px |
| Hand-designed selectors | **No** | Seven tried (ZNCC, PSR, refinement gain, best-over-grid, Bayes/Hessian sharpness, pose consistency, MRF-style rescoring); all 0.07-0.28 |
| Phase 1 residual + structural + HGB at estimated scale | **No** | 0.112, and median runtime 8.15 s with 10/25 pairs over the 20 s timeout |
| Ranker retraining on Phase 2 corpus | **No** | -5.8 pp selection-given-pool (p=0.51), cost 8.5x |
| Lattice-sibling hard negatives | **No** | -3.8 pp vs incumbent, +1.9 pp vs retrain -- short of the +4 pp gate |
| Nearest-centre equivalence *as implemented* | **Harmful** | net -12 paired, **McNemar p=0.017**. See caveat in section 5 |

### The finding that closes a whole strategy

On 67,370 held-out candidates, the lattice-relative features score **AUC
0.38-0.62** against the true-site label:

    disp_periods_x, disp_periods_y, lat_phase_res, parity_even  -> near chance

**The discriminating information is not in the current feature set.** That is
why seven hand-designed selectors, the full Phase 1 stack and two retrained
models all landed in the same 0.18-0.28 band. Re-weighting existing evidence
cannot work. New evidence is required.

---

## 4. What works

**Difference-of-Gaussians input -- CONFIRMED and shipped.** n=280 present pairs
across `p2_val` and `p2_stress`, identical pose sweep and selection rule, only
the similarity input differing:

| arm | <=1 px | rate | 95% CI | credit | pts/40 | runtime |
|---|---:|---:|---|---:|---:|---:|
| raw | 52 | 18.6% | [14.5, 23.5] | 0.189 | 7.5 | 3.49 s |
| **DoG** | **83** | **29.6%** | [24.6, 35.2] | **0.334** | **13.3** | 3.54 s |

**Paired: fixed 49, broke 18, net +31, McNemar p = 0.0002.** Non-overlapping
intervals. **+5.8 points for a one-line change and 50 ms per pair.**

The gain concentrates exactly where the scoring weight is:

| severity | raw | DoG | delta |
|---|---:|---:|---:|
| 0 | 26.4% | 35.8% | +9.4 |
| 1 | 27.6% | 28.9% | +1.3 |
| 2 | 15.9% | 26.1% | +10.1 |
| **3** | **7.3%** | **29.3%** | **+22.0** |

Severity 3 quadruples, and `p2_stress` gains +15.3 pp against `p2_val`'s +8.8.
Set B carries 0.55 of the localization weight, so this is the most valuable
possible place for a gain to land. That profile -- concentrated in the degraded
regime -- is the signature of suppressing the low-frequency charging drift that
dominates Set B, not a generic contrast effect.

Why it works when `robust_contrast` did not: `robust_contrast` is an affine
intensity map and ZNCC is invariant to those, so it provably cannot change the
result. A band-pass is not affine.

FinFET gains more than DRAM (+12.5 vs +9.6 pp), further burying the standing
claim that FinFET is the weak architecture.

**Dense full-resolution sweep over pooled funnels.** Every funnel stage that
existed to save time discarded the answer.

**45-pose grid.** Identical accuracy to 187 poses at a quarter the cost.

**A valid `register.py`.** Emits every row, zeroes pose on `found=0`, keeps
errors distinguishable from rejections via a sentinel score, writes atomically,
runs from a foreign cwd on Python 3.11 with no network, 2.87 s/pair.

### Closed by measurement, do not re-run

| experiment | verdict | evidence |
|---|---|---|
| Lattice-normalized DoG bandwidth | KILL | measured pitches 3.55-11.60 px make the band 1.8-22.5x too narrow; -16 pp credit |
| Nearest-centre evidence equivalence | KILL | n=60 end-to-end: **zero** pairs fixed at any margin, -6 pp at loose margins; equivalence sets are singletons (median \|E\|=1) so there are no ties to arbitrate |
| Lattice-cell quota harvesting | INVESTIGATE (recall claim refuted) | no-quota ablation ties it exactly; real effect is 20-38% pool compression, not recall |

The nearest-centre convention is effectively *inert* on DoG-sharpened surfaces,
because they do not produce the evidence ties the rule exists to arbitrate. It
should not be described as an active selection mechanism. The separate Phase 1
exact-wallpaper path still handles genuine non-identifiability and remains
correct -- it triggers on score collapse, which the same measurement confirms is
not occurring.

---

---

## 4b. The 31 August experiment campaign -- ten families, one night

A structured campaign ran ten experiment families overnight, each screening its
variants against a cheap within-scene gate before earning an end-to-end run.
**No family produced a localization improvement that survived full-pool
measurement.** That is the campaign's result, and it is worth more than a
feature shipped on a screen would have been.

| family | verdict | decisive number |
|---|---|---|
| F5 sibling-relative evidence | **KILL** (reversed from STRONG GO) | screen +11.6 pp -> full pool **-0.42 pp**, McNemar p=1.00 |
| F19 found-threshold robustness | **STRONG GO** | identical metrics across [0.40, 0.52]; 0.48 retained |
| F10 canonical unit-cell defect | KILL | best AUC 0.677 vs raw ZNCC 0.775 |
| F14 SEM edge profile | KILL | best 0.663 vs directionality 0.740 |
| F4 candidate peak topology | KILL | best 0.557, increment **-16.9 pp** |
| F6 bias/gain compensation | KILL | features clear 0.70 alone; increment negative for all 10 |
| F11 DRAM contact fingerprint | INVESTIGATE | 7/10 clear 0.70; increment unproven, runtime fixed 1,058x |
| F2, F3 | rejected | returned predicted or out-of-range numbers, not measurements |

### The F5 reversal, and why it is the campaign's most useful result

F5 screened at **+11.6 pp** selection-given-pool on a sibling-weighted
hard-negative subset with an untuned rank-sum, and was called STRONG GO with
three explicit caveats: subset rather than full pool, untuned rule, CI on the
gate boundary. Resolving all three reversed it.

On the full pool with a fitted LogisticRegression over 42 features, end-to-end
tiered credit moved **-0.42 pp** (90% CI [-6.67, +5.42], McNemar p=1.00), and an
ablation isolating the 11 sibling features put their own contribution at
**-4.17 pp** (CI [-8.33, +0.00]) -- excluding positive values.

The tell is the severity pattern. F5's screen showed severity 2 as its largest
gain at **+27.7 pp**; on the full pool severity 2 is its largest loss at
**-11.67 pp**. A sign reversal in the stratum carrying the original effect is
the signature of subset-selection noise, not of a real effect shrinking.

Shipping it on the screen would have cost **+0.88 s/pair for -0.42 pp**.

### What the campaign establishes

**Every attempt to add new evidence lost to evidence the pipeline already
computes.** Unit-cell defect fields, SEM edge profiles, peak topology and
photometric compensation all scored below raw ZNCC (~0.775 at estimated pose)
and the directionality channel (~0.740). The one family that reframed *existing*
evidence screened well and then failed on the full pool.

**Pairwise AUC and top-1 selection are different problems.** DoG's absolute
score reaches within-scene AUC **0.937** against the true-site label, yet
selection sits near 50%. At 0.937, roughly 6% of candidates outrank the truth --
about 30 of them in a 500-candidate pool. Beating a random wrong candidate and
beating every wrong candidate are not the same task, and features are screened
on the first while the rubric pays for the second.

**Increment must be measured against the baseline inference actually has.** The
harness control computed raw ZNCC at *ground-truth pose* and scored 0.965-0.972.
Against a near-perfect oracle nothing can show increment. The pipeline's real
estimated-pose figure is ~0.775. F11's INVESTIGATE rather than KILL rests on
exactly this distinction.

### Method notes worth keeping

Four harness-level artifacts each produced a confident wrong answer before being
caught: a `-1.0` out-of-scale score sentinel; a recall floor set so loose it
admitted 1.7M candidates and made recall trivially 100%; a `max()` tie-order
bias that inflated top-1 accuracy from chance (0.09) to an apparent 0.60; and a
wrong-channel selection that produced 255x-wrong photometric values.

The families whose nulls are trustworthy (F10, F14) are the ones that
voluntarily reproduced a known baseline as a positive control. That practice,
not the gate thresholds, is what makes a negative result believable.

Three of ten families returned projected or out-of-range numbers formatted as
results. Every reported figure needs a range and provenance check before it
enters the record.

## 4c. The 31 August calibration campaign -- what actually makes this hard

Ten experiment families had tried to *improve* localization and all failed. This
campaign asked a different question: **why is it hard**, and is our corpus a
fair proxy for the blind set? Three measurements, all on the shipped pipeline
with no code changes.

### Set A conditions do not rescue localization

The addendum describes Set A as "noise comparable to the Phase 1 sample prompt"
and never mentions decoys. Our corpus plants decoys in 41.5% of present pairs
and runs a severity ladder past the disclosed families, so the obvious
hypothesis was that we had made the problem artificially hard.

| arm | n | <=1px | 95% CI | credit -> /40 |
|---|---:|---:|---|---:|
| A  severity 0-1, **no decoys** ("Set A-like") | 180 | 30.0% | [23.8, 37.1] | 12.3 |
| B  severity 0-1, **with decoys** | 110 | 31.8% | [23.9, 41.0] | 14.8 |
| C  full severity mix | 110 | 30.0% | [22.2, 39.1] | 12.8 |

Three arms, three indistinguishable results. **Neither the severity ladder nor
the planted decoys measurably move localization.** B is nominally *above* A. The
corpus is not unfairly hard along either axis, and Set A should not be expected
to rescue the block. Evidence:
`results/phase2_experiments/set_a_calibration.json`.

### What does move it: acquisition-noise independence, by +35 points

Our generator renders the reference and the search as **independent
acquisitions** of one latent scene -- separate RNG streams, so the noise
realizations differ. Many generators instead crop the reference out of the
rendered search pixels, where the noise pattern becomes a unique fingerprint of
the true site.

Paired test, n=60, identical pairs, shipped pipeline, differing only in how the
reference was produced:

| arm | <=1px | 95% CI | credit -> /40 | median err |
|---|---:|---|---:|---:|
| I  independent acquisition (our corpus) | 30.0% | [19.9, 42.5] | 12.9 | 203.62 px |
| S  reference cropped from the search pixels | **65.0%** | [52.4, 75.8] | **33.5** | **0.87 px** |

Exact McNemar, paired: 26 wins for S against 5 for I, **p = 1.9e-4**.
Delta **+35.0 pp**. By severity the gap is +11.8 / +33.3 / +18.2 / **+90.9** pp
at levels 0-3 -- at severity 3 the pipeline goes from 1/11 to 11/11. That is
precisely the signature of the noise-fingerprint mechanism: severe noise is
catastrophic when independent and a *strong unique signature* when shared. The
effect is identical with decoys (+34.8) and without (+35.1).

**Arm S is an upper bound, not a prediction.** A real same-canvas generator
would still apply its own reference-side PSF and noise, so a generator of
unknown design lands somewhere inside [12.9, 33.5] of 40. What this establishes
is that the entire 20-point spread on the largest scored block is controlled by
one generator design choice we cannot observe -- not by anything reachable in
the solver. Evidence: `results/phase2_experiments/samecanvas_bound.json`.

### Ranked drivers of localization failure

| driver | measured effect on <=1px |
|---|---:|
| reference/search acquisition independence | **+35.0 pp** (p=1.9e-4) |
| planted near-duplicate decoys | ~0 pp (CIs overlap) |
| severity ladder, 0-1 vs full mix | ~0 pp (CIs overlap) |

This is the honest root cause, and it retires the two hypotheses the corpus was
built to stress. It also reframes the shipped result: the pipeline is not
failing at an easy problem, it is solving a materially harder one than a
same-canvas benchmark poses, and it is sub-pixel (0.87 px median) the moment the
shared-noise cue exists.

### Runtime, measured uncontended

60 pairs stratified 15 per severity, threads pinned to 4, nothing else running,
timed through the real scored code path:

| metric | value | limit |
|---|---:|---|
| median | **4.33 s** | 5 s budget |
| P95 | 4.51 s | -- |
| max | **4.62 s** | 20 s hard timeout (4.3x headroom) |
| over 5 s | 0/60 | -- |
| over 20 s | 0/60 | -- |

The distribution is unusually flat because the dense sweep is fixed-cost, so
there is no tail risk and severity does not move it (4.32-4.39 s). Forcing
`BUDGET_S` down to simulate a slower machine confirms the `Deadline` path sheds
refinement and presence rather than overrunning, and the output contract holds
at every budget tested down to 0.5 s. Evidence:
`results/phase2_experiments/uncontended_runtime.json`.

## 5. Open findings and caveats

**Candidate recall is NOT the bottleneck -- correcting an earlier claim.**
An earlier figure of 74.3% pool recall was measured through the funnel path
under *estimated* pose, and was repeatedly cited here as "a quarter of pairs
never have the true site available". A later experiment measured the pool
question directly, with pose fixed at ground truth to isolate it:

| arm | recall @5 px | median pool | p95 pool |
|---|---:|---:|---:|
| shipped harvest, 3 views | 0.900 | 413 | 4,902 |
| shipped, capped to 2000 | 0.850 | 413 | 2,000 |
| union of 4 views (adds DoG) | **0.950** | 493 | 6,685 |
| lattice-cell quota A/B/C | **0.950** | 322-396 | 5,208-5,870 |

Pool recall is **90-95%**, not 74%. The two numbers measure different things
(estimated vs ground-truth pose) and should never have been compared.

A no-quota ablation ties all three lattice-cell quota variants *exactly* --
same recall, same fixed pairs -- so the 90->95% gain comes from **adding DoG as
a fourth harvest view**, not from the cell-quota mechanism. The quota's real
measured effect is pool compression at equal recall (20-38% smaller, largest on
FinFET), and it does beat a naive fixed cap, which costs 5 pp of recall for the
same size target.

**The diagnostic is therefore unambiguous, from three independent angles:**

    detection   ~99-100%
    pool        ~90-95%
    selection    15-43%   <-- the bottleneck

Pool-high and selection-poor is the *identity-evidence* condition: not a
threshold, NMS, cap or dedup problem. Further harvest engineering is not where
the remaining points are.

**The nearest-centre result may be an implementation artefact.** The addendum
states nearest-to-centre as the official rule for equally-supported candidates,
and Set B deliberately includes pairs where global argmax and nearest-centre
disagree. My test applied the rule to the **entire correlation surface** at
margin 0.02 -- roughly 800,000 positions, many near-max on a periodic field --
which pulls the answer toward the image centre regardless of evidence. Phase 1
applies it *within a ranked candidate pool*. The measured harm (p=0.017) kills
my implementation, not necessarily the rule. It deserves a corrected retest.

**The sponsor baseline scores 0.50 on `p2_val`** per the G3 gate record, against
our best measured 0.28-0.33. Whether that gap is real or a sampling artifact was
never conclusively settled -- the control experiment was started twice and
stalled both times. It is worth one clean run, because a simple published
baseline outscoring us is the most actionable single fact available.

**This dataset is harder than the blind set by construction.** The gate report
records the matched-pose ZNCC ceiling at **0.40** on our data, with decoy losses
by design. Set A is specified as "noise comparable to the Phase 1 sample
prompt". 40/40 is not reachable on `p2_val`; the same code should score
materially higher on Set A.

---

## 6. Method lessons worth keeping

**Sample size.** Every conclusion drawn at n=25 carried a +/-18 pp interval. Two
independent shootouts of the *same* builder on *different* 25-pair subsets
returned 0.200 and 0.280 -- an 8-point swing from subset choice alone. Nothing
below n=120 should drive a decision.

**Caps applied by an untrusted score delete the answer.** This happened twice:
a 400-candidate pool cap and a 32-candidate screen cap. Both looked reasonable,
both silently discarded the true site, and the second cost an end-to-end score
of 2.9/40 that had nothing to do with the algorithm. The true site's rank by any
cheap score is in the hundreds. **A recall stage must not be truncated by the
score whose weakness is the reason the stage exists.**

**Progress came from subtraction, not addition.** Localization moved 0.072 ->
0.112 -> 0.280 by *removing* the funnel, the caps and the screening. Nothing
added -- seven selectors, Bayes sharpness, pose consistency, MRF rescoring,
retraining -- moved it at all.

**Agents parked on monitors.** Three subagents each burned ~100k tokens waiting
on background jobs and returned nothing. Single decisive measurements are
cheaper to run directly; agents earn their keep on genuinely parallel,
code-heavy work.

---

## 7. Game plan

Ordered by expected points per hour, with the deadline in mind.

### Priority 0 -- ship something valid (blocking, not optional)

**`register.py` still does not exist.** Every point above is theoretical until
there is an entrypoint that emits `pair_id, x, y, theta, scale, found, score`
for every pair. A missing or malformed row scores zero regardless of algorithm
quality, and this is the only artifact whose correctness can be fully verified
without a working solver.

Build it now against the current best configuration (dense sweep + DoG + 45
poses + refinement), wire the watchdog, make errors emit valid flagged rows
rather than fake rejections, and run `verify_offline.py` against it. From that
moment a submission exists and every later improvement is a swap-in behind a
fixed contract.

Also outstanding and cheap: the three organizer sample pairs have **not been
obtained** -- without them the `pairs.csv` parser is a guess.

### Priority 1 -- confirm and bank DoG

Re-run the DoG arm at n>=250 across `p2_val` and `p2_stress`, reported per
severity. If it holds, that is **9.0 -> 13.2 points** for a one-line change.

### Priority 2 -- the 25 points nobody has built

Rejection F1 and confidence AUC. Peak-score AUC is already 0.704, and
`presence.py` exists and is tested. This is the best points-per-hour work
remaining and it is *independent of localization quality* -- a well-calibrated
score on an 18%-accurate localizer still earns most of these points. Never
rejecting scores **zero** on the 15-point block.

### Priority 3 -- attack the harvester, not the ranker

The 92.2% sibling gap and 74.3% pool recall say the proposal stage, not the
ranking stage, is now the constraint. Raising pool recall raises the ceiling on
everything downstream. This supersedes further ranking work, which has been
measured as a dead end.

### Priority 4 -- new physical evidence (only if time remains)

Because the current feature set provably lacks the discriminating signal, the
only route left is evidence that is not currently computed: canonical unit-cell
defect fields, FinFET fin-pitch process modulation, line-edge-roughness
fingerprints. High ceiling, high cost, and it should not start until Priorities
0-2 are done.

### Priority 5 -- the evidence package

`failure_analysis.pdf` (2 pages, graded) has not been started, and the README
lacks the confidence-scale section the mentor explicitly asked for. This
document is most of the raw material. Ten points, largely writing.

### Not worth doing

- Subpixel precision -- localization is binary; hits are already at 1.00
- Finer pose grids -- 45 = 187 poses, measured
- Template filter tuning -- five filters, identical site selection
- Further ranker re-weighting -- features carry near-chance information
- Set D / RGB -- gated behind Sets A-C reaching 0.50, and cannot lift a ranking
  score above 100

---

## 8. Honest position

Localization is at 7.3/40 measured, or ~13/40 if DoG confirms. Pose is 20/20.
Rejection and confidence are 25 points sitting at zero with a working signal and
an untouched implementation path. Efficiency needs the median pulled from 6.75 s
under 5 s. The evidence package is 10 points of writing.

The realistic near-term total is meaningfully higher than the localization
number alone suggests, because five of the six scored blocks do not depend on
solving the periodic-alias problem. The single largest risk is not algorithmic:
it is that no `register.py` exists with three days left.
